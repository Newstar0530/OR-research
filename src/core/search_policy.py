"""Decide, in the loop, what the next node should be.

Until now the tree was a story told afterwards. The controller kept a single
`parent_workspace` that moved to the best node at the end of each iteration, so
every branch in an iteration mutated the same parent and every failure was a
dead end -- a chain with a tree-shaped report bolted on. `bfts_policy_queue.md`
ranked nodes for expansion after the search had already finished expanding.

This module makes the ranking happen first and be acted upon. For each branch
slot the policy answers one question -- draft, debug, or improve, and from which
parent -- and records the candidates it weighed, so the trace is an audit of a
real decision rather than a description of one.

Three kinds, in the order the policy considers them:

* `draft` opens a new root from the base code. Until `num_drafts` roots exist
  the policy drafts, because a search with one root cannot escape a bad start.
* `debug` copies a failed leaf and hands it to the repair backend. Bounded by
  `max_debug_depth` so a branch cannot spend the whole budget re-fixing itself,
  and reached with probability `debug_probability` so a broken branch does not
  monopolise the search either.
* `improve` mutates the highest-priority successful node. Best-first: value
  first, with a bonus for nodes that have not been expanded yet.

The policy never invents a parent. If nothing is eligible it says so in the
decision's reason and falls back to a draft, which is always available.
"""

from __future__ import annotations

import random
from pathlib import Path

from pydantic import BaseModel, Field

from src.core.research_journal import ResearchJournal
from src.core.research_node import NodeKind, ResearchNode


class SearchPolicyConfig(BaseModel):
    """Knobs for the tree search, all with defensible defaults."""

    #: How many independent roots to open before improving any of them.
    num_drafts: int = 2
    #: A chain of repairs longer than this is abandoned rather than continued.
    max_debug_depth: int = 2
    #: Chance of choosing repair over improvement when both are available.
    debug_probability: float = 0.5
    #: Weight on the "never expanded" bonus in the improve ranking.
    exploration_weight: float = 1.0
    #: Fixed so a run can be replayed. Set to None for wall-clock randomness.
    seed: int | None = 0

    def validated(self) -> "SearchPolicyConfig":
        if self.num_drafts < 1:
            raise ValueError("num_drafts must be at least 1; a search needs a root.")
        if self.max_debug_depth < 0:
            raise ValueError("max_debug_depth cannot be negative.")
        if not 0.0 <= self.debug_probability <= 1.0:
            raise ValueError("debug_probability must be a probability in [0, 1].")
        return self


class CandidateScore(BaseModel):
    """One node the policy weighed, and why it scored what it scored."""

    node_id: str
    kind: NodeKind
    priority: float
    value_score: float = 0.0
    exploration_bonus: float = 0.0
    expansion_count: int = 0
    debug_depth: int = 0
    status: str = ""
    eligible: bool = True
    note: str = ""


class SearchDecision(BaseModel):
    """What the policy chose for one branch slot, and what it chose over."""

    iteration: int
    branch_index: int
    kind: NodeKind
    parent_id: str | None = None
    parent_work_dir: str | None = None
    depth: int = 0
    debug_depth: int = 0
    reason: str = ""
    considered: list[CandidateScore] = Field(default_factory=list)

    def to_markdown(self) -> str:
        lines = [
            f"### iteration {self.iteration}, branch {self.branch_index}: **{self.kind}**",
            "",
            f"- parent: `{self.parent_id or 'none (new root)'}`",
            f"- depth: {self.depth}",
            f"- debug_depth: {self.debug_depth}",
            f"- reason: {self.reason}",
        ]
        if self.considered:
            lines.append("- candidates weighed:")
            for item in self.considered:
                flag = "" if item.eligible else " (ineligible)"
                detail = f" -- {item.note}" if item.note else ""
                lines.append(
                    f"  - `{item.node_id}` [{item.status}] priority={item.priority:.6g}"
                    f" value={item.value_score:.6g} exploration={item.exploration_bonus:.6g}"
                    f" children={item.expansion_count}{flag}{detail}"
                )
        else:
            lines.append("- candidates weighed: none; the journal was empty")
        lines.append("")
        return "\n".join(lines)


class SearchPolicy:
    """Picks the next node kind and parent from the journal as it stands."""

    def __init__(self, config: SearchPolicyConfig | None = None, maximize: bool = False) -> None:
        self.config = (config or SearchPolicyConfig()).validated()
        self.maximize = maximize
        self._rng = random.Random(self.config.seed)
        self.decisions: list[SearchDecision] = []

    # -- scoring -----------------------------------------------------------

    def _value(self, node: ResearchNode) -> float:
        """Higher is better, whichever way the objective points."""

        if node.metric_value is None:
            return float("-inf")
        return node.metric_value if self.maximize else -node.metric_value

    def improve_priority(self, node: ResearchNode, expansion_count: int) -> tuple[float, float, float]:
        """Best-first priority: objective value plus an unexpanded-node bonus.

        The bonus decays as `1 / (1 + children)` rather than vanishing after one
        expansion, so a strong node can still be revisited -- it just stops
        crowding out its untried siblings.
        """

        value = self._value(node)
        exploration = self.config.exploration_weight / (1.0 + expansion_count)
        return value + exploration, value, exploration

    # -- the decision ------------------------------------------------------

    def decide(
        self,
        journal: ResearchJournal,
        iteration: int,
        branch_index: int,
        reserved_parent_ids: set[str] | None = None,
        root_work_dir: str | Path | None = None,
    ) -> SearchDecision:
        """Choose draft / debug / improve for one branch slot.

        `reserved_parent_ids` are parents already claimed by an earlier slot in
        this same iteration. They are skipped so sibling branches diverge, but
        only while an unclaimed alternative exists -- the policy would rather
        expand a parent twice than waste a slot.
        """

        reserved = set(reserved_parent_ids or set())
        config = self.config
        roots = journal.roots()

        if len(roots) < config.num_drafts:
            decision = SearchDecision(
                iteration=iteration,
                branch_index=branch_index,
                kind="draft",
                parent_work_dir=str(root_work_dir) if root_work_dir else None,
                reason=(
                    f"{len(roots)} of {config.num_drafts} draft root(s) exist, so this slot"
                    " opens another independent root from the base code."
                ),
                considered=[self._describe(journal, node) for node in roots],
            )
            return self._record(decision)

        debuggable = self._debuggable(journal, reserved)
        improvable = self._improvable(journal, reserved)
        considered = [self._describe(journal, node) for node in journal.nodes]

        wants_debug = bool(debuggable) and self._rng.random() < config.debug_probability
        if debuggable and (wants_debug or not improvable):
            target = debuggable[0]
            reason = (
                "repairing the most recent actionable failure"
                if wants_debug
                else "no successful node is free to improve, so the slot repairs a failure instead"
            )
            decision = SearchDecision(
                iteration=iteration,
                branch_index=branch_index,
                kind="debug",
                parent_id=target.id,
                parent_work_dir=target.work_dir,
                depth=target.depth + 1,
                debug_depth=target.debug_depth + 1,
                reason=(
                    f"{reason}: `{target.id}` failed with"
                    f" {target.exception_type or 'an unparsed error'}"
                    f" at debug_depth {target.debug_depth} of {config.max_debug_depth}."
                ),
                considered=considered,
            )
            return self._record(decision)

        if improvable:
            target, priority, value, exploration = improvable[0]
            decision = SearchDecision(
                iteration=iteration,
                branch_index=branch_index,
                kind="improve",
                parent_id=target.id,
                parent_work_dir=target.work_dir,
                depth=target.depth + 1,
                debug_depth=0,
                reason=(
                    f"best-first: `{target.id}` has the highest priority {priority:.6g}"
                    f" (value {value:.6g} + exploration {exploration:.6g})"
                    f" among {len(improvable)} free successful node(s)."
                ),
                considered=considered,
            )
            return self._record(decision)

        decision = SearchDecision(
            iteration=iteration,
            branch_index=branch_index,
            kind="draft",
            parent_work_dir=str(root_work_dir) if root_work_dir else None,
            reason=(
                "nothing is expandable -- no successful node is free to improve and no"
                " failure is within the debug depth limit -- so the slot opens a fresh root."
            ),
            considered=considered,
        )
        return self._record(decision)

    # -- candidate sets ----------------------------------------------------

    def _debuggable(self, journal: ResearchJournal, reserved: set[str]) -> list[ResearchNode]:
        """Failed leaves still within the debug depth limit, newest first.

        Nodes whose stderr yielded a recognisable exception come first: a repair
        step with a named exception has something to act on, one without does
        not.

        A node marked `repair_exhausted` is skipped outright. That flag is set
        when the repair backend declined to change anything, so offering the
        branch again would produce the same refusal for the rest of the budget.
        """

        candidates = [
            node
            for node in journal.buggy_leaves()
            if node.debug_depth < self.config.max_debug_depth
            and node.work_dir
            and node.id not in reserved
            and not node.metadata.get("repair_exhausted")
        ]
        candidates.sort(key=lambda node: (node.exception_type is None, -node.iteration, -node.branch_index))
        return candidates

    def _improvable(
        self, journal: ResearchJournal, reserved: set[str]
    ) -> list[tuple[ResearchNode, float, float, float]]:
        """Successful nodes ranked by priority, best first."""

        scored: list[tuple[ResearchNode, float, float, float]] = []
        for node in journal.nodes:
            if not node.is_successful or not node.work_dir or node.id in reserved:
                continue
            priority, value, exploration = self.improve_priority(node, len(journal.children_of(node.id)))
            scored.append((node, priority, value, exploration))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored

    def _describe(self, journal: ResearchJournal, node: ResearchNode) -> CandidateScore:
        expansions = len(journal.children_of(node.id))
        priority, value, exploration = self.improve_priority(node, expansions)
        eligible = True
        note = ""
        if node.is_buggy:
            exhausted = bool(node.metadata.get("repair_exhausted"))
            eligible = (
                node.debug_depth < self.config.max_debug_depth
                and bool(node.work_dir)
                and not exhausted
                and expansions == 0
            )
            if eligible:
                note = "debug candidate"
            elif exhausted:
                note = "the repair backend already declined to change this branch"
            elif expansions:
                note = "already repaired once; a second fork starts from the same broken state"
            elif not node.work_dir:
                note = "no workspace on disk to repair"
            else:
                note = "debug depth limit reached"
            priority, value, exploration = float("-inf"), float("-inf"), 0.0
        elif not node.is_successful:
            eligible = False
            note = f"status `{node.status}` carries no metric"
        return CandidateScore(
            node_id=node.id,
            kind=node.kind,
            priority=priority,
            value_score=value,
            exploration_bonus=exploration,
            expansion_count=expansions,
            debug_depth=node.debug_depth,
            status=node.status,
            eligible=eligible,
            note=note,
        )

    def _record(self, decision: SearchDecision) -> SearchDecision:
        self.decisions.append(decision)
        return decision

    # -- reporting ---------------------------------------------------------

    def trace_markdown(self) -> str:
        lines = ["# Search Policy Trace", ""]
        lines.append(f"- objective_direction: {'maximize' if self.maximize else 'minimize'}")
        lines.append(f"- num_drafts: {self.config.num_drafts}")
        lines.append(f"- max_debug_depth: {self.config.max_debug_depth}")
        lines.append(f"- debug_probability: {self.config.debug_probability}")
        lines.append(f"- seed: {self.config.seed}")
        lines.append("")
        if not self.decisions:
            lines.append("No branch slots were filled.")
            return "\n".join(lines) + "\n"
        counts: dict[str, int] = {}
        for decision in self.decisions:
            counts[decision.kind] = counts.get(decision.kind, 0) + 1
        lines.append("## Decisions by kind")
        lines.append("")
        lines.extend(f"- {kind}: {count}" for kind, count in sorted(counts.items()))
        lines.append("")
        lines.append("## Decisions in order")
        lines.append("")
        lines.extend(decision.to_markdown() for decision in self.decisions)
        return "\n".join(lines)
