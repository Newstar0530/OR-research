"""What the previous runs learned, in a form the next run can read.

`append_memory` already wrote a line per run, but nothing ever read it back, so
every run started from nothing: the same dead end could be re-explored on
Monday and again on Tuesday with no record that Monday had already tried it.
This module closes that loop.

Two honesty rules govern everything below.

First, a prior run is *evidence about that run*, not a proof about the world. A
hypothesis that failed once is recorded as "was not supported in run X", never
as "is false", and the rendered brief says so in as many words. A search that
treats one non-significant result as settled is a search that stops looking.

Second, the record stores what was measured, not what was hoped. `best_metric`
comes from the journal, `supports_improvement_claim` from the statistical
verdict, `verification_failures` from the nodes that failed their contract. A
run with no evidence produces a record that says so rather than an empty one
that reads like a clean result.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

from pydantic import BaseModel, Field


MEMORY_DIRNAME = "research_memory"
RUNS_FILENAME = "runs.jsonl"

#: Words too common to say anything about what a study was about.
_STOPWORDS = frozenset(
    """
    a an and are as at be by for from has have how in into is it its of on or that the
    their there these this to was were what when which who will with using use used
    study research problem method methods approach approaches model models analysis
    """.split()
)


def append_memory(project_root: Path, entry_type: str, payload: dict[str, Any]) -> Path:
    """Append one timestamped record to `research_memory/<entry_type>.jsonl`."""

    memory_dir = Path(project_root) / MEMORY_DIRNAME
    memory_dir.mkdir(parents=True, exist_ok=True)
    path = memory_dir / f"{entry_type}.jsonl"
    record = {"timestamp": datetime.now().isoformat(timespec="seconds"), **payload}
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    return path


class RunRecord(BaseModel):
    """One finished run, reduced to what a later run would want to know."""

    timestamp: str = ""
    project_name: str = ""
    research_goal: str = ""
    run_dir: str = ""
    domain: str = ""
    primary_metric: str = ""
    objective_direction: str = "minimize"
    best_metric: float | None = None
    best_node_id: str | None = None
    best_node_status: str = ""
    node_count: int = 0
    successful_nodes: int = 0
    failed_nodes: int = 0
    #: The statistical verdict, e.g. `strong_preliminary`, `inconclusive_not_significant`.
    evidence_strength: str = ""
    #: True only when the comparison actually cleared its significance test.
    supports_improvement_claim: bool = False
    #: Nodes that ran but failed verification or the experiment contract.
    verification_failures: list[str] = Field(default_factory=list)
    #: Mutation kinds that appeared on nodes better than their parent.
    helpful_mutations: list[str] = Field(default_factory=list)
    #: Mutation kinds that appeared only on nodes no better than their parent.
    unhelpful_mutations: list[str] = Field(default_factory=list)
    #: Hypotheses this run tested and did not support. Not disproved -- unsupported.
    refuted_hypotheses: list[str] = Field(default_factory=list)
    review_overall_score: float | None = None
    notes: list[str] = Field(default_factory=list)

    def one_line(self) -> str:
        metric = (
            f"{self.primary_metric}={self.best_metric:.6g}"
            if self.best_metric is not None
            else f"{self.primary_metric or 'metric'}=not measured"
        )
        claim = "supported" if self.supports_improvement_claim else "not supported"
        return (
            f"{self.project_name or 'run'} ({self.timestamp or 'undated'}): {metric},"
            f" improvement claim {claim}, evidence `{self.evidence_strength or 'unrecorded'}`"
        )


def record_run(project_root: str | Path, record: RunRecord) -> Path:
    """Write a run record to the project's memory."""

    payload = record.model_dump()
    payload.pop("timestamp", None)
    return append_memory(Path(project_root), "runs", payload)


def load_run_records(project_root: str | Path, limit: int | None = None) -> list[RunRecord]:
    """Read run records back, newest last.

    A line this version cannot parse is skipped rather than crashing the run
    that is trying to learn from it. Memory is a convenience; losing one line of
    it must never cost a run.
    """

    path = Path(project_root) / MEMORY_DIRNAME / RUNS_FILENAME
    if not path.exists():
        return []
    records: list[RunRecord] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        try:
            records.append(RunRecord(**payload))
        except Exception:
            continue
    if limit is not None:
        return records[-limit:]
    return records


def _tokens(text: str) -> set[str]:
    words = re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", (text or "").lower())
    return {word for word in words if word not in _STOPWORDS}


def goal_similarity(goal_a: str, goal_b: str) -> float:
    """Jaccard overlap of content words, in [0, 1].

    Deliberately crude. It decides which prior runs are worth showing, not what
    is true, and a transparent rule that a reader can check beats an opaque one
    that is slightly better at ranking.
    """

    left, right = _tokens(goal_a), _tokens(goal_b)
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def find_related_runs(
    project_root: str | Path,
    research_goal: str,
    limit: int = 5,
    min_similarity: float = 0.08,
    exclude_run_dirs: Iterable[str] = (),
) -> list[tuple[RunRecord, float]]:
    """Prior runs whose goal overlaps this one, most similar first."""

    excluded = {str(item) for item in exclude_run_dirs}
    scored = [
        (record, goal_similarity(research_goal, record.research_goal))
        for record in load_run_records(project_root)
        if record.run_dir not in excluded
    ]
    scored = [item for item in scored if item[1] >= min_similarity]
    scored.sort(key=lambda item: (item[1], item[0].timestamp), reverse=True)
    return scored[:limit]


class PriorFindings(BaseModel):
    """The brief handed to the agents that plan the next run."""

    research_goal: str = ""
    matched: list[RunRecord] = Field(default_factory=list)
    similarities: list[float] = Field(default_factory=list)
    total_runs_in_memory: int = 0

    @property
    def is_empty(self) -> bool:
        return not self.matched

    def best_prior_metric(self) -> float | None:
        """The best metric any related run reached, in its own direction."""

        values = [
            (record.best_metric, record.objective_direction)
            for record in self.matched
            if record.best_metric is not None
        ]
        if not values:
            return None
        maximize = values[0][1] == "maximize"
        numbers = [value for value, _ in values]
        return max(numbers) if maximize else min(numbers)

    def to_prompt(self, max_runs: int = 3) -> str:
        """Compact text for an agent prompt. Empty string when there is nothing."""

        if self.is_empty:
            return ""
        lines = [
            "Prior runs on a similar goal (evidence from those runs, not established fact):",
        ]
        for record, score in list(zip(self.matched, self.similarities))[:max_runs]:
            lines.append(f"- [{score:.2f} goal overlap] {record.one_line()}")
            if record.refuted_hypotheses:
                lines.append(
                    "  - not supported there (worth retesting, not worth assuming): "
                    + "; ".join(record.refuted_hypotheses[:3])
                )
            if record.helpful_mutations:
                lines.append("  - changes that helped there: " + ", ".join(record.helpful_mutations[:5]))
            if record.verification_failures:
                lines.append(
                    f"  - {len(record.verification_failures)} node(s) failed verification there"
                )
        lines.append(
            "Do not treat any of the above as settled. A single unsupported result is"
            " weak evidence; prefer a design that would distinguish the two outcomes."
        )
        return "\n".join(lines)

    def to_markdown(self) -> str:
        lines = ["# Prior Findings", ""]
        lines.append(f"- research_goal: {self.research_goal}")
        lines.append(f"- runs in memory: {self.total_runs_in_memory}")
        lines.append(f"- related runs matched: {len(self.matched)}")
        lines.append("")
        if self.is_empty:
            lines.append(
                "No prior run in this project's memory is close enough to this goal to be"
                " informative. This run starts without carried-over evidence."
            )
            return "\n".join(lines) + "\n"
        best = self.best_prior_metric()
        if best is not None:
            lines.append(f"- best metric previously reached on a similar goal: {best:.6g}")
            lines.append("")
        for record, score in zip(self.matched, self.similarities):
            lines.append(f"## {record.project_name or 'unnamed run'} ({record.timestamp or 'undated'})")
            lines.append(f"- goal overlap: {score:.3f}")
            lines.append(f"- goal: {record.research_goal}")
            lines.append(f"- run_dir: `{record.run_dir}`")
            lines.append(
                f"- best {record.primary_metric or 'metric'}: "
                f"{record.best_metric if record.best_metric is not None else 'not measured'}"
                f" ({record.objective_direction})"
            )
            lines.append(f"- nodes: {record.node_count} ({record.successful_nodes} succeeded, {record.failed_nodes} failed)")
            lines.append(f"- evidence_strength: {record.evidence_strength or 'unrecorded'}")
            lines.append(f"- supports_improvement_claim: {record.supports_improvement_claim}")
            if record.helpful_mutations:
                lines.append("- changes that preceded an improvement: " + ", ".join(record.helpful_mutations))
            if record.unhelpful_mutations:
                lines.append("- changes that did not: " + ", ".join(record.unhelpful_mutations))
            if record.refuted_hypotheses:
                lines.append("- tested and not supported there:")
                lines.extend(f"  - {item}" for item in record.refuted_hypotheses)
            if record.verification_failures:
                lines.append("- verification failures:")
                lines.extend(f"  - {item}" for item in record.verification_failures[:10])
            if record.notes:
                lines.extend(f"- note: {note}" for note in record.notes)
            lines.append("")
        lines.append(
            "These are observations from earlier runs of this project. They are a reason"
            " to design a sharper test, not a reason to skip one."
        )
        lines.append("")
        return "\n".join(lines)


def build_prior_findings(
    project_root: str | Path,
    research_goal: str,
    limit: int = 5,
    min_similarity: float = 0.08,
    exclude_run_dirs: Iterable[str] = (),
) -> PriorFindings:
    matches = find_related_runs(
        project_root,
        research_goal,
        limit=limit,
        min_similarity=min_similarity,
        exclude_run_dirs=exclude_run_dirs,
    )
    return PriorFindings(
        research_goal=research_goal,
        matched=[record for record, _ in matches],
        similarities=[score for _, score in matches],
        total_runs_in_memory=len(load_run_records(project_root)),
    )


def write_prior_findings(run_dir: str | Path, findings: PriorFindings) -> Path:
    root = Path(run_dir)
    root.mkdir(parents=True, exist_ok=True)
    (root / "prior_findings.json").write_text(findings.model_dump_json(indent=2), encoding="utf-8")
    path = root / "prior_findings.md"
    path.write_text(findings.to_markdown(), encoding="utf-8")
    return path


def summarize_journal(journal, primary_metric: str = "", objective_direction: str = "minimize") -> dict[str, Any]:
    """Reduce a finished journal to the fields of a `RunRecord`.

    Takes the journal duck-typed so this module does not import the search
    stack; memory should never be the reason a run fails to start.
    """

    nodes = list(getattr(journal, "nodes", []) or [])
    best = journal.best_node() if hasattr(journal, "best_node") else None
    successful = [node for node in nodes if getattr(node, "is_successful", False)]
    failed = [node for node in nodes if getattr(node, "is_buggy", False)]
    helpful, unhelpful = _mutation_verdicts(journal, nodes)
    return {
        "primary_metric": primary_metric,
        "objective_direction": objective_direction,
        "best_metric": getattr(best, "metric_value", None) if best else None,
        "best_node_id": getattr(best, "id", None) if best else None,
        "best_node_status": getattr(best, "status", "") if best else "",
        "node_count": len(nodes),
        "successful_nodes": len(successful),
        "failed_nodes": len(failed),
        "verification_failures": [
            f"{node.id}: {node.failure_summary or node.status}" for node in failed
        ][:20],
        "helpful_mutations": helpful,
        "unhelpful_mutations": unhelpful,
    }


def _mutation_verdicts(journal, nodes: Sequence[Any]) -> tuple[list[str], list[str]]:
    """Which mutation kinds preceded an improvement over the node's own parent.

    Compared against the parent rather than against the run's best, because a
    mutation is credited for the change it made, not for the branch it happened
    to land on.
    """

    helped: set[str] = set()
    did_not: set[str] = set()
    lookup = {getattr(node, "id", None): node for node in nodes}
    for node in nodes:
        mutation = (getattr(node, "metadata", {}) or {}).get("mutation")
        if not isinstance(mutation, dict):
            continue
        kind = str(mutation.get("kind") or "").strip()
        if not kind:
            continue
        parent = lookup.get(getattr(node, "parent_id", None))
        if not getattr(node, "is_successful", False):
            did_not.add(kind)
            continue
        if parent is None:
            continue
        if node.better_than(parent):
            helped.add(kind)
        else:
            did_not.add(kind)
    return sorted(helped), sorted(did_not - helped)
