from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Sequence

from pydantic import BaseModel, Field

from src.core.research_journal import ResearchJournal


class BFTSQueueItem(BaseModel):
    node_id: str
    priority: float
    value_score: float
    exploration_bonus: float
    failure_penalty: float
    expansion_count: int
    selected_for_expansion: bool = False
    #: How many branch slots the live search policy actually spent on this node.
    #: Zero for a node the policy ranked highly but never reached.
    policy_expansions: int = 0


class BFTSPolicyReport(BaseModel):
    objective_direction: str
    queue: list[BFTSQueueItem] = Field(default_factory=list)
    #: `observed` when the selections below are the ones the search really made,
    #: `predicted` when no decision record was supplied and the top of the queue
    #: is a hypothesis about what would be expanded next.
    selection_basis: str = "predicted"

    def to_markdown(self) -> str:
        lines = [
            "# BFTS Policy Queue",
            "",
            f"- objective_direction: {self.objective_direction}",
            f"- selection_basis: {self.selection_basis}",
            "",
        ]
        for item in self.queue:
            lines.append(f"## Node {item.node_id}")
            lines.append(f"- priority: {item.priority:.6g}")
            lines.append(f"- value_score: {item.value_score:.6g}")
            lines.append(f"- exploration_bonus: {item.exploration_bonus:.6g}")
            lines.append(f"- failure_penalty: {item.failure_penalty:.6g}")
            lines.append(f"- expansion_count: {item.expansion_count}")
            lines.append(f"- policy_expansions: {item.policy_expansions}")
            lines.append(f"- selected_for_expansion: {item.selected_for_expansion}")
            lines.append("")
        return "\n".join(lines)


def build_bfts_policy_queue(
    journal: ResearchJournal,
    output_dir: str | Path,
    objective_direction: str = "minimize",
    frontier_size: int = 3,
    expanded_parent_ids: Sequence[str] | None = None,
) -> BFTSPolicyReport:
    """Rank the journal's nodes for expansion.

    When `expanded_parent_ids` is supplied -- the parents the live search policy
    actually chose -- the report marks what was expanded rather than guessing
    what would be. Without it the report falls back to ranking the top
    `frontier_size` nodes, and says so, so the two are never confused.
    """

    maximize = objective_direction == "maximize"
    expansions = Counter(expanded_parent_ids or [])
    child_counts = {node.id: 0 for node in journal.nodes}
    for node in journal.nodes:
        if node.parent_id in child_counts:
            child_counts[node.parent_id] += 1
    items = []
    for node in journal.nodes:
        if node.metric_value is None:
            value = -1_000.0
        else:
            value = node.metric_value if maximize else -node.metric_value
        expansion_count = child_counts.get(node.id, 0)
        exploration = 1.0 / math.sqrt(1 + expansion_count)
        failure_penalty = 5.0 if node.status != "success" else 0.0
        priority = value + exploration - failure_penalty
        items.append(
            BFTSQueueItem(
                node_id=node.id,
                priority=priority,
                value_score=value,
                exploration_bonus=exploration,
                failure_penalty=failure_penalty,
                expansion_count=expansion_count,
                policy_expansions=expansions.get(node.id, 0),
            )
        )
    items.sort(key=lambda item: item.priority, reverse=True)
    if expanded_parent_ids is None:
        for item in items[:frontier_size]:
            item.selected_for_expansion = True
        basis = "predicted"
    else:
        for item in items:
            item.selected_for_expansion = item.policy_expansions > 0
        basis = "observed"
    report = BFTSPolicyReport(
        objective_direction=objective_direction, queue=items, selection_basis=basis
    )
    root = Path(output_dir)
    (root / "bfts_policy_queue.json").write_text(json.dumps(report.model_dump(), indent=2), encoding="utf-8")
    (root / "bfts_policy_queue.md").write_text(report.to_markdown(), encoding="utf-8")
    return report
