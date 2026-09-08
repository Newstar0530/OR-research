from __future__ import annotations

import json
import math
from pathlib import Path

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


class BFTSPolicyReport(BaseModel):
    objective_direction: str
    queue: list[BFTSQueueItem] = Field(default_factory=list)

    def to_markdown(self) -> str:
        lines = ["# BFTS Policy Queue", "", f"- objective_direction: {self.objective_direction}", ""]
        for item in self.queue:
            lines.append(f"## Node {item.node_id}")
            lines.append(f"- priority: {item.priority:.6g}")
            lines.append(f"- value_score: {item.value_score:.6g}")
            lines.append(f"- exploration_bonus: {item.exploration_bonus:.6g}")
            lines.append(f"- failure_penalty: {item.failure_penalty:.6g}")
            lines.append(f"- expansion_count: {item.expansion_count}")
            lines.append(f"- selected_for_expansion: {item.selected_for_expansion}")
            lines.append("")
        return "\n".join(lines)


def build_bfts_policy_queue(
    journal: ResearchJournal,
    output_dir: str | Path,
    objective_direction: str = "minimize",
    frontier_size: int = 3,
) -> BFTSPolicyReport:
    maximize = objective_direction == "maximize"
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
            )
        )
    items.sort(key=lambda item: item.priority, reverse=True)
    for item in items[:frontier_size]:
        item.selected_for_expansion = True
    report = BFTSPolicyReport(objective_direction=objective_direction, queue=items)
    root = Path(output_dir)
    (root / "bfts_policy_queue.json").write_text(json.dumps(report.model_dump(), indent=2), encoding="utf-8")
    (root / "bfts_policy_queue.md").write_text(report.to_markdown(), encoding="utf-8")
    return report
