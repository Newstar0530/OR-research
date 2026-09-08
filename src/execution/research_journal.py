from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ResearchNode:
    plan: str
    parameters: dict[str, Any]
    method: str
    metric_name: str = "objective_gap"
    metric_value: float | None = None
    maximize: bool = False
    status: str = "pending"
    analysis: str = ""
    parent_id: str | None = None
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    created_at: float = field(default_factory=time.time)


class ResearchJournal:
    """Small OR-safe adaptation of AI-Scientist-style experiment journals."""

    def __init__(self) -> None:
        self.nodes: list[ResearchNode] = []

    def append(self, node: ResearchNode) -> None:
        self.nodes.append(node)

    def best_node(self) -> ResearchNode | None:
        candidates = [n for n in self.nodes if n.status == "success" and n.metric_value is not None]
        if not candidates:
            return None
        return min(candidates, key=lambda n: n.metric_value)

    def to_dict(self) -> dict[str, Any]:
        return {"nodes": [n.__dict__ for n in self.nodes]}

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2, default=str), encoding="utf-8")

    def render_markdown(self) -> str:
        lines = ["# Research Tree Journal", ""]
        best = self.best_node()
        if best:
            lines.append("## Best Node")
            lines.append(f"- id: `{best.id}`")
            lines.append(f"- method: {best.method}")
            lines.append(f"- {best.metric_name}: {best.metric_value}")
            lines.append(f"- parameters: `{best.parameters}`")
            lines.append("")
        lines.append("## Nodes")
        for node in self.nodes:
            lines.append(f"### {node.method} / {node.id[:8]}")
            lines.append(f"- status: {node.status}")
            lines.append(f"- plan: {node.plan}")
            lines.append(f"- parameters: `{node.parameters}`")
            lines.append(f"- metric: {node.metric_name}={node.metric_value}")
            if node.analysis:
                lines.append(f"- analysis: {node.analysis}")
            lines.append("")
        return "\n".join(lines)

