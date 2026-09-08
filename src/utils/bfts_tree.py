from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field


class ResearchTreeNode(BaseModel):
    node_id: str
    parent_id: str | None = None
    iteration: int
    branch_index: int
    status: str
    metric_value: float | None = None
    score: float
    selected_for_expansion: bool = False


class ResearchTreeReport(BaseModel):
    nodes: list[ResearchTreeNode] = Field(default_factory=list)
    best_node_id: str | None = None
    next_expansion_candidates: list[str] = Field(default_factory=list)

    def to_markdown(self) -> str:
        lines = ["# BFTS Research Tree", "", f"- nodes: {len(self.nodes)}", f"- best_node_id: `{self.best_node_id or 'none'}`", ""]
        lines.append("## Next Expansion Candidates")
        lines.extend(f"- `{item}`" for item in self.next_expansion_candidates or ["none"])
        lines.append("")
        lines.append("## Nodes")
        for node in sorted(self.nodes, key=lambda item: item.score, reverse=True):
            lines.append(f"### {node.node_id}")
            lines.append(f"- parent: `{node.parent_id or 'none'}`")
            lines.append(f"- iteration: {node.iteration}")
            lines.append(f"- branch: {node.branch_index}")
            lines.append(f"- status: {node.status}")
            lines.append(f"- metric_value: {node.metric_value if node.metric_value is not None else 'n/a'}")
            lines.append(f"- score: {node.score:.6g}")
            lines.append(f"- selected_for_expansion: {node.selected_for_expansion}")
            lines.append("")
        return "\n".join(lines)


def build_research_tree(journal_json: str | Path, output_dir: str | Path, maximize: bool = False) -> ResearchTreeReport:
    path = Path(journal_json)
    if not path.exists():
        report = ResearchTreeReport()
        return _write(report, output_dir)
    data = json.loads(path.read_text(encoding="utf-8"))
    raw_nodes = data.get("nodes", [])
    metric_values = [node.get("metric_value") for node in raw_nodes if isinstance(node.get("metric_value"), (int, float))]
    best_value = (max(metric_values) if maximize else min(metric_values)) if metric_values else None
    nodes = []
    for raw in raw_nodes:
        metric = raw.get("metric_value")
        success_bonus = 1.0 if raw.get("status") == "success" else -1.0
        if isinstance(metric, (int, float)) and best_value is not None:
            distance = metric - best_value if maximize else best_value - metric
            score = success_bonus + float(distance)
        else:
            score = success_bonus - 10.0
        nodes.append(
            ResearchTreeNode(
                node_id=raw.get("id", "unknown"),
                parent_id=raw.get("parent_id"),
                iteration=int(raw.get("iteration", 0)),
                branch_index=int(raw.get("branch_index", 0)),
                status=raw.get("status", "unknown"),
                metric_value=metric if isinstance(metric, (int, float)) else None,
                score=score,
            )
        )
    ranked = sorted(nodes, key=lambda item: item.score, reverse=True)
    for item in ranked[:3]:
        item.selected_for_expansion = True
    report = ResearchTreeReport(
        nodes=nodes,
        best_node_id=ranked[0].node_id if ranked else None,
        next_expansion_candidates=[item.node_id for item in ranked[:3]],
    )
    return _write(report, output_dir)


def _write(report: ResearchTreeReport, output_dir: str | Path) -> ResearchTreeReport:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    (root / "research_tree.json").write_text(report.model_dump_json(indent=2), encoding="utf-8")
    (root / "research_tree.md").write_text(report.to_markdown(), encoding="utf-8")
    return report
