from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field

from src.core.research_node import ResearchNode


class ResearchJournal(BaseModel):
    nodes: list[ResearchNode] = Field(default_factory=list)

    def append(self, node: ResearchNode) -> None:
        self.nodes.append(node)

    def best_node(self) -> ResearchNode | None:
        best: ResearchNode | None = None
        for node in self.nodes:
            if node.better_than(best):
                best = node
        return best

    # -- tree queries ------------------------------------------------------
    #
    # The journal stores a flat list, but the nodes form a forest through
    # `parent_id`. The search policy asks structural questions of it -- how many
    # roots exist, which leaves are broken -- so those questions belong here
    # rather than being re-derived by every caller.

    def node_by_id(self, node_id: str) -> ResearchNode | None:
        for node in self.nodes:
            if node.id == node_id:
                return node
        return None

    def roots(self) -> list[ResearchNode]:
        """Nodes with no parent: the independent drafts."""

        return [node for node in self.nodes if node.parent_id is None]

    def children_of(self, node_id: str) -> list[ResearchNode]:
        return [node for node in self.nodes if node.parent_id == node_id]

    def leaves(self) -> list[ResearchNode]:
        """Nodes nothing has been expanded from yet."""

        expanded = {node.parent_id for node in self.nodes if node.parent_id}
        return [node for node in self.nodes if node.id not in expanded]

    def buggy_leaves(self) -> list[ResearchNode]:
        """Failed leaves: the only failures a repair step can usefully target.

        A failure that already has a child has been acted on. Repairing it again
        would fork a second attempt from the same broken state.
        """

        return [node for node in self.leaves() if node.is_buggy]

    def successful_nodes(self) -> list[ResearchNode]:
        return [node for node in self.nodes if node.is_successful]

    def depth_of(self, node_id: str) -> int:
        """Edges from this node up to its root. Cycles cannot occur, but a
        malformed journal is treated as terminating rather than hanging."""

        depth = 0
        seen: set[str] = set()
        current = self.node_by_id(node_id)
        while current is not None and current.parent_id and current.parent_id not in seen:
            seen.add(current.id)
            depth += 1
            current = self.node_by_id(current.parent_id)
        return depth

    def nodes_for_iteration(self, iteration: int) -> list[ResearchNode]:
        return [node for node in self.nodes if node.iteration == iteration]

    def best_by_iteration(self) -> list[ResearchNode]:
        best_nodes: list[ResearchNode] = []
        iterations = sorted({node.iteration for node in self.nodes})
        for iteration in iterations:
            best: ResearchNode | None = None
            for node in self.nodes_for_iteration(iteration):
                if node.better_than(best):
                    best = node
            if best:
                best_nodes.append(best)
        return best_nodes

    def plateau_count(self, min_improvement: float = 0.0) -> int:
        best_nodes = self.best_by_iteration()
        if len(best_nodes) < 2:
            return 0
        count = 0
        incumbent = best_nodes[0]
        for node in best_nodes[1:]:
            if _meaningfully_better(node, incumbent, min_improvement):
                incumbent = node
                count = 0
            else:
                count += 1
        return count

    def save_json(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.model_dump(), indent=2, default=str), encoding="utf-8")

    def to_markdown(self) -> str:
        lines = ["# Autonomous Research Journal", ""]
        if not self.nodes:
            return "# Autonomous Research Journal\n\nNo autonomous research nodes were executed.\n"
        for node in self.nodes:
            lines.append(f"## Node {node.id}")
            lines.append(f"- parent: {node.parent_id or 'none'}")
            lines.append(f"- iteration: {node.iteration}")
            lines.append(f"- branch: {node.branch_index}")
            lines.append(f"- status: {node.status}")
            lines.append(f"- metric: {node.metric_name or 'n/a'} = {node.metric_value if node.metric_value is not None else 'n/a'}")
            lines.append(f"- plan: {node.plan}")
            if node.analysis:
                lines.append(f"- analysis: {node.analysis}")
            if node.work_dir:
                lines.append(f"- work_dir: `{node.work_dir}`")
            lines.append("")
        best = self.best_node()
        if best:
            lines.append("## Best Node")
            lines.append(f"- id: {best.id}")
            lines.append(f"- metric: {best.metric_name} = {best.metric_value}")
            lines.append(f"- status: {best.status}")
            lines.append("")
        return "\n".join(lines)


def _meaningfully_better(candidate: ResearchNode, incumbent: ResearchNode, min_improvement: float) -> bool:
    if not candidate.better_than(incumbent):
        return False
    if candidate.metric_value is None or incumbent.metric_value is None:
        return False
    delta = candidate.metric_value - incumbent.metric_value
    if candidate.maximize:
        return delta >= min_improvement
    return -delta >= min_improvement
