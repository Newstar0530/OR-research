from __future__ import annotations

from src.core.research_journal import ResearchJournal
from src.llm_client import LLMClient


class EvidenceAgent:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def run(self, journal: ResearchJournal) -> str:
        lines = ["# Evidence Synthesis", ""]
        if not journal.nodes:
            lines.append("No autonomous experiments were executed.")
            return "\n".join(lines) + "\n"
        successful = [node for node in journal.nodes if node.is_successful]
        failed = [node for node in journal.nodes if not node.is_successful]
        lines.append(f"- Nodes executed: {len(journal.nodes)}")
        lines.append(f"- Successful nodes: {len(successful)}")
        lines.append(f"- Failed or invalid nodes: {len(failed)}")
        best = journal.best_node()
        if best:
            lines.append(f"- Best node: `{best.id}` with {best.metric_name}={best.metric_value}")
            lines.append(f"- Best-node interpretation: {best.analysis}")
            comparison = best.metadata.get("method_comparison")
            if isinstance(comparison, dict):
                lines.append(f"- Baseline method: {comparison.get('baseline_method', 'unknown')}")
                lines.append(f"- Proposed method: {comparison.get('proposed_method', 'unknown')}")
                lines.append(f"- Estimated improvement: {comparison.get('improvement', 'unknown')}")
                lines.append(f"- Supports hypothesis in this run: {comparison.get('supports_hypothesis', False)}")
        lines.append("")
        lines.append("## Search Dynamics")
        for node in journal.best_by_iteration():
            lines.append(f"- iteration {node.iteration}: best node `{node.id}`, {node.metric_name}={node.metric_value}")
        lines.append("")
        mutation_counts: dict[str, int] = {}
        for node in journal.nodes:
            mutation = node.metadata.get("mutation")
            if isinstance(mutation, dict):
                kind = str(mutation.get("kind", "unknown"))
                mutation_counts[kind] = mutation_counts.get(kind, 0) + 1
        if mutation_counts:
            lines.append("## Mutation Coverage")
            for kind, count in sorted(mutation_counts.items()):
                lines.append(f"- {kind}: {count}")
            lines.append("")
        lines.append("## Claim Boundary")
        lines.append("- This is computational evidence from generated experiments, not proof of novelty or correctness.")
        lines.append("- Any conclusion must be checked against model assumptions, literature, and stronger baselines.")
        return "\n".join(lines) + "\n"
