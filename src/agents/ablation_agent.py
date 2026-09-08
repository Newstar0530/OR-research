from __future__ import annotations

from src.core.research_journal import ResearchJournal
from src.llm_client import LLMClient


class AblationAgent:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def run(self, journal: ResearchJournal) -> str:
        best = journal.best_node()
        lines = ["# Ablation and Robustness Plan", ""]
        if best is None:
            lines.append("No successful node is available for ablation planning.")
        else:
            lines.append(f"Best node `{best.id}` should be stress-tested before any strong claim.")
            comparison = best.metadata.get("method_comparison")
            if isinstance(comparison, dict):
                lines.append(
                    f"Current estimated improvement is {comparison.get('improvement', 'unknown')} "
                    f"for `{comparison.get('proposed_method', 'proposed')}` over `{comparison.get('baseline_method', 'baseline')}`."
                )
            lines.append("")
            lines.append("Recommended checks:")
            lines.append("- Repeat the best variant over additional random seeds.")
            lines.append("- Disable or weaken the proposed component to estimate its marginal contribution.")
            lines.append("- Sweep the main problem-size or uncertainty parameter.")
            lines.append("- Compare against at least one stronger baseline if available.")
            lines.append("- Report confidence intervals or distributional summaries, not only means.")
            lines.append("- Inspect `mutation_trace.md` to verify which experimental changes were actually applied.")
        lines.append("")
        lines.append("Human verification is required before treating these checks as sufficient evidence.")
        return "\n".join(lines) + "\n"
