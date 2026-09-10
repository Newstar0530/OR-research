from __future__ import annotations

from src.llm_client import LLMClient
from src.llm_errors import LLMCallError
from src.schemas import ResearchIdea


class PlannerAgent:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def run(self, idea: ResearchIdea, max_branches: int, directives: list[str] | None = None) -> list[str]:
        directive_branches = self._branches_from_directives(directives or [])
        if directive_branches:
            return directive_branches[:max_branches]
        if not self.llm.use_mock:
            try:
                payload = self.llm.chat_json(
                    "You are an OR research planner. Propose distinct executable experiment branches.",
                    (
                        "Return JSON with key branches, a list of short branch plans. "
                        f"Research idea: {idea.model_dump()}"
                    ),
                )
                branches = [str(item) for item in payload.get("branches", [])]
                if branches:
                    return branches[:max_branches]
            except LLMCallError as error:
                if error.is_permanent:
                    raise
            except Exception:
                pass
        defaults = [
            "Baseline comparison with conservative proposed-method settings.",
            "More aggressive proposed-method settings to test whether improvement survives larger effect sizes.",
            "Stress-test variant with altered seed and parameter scaling.",
        ]
        return defaults[:max_branches]

    @staticmethod
    def _branches_from_directives(directives: list[str]) -> list[str]:
        text = " ".join(directives).lower()
        branches: list[str] = []
        if "baseline" in text:
            branches.append("Baseline strengthening branch from LLM agent directive.")
        if "ablation" in text:
            branches.append("Ablation branch from LLM agent directive.")
        if "stress" in text or "adversarial" in text:
            branches.append("Adversarial stress test branch from LLM agent directive.")
        if "replication" in text or "repeat" in text or "seed" in text:
            branches.append("Robustness replication branch from LLM agent directive.")
        return branches
