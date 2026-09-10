from __future__ import annotations

from src.core.research_journal import ResearchJournal
from src.core.search_strategy import SearchStrategy
from src.llm_client import LLMClient


class StrategyAgent:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def run(
        self,
        journal: ResearchJournal,
        iteration: int,
        initial_branches: list[str],
        max_branches: int,
        patience: int,
        min_improvement: float,
        directives: list[str] | None = None,
        prior_findings: str = "",
    ) -> SearchStrategy:
        if iteration == 0 or not journal.nodes:
            # The first iteration has no journal to reason from, which is exactly
            # where memory of earlier runs is worth the most.
            reasoning = "Initial autonomous search uses broad branch plans from the planner."
            plans = initial_branches[:max_branches]
            if prior_findings:
                reasoning += " Prior runs on a similar goal were available as context."
                plans = self._plans_with_prior_check(plans, prior_findings, max_branches)
            return SearchStrategy(iteration=iteration, branch_plans=plans, reasoning=reasoning)

        plateau_count = journal.plateau_count(min_improvement=min_improvement)
        best = journal.best_node()
        if plateau_count >= patience:
            return SearchStrategy(
                iteration=iteration,
                reasoning=f"No meaningful improvement for {plateau_count} iteration(s).",
                should_stop=True,
                stop_reason="plateau",
            )

        best_text = f"current best node {best.id} with {best.metric_name}={best.metric_value}" if best else "no successful best node"
        directive_plans = self._plans_from_directives(directives or [], best_text)
        if directive_plans:
            return SearchStrategy(
                iteration=iteration,
                branch_plans=directive_plans[:max_branches],
                reasoning=f"Adaptive strategy used refreshed LLM-agent directives; plateau_count={plateau_count}.",
            )
        plans = [
            f"Exploit {best_text}; increase proposed-method effect if supported; effect_range=1.25,4.75",
            f"Robustness check around {best_text}; moderate proposed-method effect; effect_range=0.60,3.20",
            f"Adversarial stress test for {best_text}; weak proposed-method effect; effect_range=0.05,1.50",
        ]
        if best and isinstance(best.metadata.get("method_comparison"), dict):
            comparison = best.metadata["method_comparison"]
            if comparison.get("supports_hypothesis"):
                plans[1] = f"Ablation of {best_text}; weaken proposed component to estimate contribution"
            else:
                plans[0] = f"Baseline strengthening around {best_text}; test whether proposed still wins against a stronger baseline"
        return SearchStrategy(
            iteration=iteration,
            branch_plans=plans[:max_branches],
            reasoning=f"Adaptive strategy selected branches from journal evidence; plateau_count={plateau_count}.",
        )

    @staticmethod
    def _plans_with_prior_check(plans: list[str], prior_findings: str, max_branches: int) -> list[str]:
        """Spend one branch slot re-testing what an earlier run failed to support.

        A previous run's negative result is a reason to look harder, not a
        reason to stop looking, so the retest is added as a branch rather than
        used to prune one. It is added only when a slot is free -- memory never
        displaces the planner's own branches.
        """

        if len(plans) >= max_branches:
            return plans[:max_branches]
        headline = next(
            (line.strip("- ").strip() for line in prior_findings.splitlines() if line.strip().startswith("-")),
            "",
        )
        if not headline:
            return plans[:max_branches]
        return (
            plans
            + [
                "Retest what an earlier run left unsupported, with a design that would"
                f" separate a real null from an underpowered test: {headline[:200]}"
            ]
        )[:max_branches]

    @staticmethod
    def _plans_from_directives(directives: list[str], best_text: str) -> list[str]:
        text = " ".join(directives[-3:]).lower()
        plans: list[str] = []
        if "baseline" in text:
            plans.append(f"Baseline strengthening around {best_text}; directive refreshed after observing journal")
        if "ablation" in text:
            plans.append(f"Ablation of {best_text}; directive refreshed after observing journal")
        if "stress" in text or "adversarial" in text:
            plans.append(f"Adversarial stress test for {best_text}; directive refreshed after observing journal")
        if "replication" in text or "repeat" in text or "seed" in text:
            plans.append(f"Robustness replication around {best_text}; directive refreshed after observing journal")
        return plans
