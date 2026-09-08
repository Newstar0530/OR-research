from __future__ import annotations

from src.llm_client import LLMClient
from src.schemas import AlgorithmPlan, ResearchIdea


class AlgorithmAgent:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def run(self, idea: ResearchIdea, model_markdown: str) -> AlgorithmPlan:
        return AlgorithmPlan(
            exact_solver_plan="Use dynamic programming for exact optimal values on small integer-weight instances; use brute force only for tiny validation cases.",
            heuristic_plan="Sort items by value density, fill greedily, then try one-for-one replacement moves that improve value while preserving capacity.",
            baseline_methods=["Value-density greedy", "Exact dynamic programming"],
            pseudocode="1. Generate seeded instance\n2. Solve exact DP\n3. Run greedy\n4. Run repaired greedy\n5. Record objective, runtime, and gap",
            complexity_discussion="Exact DP is O(nC), suitable only when capacity is moderate. Greedy sorting is O(n log n); one-for-one repair is O(n^2).",
            stopping_criteria="Stop repair when no improving feasible one-for-one swap exists or after a fixed pass limit.",
            required_packages=["pandas", "numpy", "matplotlib"],
            evaluation_metrics=["objective_value", "runtime_seconds", "gap_percent", "selected_weight"],
        )

