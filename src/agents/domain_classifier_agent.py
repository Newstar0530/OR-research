from __future__ import annotations

from src.domain_profiles import list_profiles
from src.llm_client import LLMClient


KEYWORDS = {
    "scheduling": ["schedule", "scheduling", "job shop", "flow shop", "tardiness", "makespan", "due date"],
    "inventory": ["inventory", "stock", "newsvendor", "eoq", "reorder", "safety stock", "shortage"],
    "network_analysis": [
        "network",
        "graph",
        "supply chain",
        "reliability",
        "chaos",
        "bullwhip",
        "centrality",
        "shortest path",
        "flow",
        "routing",
    ],
    "stochastic_models": ["stochastic", "random", "queue", "simulation", "uncertainty", "queueing", "monte carlo"],
    "decision_analysis": ["decision", "ahp", "mcdm", "multi-criteria", "preference", "utility"],
    "soft_computing": ["fuzzy", "soft computing", "genetic", "particle swarm", "neural", "evolutionary"],
    "heuristics_metaheuristics": ["heuristic", "metaheuristic", "local search", "simulated annealing", "tabu"],
    "ai_for_ie": ["machine learning", "ai", "prediction", "reinforcement learning", "industrial ai"],
    "optimization": ["optimization", "linear programming", "integer programming", "robust", "facility location"],
}
SPECIFICITY_WEIGHT = {
    "optimization": 1,
    "heuristics_metaheuristics": 2,
    "ai_for_ie": 2,
    "scheduling": 3,
    "inventory": 3,
    "network_analysis": 3,
    "stochastic_models": 3,
    "decision_analysis": 3,
    "soft_computing": 3,
}


class DomainClassifierAgent:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def run(self, research_goal: str, requested_mode: str = "auto") -> dict:
        profiles = list_profiles()
        if requested_mode and requested_mode != "auto" and requested_mode in profiles:
            return {"profile": requested_mode, "confidence": 1.0, "reason": "User-configured domain profile."}

        text = research_goal.lower()
        scores = {profile: 0 for profile in profiles}
        for profile, words in KEYWORDS.items():
            if profile in scores:
                hits = sum(1 for word in words if word.lower() in text)
                scores[profile] += hits * SPECIFICITY_WEIGHT.get(profile, 1)
        best = max(scores, key=scores.get) if scores else "optimization"
        if scores.get(best, 0) == 0:
            best = "optimization"
        return {
            "profile": best,
            "confidence": 0.75 if scores.get(best, 0) > 0 else 0.45,
            "reason": f"Keyword-based classifier selected {best}.",
            "available_profiles": profiles,
        }
