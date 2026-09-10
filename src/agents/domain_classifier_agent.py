"""Which domain profile fits the research goal, and how sure that really is.

Two problems, both about honesty rather than accuracy.

The classifier counts English keywords, so a goal written in Chinese matched
nothing at all -- and it still reported `confidence: 0.45`, a number that
sounds like a weak match rather than what it was: no match, defaulted. Zero
hits now reports zero confidence and says it defaulted.

And the vocabulary had no entry for bilevel programming, KKT, MPEC, QUBO or
quantum annealing, so an entire research area was invisible to it. Those terms
are added, in English and Chinese. There is still no domain *profile* for
bilevel or quantum optimization -- they fall to `optimization`, which is the
nearest honest answer -- so the result says which terms matched, and a reader
can see the profile was chosen on a partial match.
"""

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
    "optimization": [
        "optimization", "linear programming", "integer programming", "robust", "facility location",
        # Bilevel, complementarity and quantum-adjacent formulations. They have
        # no profile of their own yet, so they land here rather than nowhere.
        "bilevel", "bi-level", "kkt", "mpec", "mpcc", "complementarity", "big-m", "stackelberg",
        "interdiction", "qubo", "ising", "quantum", "annealing", "qaoa", "penalty method",
        "mixed integer", "0-1", "binary programming", "reformulation", "relaxation",
        # Traditional Chinese: the classifier is a substring match, so these work
        # the same way. A goal written in Chinese used to match nothing at all.
        "最佳化", "最佳化問題", "線性規劃", "整數規劃", "混合整數", "雙層規劃", "雙層",
        "量子", "退火", "轉化", "鬆弛",
    ],
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
        matched: dict[str, list[str]] = {profile: [] for profile in profiles}
        for profile, words in KEYWORDS.items():
            if profile not in scores:
                continue
            for word in words:
                if word.lower() in text:
                    matched[profile].append(word)
            scores[profile] += len(matched[profile]) * SPECIFICITY_WEIGHT.get(profile, 1)

        best = max(scores, key=scores.get) if scores else "optimization"
        best_score = scores.get(best, 0)
        if best_score == 0:
            # Nothing matched. Saying so is the useful answer; a confidence of
            # 0.45 reads as a weak match, which is a different claim entirely.
            return {
                "profile": "optimization",
                "confidence": 0.0,
                "matched_terms": [],
                "reason": (
                    "No keyword matched the research goal, so no domain was identified and the"
                    " run defaulted to `optimization`. This is a fallback, not a classification."
                ),
                "available_profiles": profiles,
            }

        runner_up = sorted(scores.values(), reverse=True)[1] if len(scores) > 1 else 0
        # Confidence reflects how far ahead the winner is, not a fixed number.
        # A goal that matches two domains equally is a genuinely ambiguous goal.
        confidence = round(min(0.9, 0.5 + 0.1 * (best_score - runner_up)), 2)
        return {
            "profile": best,
            "confidence": confidence,
            "matched_terms": sorted(matched[best]),
            "reason": (
                f"Keyword classifier selected `{best}` on {len(matched[best])} matching term(s)"
                f" ({', '.join(sorted(matched[best]))}); weighted score {best_score}"
                f" versus {runner_up} for the next best domain."
            ),
            "available_profiles": profiles,
        }
