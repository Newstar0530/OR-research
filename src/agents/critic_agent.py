from __future__ import annotations

from src.llm_client import LLMClient
from src.schemas import ModelCritique


class CriticAgent:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def run(self, model_markdown: str) -> ModelCritique:
        return ModelCritique(
            critical_issues=[
                "The formulation is still a scaffold; all sets, parameters, variables, and domains must be specialized to the actual OR problem.",
                "The objective direction and evaluation metric must be checked against the stated hypothesis before interpreting results.",
                "Novelty cannot be inferred from the model draft; verified literature comparison is mandatory.",
            ],
            minor_issues=[
                "Sensitivity dimensions are generic until benchmark data or domain parameters are provided.",
                "The draft should identify whether the problem is deterministic, stochastic, robust, dynamic, or network-based.",
            ],
            missing_constraints=[
                "Capacity/resource constraints may be missing.",
                "Boundary conditions and variable domains may be underspecified.",
                "Feasibility and service/reliability constraints may need explicit definitions.",
            ],
            questionable_assumptions=[
                "Synthetic data may not represent the target industrial system.",
                "A single baseline is usually insufficient for research claims.",
            ],
            suggested_revisions=[
                "Convert generic notation into domain-specific notation before thesis or paper use.",
                "Define benchmark generation, real data sources, and statistical comparison rules.",
                "Declare what would falsify the hypothesis.",
            ],
            human_verification_checklist=[
                "Check all notation and indices.",
                "Verify the objective matches the intended research question.",
                "Confirm all constraints needed for feasibility are present.",
                "Confirm selected baselines are fair and sufficient.",
                "Confirm conclusions are supported by repeated experiments.",
            ],
        )

