from __future__ import annotations

from src.llm_client import LLMClient
from src.schemas import AutomatedReview


class ReviewerAgent:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def run(self, final_report: str) -> AutomatedReview:
        return AutomatedReview(
            summary="The work is a useful human-in-the-loop research automation demo, but it should not be framed as a validated scientific contribution without further review.",
            strengths=["Inspectable artifacts", "Reproducible mock demo", "Clear caution around novelty and claims"],
            weaknesses=["Literature search is mocked", "Benchmark scope is narrow", "Mathematical model is a simple baseline"],
            questions_for_authors=[
                "Which verified papers establish the baseline landscape?",
                "How sensitive are results to instance generation distributions?",
                "Are the baselines sufficient for the intended venue or internal review?",
            ],
            soundness_score=6,
            novelty_score=4,
            technical_quality_score=7,
            reproducibility_score=8,
            presentation_score=7,
            overall_score=6,
            confidence_score=6,
            recommendation="borderline",
            required_revision_checklist=[
                "Run a real literature review before making novelty claims.",
                "Add more baselines and benchmark families.",
                "Have a human verify the mathematical notation and conclusions.",
            ],
        )

