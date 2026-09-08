from __future__ import annotations

from pathlib import Path

from src.core.template_registry import select_template
from src.llm_client import LLMClient
from src.schemas import AlgorithmPlan, ExperimentSpec, ResearchIdea


class ExperimentAgent:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def run(
        self,
        idea: ResearchIdea,
        plan: AlgorithmPlan,
        starter_template: str = "",
        domain_profile: str = "optimization",
        template_key: str | None = None,
    ) -> ExperimentSpec:
        if starter_template:
            return ExperimentSpec(
                code=starter_template,
                description="Experiment generated from the provided starter template.",
                expected_outputs=["results.csv", "figures/gap_by_size.png", "SUMMARY_JSON stdout"],
            )
        project_root = Path(__file__).resolve().parents[2]
        template = select_template(domain_profile, project_root, template_key=template_key)
        return ExperimentSpec(
            code=template.path.read_text(encoding="utf-8"),
            description=f"Registry-selected experiment scaffold: {template.key}. {template.description}",
            expected_outputs=["results.csv", "figures/", "SUMMARY_JSON stdout"],
        )
