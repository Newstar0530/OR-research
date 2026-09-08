from __future__ import annotations

from src.llm_client import LLMClient
from src.schemas import AlgorithmPlan, ModelCritique, ReportDraft, ResearchIdea
from src.utils.prompt_utils import human_verification_footer


class ReportAgent:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def run(self, idea: ResearchIdea, model: str, critique: ModelCritique, algorithm: AlgorithmPlan, sensitivity: str) -> ReportDraft:
        markdown = f"""# {idea.title}

## Abstract
This draft reports an inspectable IE/OR research-assistant run. It proposes a model, algorithm plan, executable experiment scaffold, sensitivity analysis, and automated review. The findings are preliminary and require human verification.

## Introduction
Human-in-the-loop research automation can help organize ideas, assumptions, computational experiments, and review feedback while keeping intermediate artifacts inspectable.

## Research Gap
Needs human verification: the system may identify candidate gaps, but it cannot prove novelty without verified literature review.

## Hypothesis
{idea.core_hypothesis}

## Mathematical Model
{model}

## Model Critique Summary
Critical issues:
{chr(10).join(f"- {item}" for item in critique.critical_issues)}

## Algorithm
{algorithm.heuristic_plan}

## Experimental Setup
The generated experiment is required to write `results.csv`, include at least two methods when possible, record runtime and seed information, and produce machine-readable `SUMMARY_JSON` output.

## Results
See `results.csv`, `result_evaluation.md`, and generated figures in the run folder. Interpret all aggregate claims cautiously.

## Sensitivity Analysis
{sensitivity}

## Discussion
The pipeline demonstrates structured research assistance, not autonomous scientific validation.

## Limitations
- The formulation may still be generic.
- Literature novelty has not been externally verified unless configured APIs and human checks are used.
- Generated experiments may be synthetic and should not be overgeneralized.

## Future Work
- Replace generic scaffolds with validated domain models.
- Add real benchmark data.
- Expand baselines and statistical tests.
- Review all notation, assumptions, and claims with a human researcher.

## Reproducibility Notes
All stages write intermediate files into the timestamped run directory.
"""
        return ReportDraft(markdown=markdown + human_verification_footer())

