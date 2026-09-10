from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field


#: `unknown` is the honest value when nothing assessed the risk. Defaulting
#: an unassessed similarity risk to "medium" reads as a finding.
RiskLevel = Literal["unknown", "low", "medium", "high"]
NoveltyRecommendation = Literal["keep", "revise", "reject"]
ExecutionStatus = Literal["success", "failed", "timeout"]
ReviewRecommendation = Literal["accept", "weak accept", "borderline", "weak reject", "reject"]


class ResearchIdea(BaseModel):
    title: str
    problem_context: str
    core_hypothesis: str
    expected_contribution: str
    proposed_model_type: str
    proposed_algorithm_type: str
    experimental_plan: str
    interestingness_score: int = Field(ge=1, le=10)
    novelty_score: int = Field(ge=1, le=10)
    feasibility_score: int = Field(ge=1, le=10)
    risk_score: int = Field(ge=1, le=10)
    assumptions: list[str]
    required_data: list[str]
    expected_outputs: list[str]


class IdeaArchive(BaseModel):
    ideas: list[ResearchIdea]
    selected_index: int = 0


class NoveltyReport(BaseModel):
    novelty_summary: str
    search_queries: list[str]
    potentially_related_works: list[str]
    similarity_risk: RiskLevel
    recommendation: NoveltyRecommendation
    revision_suggestions: list[str]


class ModelDraft(BaseModel):
    markdown: str


class ModelCritique(BaseModel):
    critical_issues: list[str]
    minor_issues: list[str]
    missing_constraints: list[str]
    questionable_assumptions: list[str]
    suggested_revisions: list[str]
    human_verification_checklist: list[str]


class AlgorithmPlan(BaseModel):
    exact_solver_plan: str
    heuristic_plan: str
    baseline_methods: list[str]
    pseudocode: str
    complexity_discussion: str
    stopping_criteria: str
    required_packages: list[str]
    evaluation_metrics: list[str]


class ExperimentSpec(BaseModel):
    code: str
    description: str
    expected_outputs: list[str]


class ExecutionResult(BaseModel):
    status: ExecutionStatus
    returncode: int | None
    stdout: str
    stderr: str
    duration_seconds: float
    timed_out: bool = False
    metrics: dict[str, Any] = Field(default_factory=dict)
    log_path: Path | None = None


class DebugReport(BaseModel):
    debug_attempts: int
    final_status: ExecutionStatus
    remaining_errors: str
    patch_summary: list[str]


class SensitivityReport(BaseModel):
    markdown: str
    tested_parameters: list[str]
    observed_effects: list[str]


class ReportDraft(BaseModel):
    markdown: str


class AutomatedReview(BaseModel):
    summary: str
    strengths: list[str]
    weaknesses: list[str]
    questions_for_authors: list[str]
    # Scores are optional because a review that did not happen must not report
    # one. The previous version returned a fixed overall_score of 6 without
    # reading the report, and that constant was carried into cross-run memory
    # and the readiness assessment as if it meant something.
    soundness_score: int | None = Field(default=None, ge=1, le=10)
    novelty_score: int | None = Field(default=None, ge=1, le=10)
    technical_quality_score: int | None = Field(default=None, ge=1, le=10)
    reproducibility_score: int | None = Field(default=None, ge=1, le=10)
    presentation_score: int | None = Field(default=None, ge=1, le=10)
    overall_score: int | None = Field(default=None, ge=1, le=10)
    confidence_score: int | None = Field(default=None, ge=1, le=10)
    recommendation: ReviewRecommendation
    required_revision_checklist: list[str]
    #: True only when a model actually read the report and produced the scores.
    review_performed: bool = False


class ExperimentLogEntry(BaseModel):
    timestamp: datetime
    stage: str
    status: str
    code_path: str | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)
    plots: list[str] = Field(default_factory=list)
    notes: str = ""

