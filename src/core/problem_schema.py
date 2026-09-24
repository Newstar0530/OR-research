from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


ProblemFamily = Literal[
    "optimization",
    "scheduling",
    "inventory",
    "network_analysis",
    "stochastic_models",
    "decision_analysis",
    "soft_computing",
    "heuristics_metaheuristics",
    "ai_for_ie",
]


class ResearchProblem(BaseModel):
    """Provider-neutral, domain-neutral description of an OR/IE research task."""

    title: str
    goal: str
    domain: str = "Industrial Engineering / Operations Research"
    problem_family: ProblemFamily = "optimization"
    hypothesis: str | None = None
    decision_scope: list[str] = Field(default_factory=list)
    system_entities: list[str] = Field(default_factory=list)
    objective_candidates: list[str] = Field(default_factory=list)
    constraint_candidates: list[str] = Field(default_factory=list)
    uncertainty_sources: list[str] = Field(default_factory=list)
    controllable_factors: list[str] = Field(default_factory=list)
    baseline_candidates: list[str] = Field(default_factory=list)
    available_data: list[str] = Field(default_factory=list)
    human_verification_needed: list[str] = Field(default_factory=list)


class ResearchArtifactManifest(BaseModel):
    """Inspectable artifacts expected from a complete human-in-the-loop run."""

    required_files: list[str] = Field(
        default_factory=lambda: [
            "config_used.yaml",
            "run_status.json",
            "run_status.md",
            "domain_selection.json",
            "benchmark_manifest.json",
            "benchmark_manifest.md",
            "solver_tool_report.json",
            "solver_tool_report.md",
            "recommended_solver_tools.json",
            "domain_profile.md",
            "idea_archive.json",
            "selected_idea.json",
            "novelty_report.md",
            "research_gap_analysis.md",
            "requirement_set.json",
            "model_draft.md",
            "requirement_coverage.json",
            "requirement_coverage.md",
            "model_ir.json",
            "model_ir.md",
            "model_export_manifest.json",
            "model_export_pyomo.py",
            "model_export_ortools.py",
            "model_export_diagnostics.md",
            "model_compile_report.json",
            "model_compile_report.md",
            "model_critique.md",
            "algorithm_plan.md",
            "generated_experiment.py",
            "best_experiment.py",
            "best_prompt.json",
            "best_notes.md",
            "best_plot.py",
            "workspace_lineage.json",
            "workspace_lineage.md",
            "bfts_frontier.json",
            "bfts_frontier.md",
            "bfts_policy_queue.json",
            "bfts_policy_queue.md",
            "repair_trace.json",
            "repair_trace.md",
            "plot_aggregation.json",
            "plot_aggregation.md",
            "execution_log.json",
            "experiment_journal.md",
            "results.csv",
            "result_evaluation.md",
            "statistical_evidence.json",
            "statistical_evidence.md",
            "research_tree.json",
            "research_tree.md",
            "sensitivity_report.md",
            "claim_check.md",
            "citation_report.json",
            "citation_report.md",
            "references.bib",
            "literature_grounding.json",
            "literature_grounding.md",
            "autonomy_readiness.json",
            "autonomy_readiness.md",
            "next_research_cycle_plan.json",
            "next_research_cycle_plan.md",
            "next_config_patch.yaml",
            "final_report.md",
            "final_paper.tex",
            "latex_compile_report.json",
            "latex_compile_report.md",
            "pdf_review_stub.json",
            "pdf_review_stub.md",
            "automated_review.md",
            "llm_interactions.json",
            "llm_interactions.md",
            "stage_order.json",
        ]
    )
    optional_dirs: list[str] = Field(default_factory=lambda: ["figures"])


def infer_problem_family(profile_name: str) -> ProblemFamily:
    allowed = set(ProblemFamily.__args__)  # type: ignore[attr-defined]
    return profile_name if profile_name in allowed else "optimization"  # type: ignore[return-value]
