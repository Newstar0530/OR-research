from __future__ import annotations

from pydantic import BaseModel, Field


class StageDefinition(BaseModel):
    name: str
    purpose: str
    required_inputs: list[str] = Field(default_factory=list)
    required_outputs: list[str] = Field(default_factory=list)
    success_checks: list[str] = Field(default_factory=list)
    human_gate: bool = False


class ResearchProtocol(BaseModel):
    """A reusable, inspectable OR research workflow."""

    name: str = "generic_or_human_in_the_loop_protocol"
    stages: list[StageDefinition]

    def stage_names(self) -> list[str]:
        return [stage.name for stage in self.stages]

    def to_markdown(self) -> str:
        lines = [f"# Research Protocol: {self.name}", ""]
        for index, stage in enumerate(self.stages, start=1):
            lines.append(f"## {index}. {stage.name}")
            lines.append(stage.purpose)
            if stage.required_inputs:
                lines.append("")
                lines.append("Inputs:")
                lines.extend(f"- {item}" for item in stage.required_inputs)
            if stage.required_outputs:
                lines.append("")
                lines.append("Outputs:")
                lines.extend(f"- {item}" for item in stage.required_outputs)
            if stage.success_checks:
                lines.append("")
                lines.append("Checks:")
                lines.extend(f"- {item}" for item in stage.success_checks)
            lines.append("")
        return "\n".join(lines)


def default_or_protocol() -> ResearchProtocol:
    return ResearchProtocol(
        stages=[
            StageDefinition(
                name="domain_classification",
                purpose="Map the topic to a reusable OR problem family and load its domain profile.",
                required_inputs=["research_goal"],
                required_outputs=["domain_selection.json", "domain_profile.md"],
                success_checks=["Selected profile is from the registry."],
            ),
            StageDefinition(
                name="idea_generation",
                purpose="Generate testable research ideas with assumptions, baselines, and expected outputs.",
                required_inputs=["research_goal", "domain_profile.md"],
                required_outputs=["idea_archive.json", "selected_idea.json"],
                success_checks=["Ideas include feasibility and risk scores."],
                human_gate=True,
            ),
            StageDefinition(
                name="novelty_checking",
                purpose="Check local literature and configured APIs without fabricating citations.",
                required_inputs=["selected_idea.json", "optional literature_dir"],
                required_outputs=["novelty_report.md"],
                success_checks=["Unknown literature is marked as unknown, not invented."],
                human_gate=True,
            ),
            StageDefinition(
                name="gap_synthesis",
                purpose="Synthesize inspectable research gaps from the topic, domain profile, and local literature.",
                required_inputs=["domain_profile.md", "literature_index.csv"],
                required_outputs=["research_gap_analysis.md"],
                success_checks=["Gaps distinguish evidence from speculation."],
                human_gate=True,
            ),
            StageDefinition(
                name="mathematical_modeling",
                purpose="Draft mathematical formulation candidates and mark uncertain parts.",
                required_inputs=["selected_idea.json", "optional template_path"],
                required_outputs=["model_draft.md"],
                success_checks=["Variables, objective, constraints, and assumptions are inspectable."],
                human_gate=True,
            ),
            StageDefinition(
                name="model_critique",
                purpose="Critique mathematical coherence, missing constraints, and verification needs.",
                required_inputs=["model_draft.md"],
                required_outputs=["model_critique.md"],
                success_checks=["Critical issues are separated from minor issues."],
                human_gate=True,
            ),
            StageDefinition(
                name="algorithm_proposal",
                purpose="Choose exact, heuristic, simulation, or statistical methods from the method registry.",
                required_inputs=["selected_idea.json", "model_draft.md"],
                required_outputs=["algorithm_plan.md"],
                success_checks=["Baselines and evaluation metrics are explicit."],
            ),
            StageDefinition(
                name="experiment_implementation",
                purpose="Generate or adapt executable code that satisfies the experiment contract.",
                required_inputs=["algorithm_plan.md", "template registry"],
                required_outputs=["generated_experiment.py"],
                success_checks=["Code writes results.csv and SUMMARY_JSON."],
            ),
            StageDefinition(
                name="experiment_execution_and_debugging",
                purpose="Run code in a subprocess with timeout and log all failures.",
                required_inputs=["generated_experiment.py"],
                required_outputs=["execution_log.json", "experiment_journal.md", "results.csv"],
                success_checks=["Runtime errors, infeasibility, and timeout are not hidden."],
            ),
            StageDefinition(
                name="sensitivity_analysis",
                purpose="Analyze result robustness using registered metrics and domain-relevant factors.",
                required_inputs=["results.csv", "metric registry"],
                required_outputs=["sensitivity_report.md", "result_evaluation.md"],
                success_checks=["Observed effects and unstable findings are reported."],
            ),
            StageDefinition(
                name="report_generation",
                purpose="Draft a cautious report with reproducibility notes and claim constraints.",
                required_inputs=["all previous artifacts"],
                required_outputs=["final_report.md", "claim_check.md"],
                success_checks=["Generated claims requiring verification are marked."],
                human_gate=True,
            ),
            StageDefinition(
                name="automated_review",
                purpose="Simulate reviewer feedback without treating it as acceptance or validation.",
                required_inputs=["final_report.md"],
                required_outputs=["automated_review.md"],
                success_checks=["Scores and required revisions are explicit."],
            ),
        ]
    )

