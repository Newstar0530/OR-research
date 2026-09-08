from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from src.config import AppConfig
from src.llm_client import LLMClient
from src.utils.autonomy_readiness import AutonomyReadinessReport


RemediationActionType = Literal[
    "increase_replication",
    "strengthen_baselines",
    "add_diagnostics",
    "strengthen_novelty_evidence",
    "verify_claims",
    "resolve_human_gate",
    "fix_blocker",
    "extend_sensitivity",
]


class RemediationAction(BaseModel):
    priority: int = Field(ge=1, le=5)
    action_type: RemediationActionType
    issue: str
    recommended_action: str
    rationale: str
    suggested_config_changes: dict[str, int | float | str | bool] = Field(default_factory=dict)
    human_required: bool = False


class ReadinessRemediationPlan(BaseModel):
    readiness_score: int
    readiness_level: str
    can_autorun_without_human: bool
    next_run_config_patch: dict[str, int | float | str | bool] = Field(default_factory=dict)
    actions: list[RemediationAction] = Field(default_factory=list)

    def to_markdown(self) -> str:
        lines = [
            "# Next Research Cycle Plan",
            "",
            f"- readiness_score: {self.readiness_score}/100",
            f"- readiness_level: {self.readiness_level}",
            f"- can_autorun_without_human: {self.can_autorun_without_human}",
            "",
            "## Suggested Config Patch",
        ]
        if self.next_run_config_patch:
            for key, value in self.next_run_config_patch.items():
                lines.append(f"- `{key}`: `{value}`")
        else:
            lines.append("- No automatic config changes suggested.")
        lines.append("")
        lines.append("## Actions")
        if not self.actions:
            lines.append("- No remediation actions were generated.")
        for action in sorted(self.actions, key=lambda item: item.priority):
            lines.append(f"### P{action.priority} {action.action_type}")
            lines.append(f"- issue: {action.issue}")
            lines.append(f"- recommended_action: {action.recommended_action}")
            lines.append(f"- rationale: {action.rationale}")
            lines.append(f"- human_required: {action.human_required}")
            if action.suggested_config_changes:
                lines.append("- suggested_config_changes:")
                for key, value in action.suggested_config_changes.items():
                    lines.append(f"  - `{key}`: `{value}`")
            lines.append("")
        lines.append("## Safety Note")
        lines.append(
            "Only actions with human_required=false are candidates for an automatic next run. "
            "Literature claims, flagged novelty statements, and human gate requests must be reviewed by a researcher."
        )
        return "\n".join(lines) + "\n"

    def config_patch_yaml(self) -> str:
        if not self.next_run_config_patch:
            return "# No automatic config patch suggested.\n"
        return "\n".join(f"{key}: {self._yaml_value(value)}" for key, value in self.next_run_config_patch.items()) + "\n"

    @staticmethod
    def _yaml_value(value: int | float | str | bool) -> str:
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, str):
            return f'"{value}"'
        return str(value)


class ReadinessRemediationAgent:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def run(self, readiness: AutonomyReadinessReport, config: AppConfig) -> ReadinessRemediationPlan:
        actions: list[RemediationAction] = []

        for blocker in readiness.blockers:
            actions.append(
                RemediationAction(
                    priority=1,
                    action_type="fix_blocker",
                    issue=blocker,
                    recommended_action="Fix missing or invalid required artifacts before running another research cycle.",
                    rationale="A blocker means the current run is not inspectable enough to support downstream conclusions.",
                    human_required=True,
                )
            )

        warning_text = "\n".join(readiness.warnings).lower()
        if "few or no repeated seeds" in warning_text:
            actions.append(
                RemediationAction(
                    priority=2,
                    action_type="increase_replication",
                    issue="The result table has weak replication evidence.",
                    recommended_action="Run one more autonomous iteration and keep at least three branches to generate additional seeds or replications.",
                    rationale="Repeated seeds reduce the risk of drawing conclusions from a lucky computational instance.",
                    suggested_config_changes={
                        "max_research_iterations": max(config.max_research_iterations + 1, 3),
                        "max_candidate_branches": max(config.max_candidate_branches, 3),
                    },
                    human_required=False,
                )
            )
        if "multiple methods" in warning_text:
            actions.append(
                RemediationAction(
                    priority=2,
                    action_type="strengthen_baselines",
                    issue="The results do not clearly compare multiple methods.",
                    recommended_action="Force the next experiment to include a named baseline and a named proposed method.",
                    rationale="OR research claims need fair baselines before method performance can be interpreted.",
                    suggested_config_changes={"max_candidate_branches": max(config.max_candidate_branches, 3)},
                    human_required=False,
                )
            )
        if "diagnostic" in warning_text:
            actions.append(
                RemediationAction(
                    priority=3,
                    action_type="add_diagnostics",
                    issue="The result table lacks feasibility, gap, violation, or robustness diagnostics.",
                    recommended_action="Add at least one diagnostic column such as gap, feasible, constraint_violation, or robustness_metric.",
                    rationale="Diagnostics make failures and infeasible model behavior visible instead of hiding them in aggregate objective values.",
                    human_required=False,
                )
            )
        if "novelty evidence is weak" in warning_text:
            actions.append(
                RemediationAction(
                    priority=1,
                    action_type="strengthen_novelty_evidence",
                    issue="Novelty evidence is mock-generated or too weak.",
                    recommended_action="Provide a literature_dir or enable a real literature search before treating novelty claims as research evidence.",
                    rationale="The system must not fabricate citations or infer novelty without inspectable sources.",
                    suggested_config_changes={"enable_novelty_check": True},
                    human_required=True,
                )
            )
        if "claim checker flagged" in warning_text:
            actions.append(
                RemediationAction(
                    priority=1,
                    action_type="verify_claims",
                    issue="High-risk report claims were flagged.",
                    recommended_action="Open claim_check.md and either support, weaken, or remove each flagged claim.",
                    rationale="Automated reports should stay cautious until evidence is verified.",
                    human_required=True,
                )
            )
        if "human review gates" in warning_text:
            actions.append(
                RemediationAction(
                    priority=1,
                    action_type="resolve_human_gate",
                    issue="LLM agents requested explicit human review.",
                    recommended_action="Review human_gate_requests.md before authorizing another fully automatic cycle.",
                    rationale="Human gates preserve the human-in-the-loop boundary when assumptions or claims are uncertain.",
                    human_required=True,
                )
            )

        if readiness.score >= 80 and not any(action.action_type == "extend_sensitivity" for action in actions):
            actions.append(
                RemediationAction(
                    priority=4,
                    action_type="extend_sensitivity",
                    issue="The run is review-ready but can still be strengthened.",
                    recommended_action="Add a larger sensitivity sweep or harder stress-test branch in the next cycle.",
                    rationale="Review-ready exploratory evidence is improved by showing robustness across domain-specific parameters.",
                    suggested_config_changes={"max_research_iterations": max(config.max_research_iterations + 1, 3)},
                    human_required=False,
                )
            )

        config_patch: dict[str, int | float | str | bool] = {}
        for action in actions:
            if not action.human_required:
                config_patch.update(action.suggested_config_changes)

        can_autorun = not readiness.blockers and not any(action.human_required for action in actions if action.priority <= 1)
        return ReadinessRemediationPlan(
            readiness_score=readiness.score,
            readiness_level=readiness.level,
            can_autorun_without_human=can_autorun,
            next_run_config_patch=config_patch,
            actions=actions,
        )
