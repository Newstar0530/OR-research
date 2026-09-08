from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


ActionType = Literal[
    "propose_hypothesis",
    "revise_model",
    "design_experiment",
    "modify_experiment_code",
    "debug_error",
    "propose_ablation",
    "summarize_evidence",
    "request_human_check",
    "write_note",
]


class AgentAction(BaseModel):
    action_type: ActionType
    rationale: str
    content: str
    target_artifact: str | None = None
    expected_effect: str | None = None
    safety_checks: list[str] = Field(default_factory=list)
    requires_human_review: bool = True


class PolicyDecision(BaseModel):
    allowed: bool
    reason: str
    action: AgentAction

