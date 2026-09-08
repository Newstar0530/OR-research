from __future__ import annotations

from pathlib import Path

from src.agent_system.action import AgentAction, PolicyDecision


class AgentPolicy:
    """Safety policy for LLM-proposed actions.

    The runtime accepts structured suggestions and state updates, but does not
    allow arbitrary shell execution or broad filesystem writes.
    """

    allowed_action_types = {
        "propose_hypothesis",
        "revise_model",
        "design_experiment",
        "debug_error",
        "propose_ablation",
        "summarize_evidence",
        "request_human_check",
        "write_note",
    }
    allowed_write_artifacts = {"agent_action_plan.md", "human_gate_requests.md", "llm_agent_notes.md"}
    state_only_action_types = {"propose_hypothesis", "revise_model", "debug_error", "summarize_evidence", "request_human_check", "write_note"}

    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir).resolve()

    def evaluate(self, action: AgentAction) -> PolicyDecision:
        if action.action_type not in self.allowed_action_types:
            return PolicyDecision(allowed=False, reason=f"Action type `{action.action_type}` is not allowed in this runtime.", action=action)
        if action.target_artifact:
            target = Path(action.target_artifact)
            if target.is_absolute():
                try:
                    target.resolve().relative_to(self.run_dir)
                except ValueError:
                    return PolicyDecision(allowed=False, reason="Target artifact is outside the run directory.", action=action)
                name = target.name
            else:
                name = target.name
            if name not in self.allowed_write_artifacts:
                if action.action_type in self.state_only_action_types:
                    action.target_artifact = None
                    return PolicyDecision(
                        allowed=True,
                        reason=f"Ignored non-artifact target `{name}` for state-only action.",
                        action=action,
                    )
                return PolicyDecision(allowed=False, reason=f"Writing `{name}` is not allowed by policy.", action=action)
        return PolicyDecision(allowed=True, reason="Action allowed by policy.", action=action)
