from __future__ import annotations

import json
from pathlib import Path

from src.agent_system.action import AgentAction
from src.agent_system.base_agent import BaseResearchAgent
from src.agent_system.message import AgentMessage
from src.agent_system.policy import AgentPolicy
from src.agent_system.state import ResearchState


class AgentRuntime:
    def __init__(self, agents: list[BaseResearchAgent], policy: AgentPolicy) -> None:
        self.agents = agents
        self.policy = policy
        self.messages: list[AgentMessage] = []

    def run_round(self, state: ResearchState) -> ResearchState:
        for agent in self.agents:
            action = agent.decide(state)
            decision = self.policy.evaluate(action)
            message = AgentMessage(agent_name=agent.name, role=agent.role, action=action, policy_decision=decision)
            self.messages.append(message)
            if decision.allowed:
                self._apply_action(state, action)
        return state

    def message_count(self) -> int:
        return len(self.messages)

    def save(self, run_dir: str | Path, state: ResearchState) -> None:
        root = Path(run_dir)
        (root / "llm_agent_trace.json").write_text(
            json.dumps([message.model_dump() for message in self.messages], indent=2, default=str),
            encoding="utf-8",
        )
        (root / "llm_agent_trace.md").write_text(self._trace_markdown(), encoding="utf-8")
        (root / "research_state.json").write_text(json.dumps(state.model_dump(), indent=2, default=str), encoding="utf-8")

    def _apply_action(self, state: ResearchState, action: AgentAction) -> None:
        if action.action_type == "propose_hypothesis":
            state.hypotheses.append(action.content)
        elif action.action_type in {"design_experiment", "propose_ablation"}:
            state.proposed_experiments.append(action.content)
            state.search_directives.append(action.content)
            if action.action_type == "propose_ablation":
                state.mutation_directives.append(action.content)
        elif action.action_type == "summarize_evidence":
            state.evidence.append(action.content)
        elif action.action_type == "request_human_check":
            state.human_checks.append(action.content)
        elif action.action_type in {"revise_model", "debug_error"}:
            state.open_questions.append(action.content)
        elif action.action_type == "write_note":
            state.metadata.setdefault("notes", []).append(action.content)
        if action.target_artifact:
            self._append_artifact_note(state.run_path / Path(action.target_artifact).name, action)

    @staticmethod
    def _append_artifact_note(path: Path, action: AgentAction) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(f"## {action.action_type}\n\n")
            f.write(f"Rationale: {action.rationale}\n\n")
            f.write(action.content + "\n\n")
            if action.safety_checks:
                f.write("Safety checks:\n")
                for check in action.safety_checks:
                    f.write(f"- {check}\n")
                f.write("\n")

    def _trace_markdown(self) -> str:
        lines = ["# LLM Agent Trace", ""]
        for message in self.messages:
            decision = message.policy_decision
            lines.append(f"## {message.agent_name}")
            lines.append(f"- role: {message.role}")
            lines.append(f"- action: {message.action.action_type}")
            lines.append(f"- allowed: {decision.allowed if decision else False}")
            lines.append(f"- policy_reason: {decision.reason if decision else 'missing'}")
            lines.append(f"- rationale: {message.action.rationale}")
            lines.append(f"- content: {message.action.content}")
            lines.append("")
        return "\n".join(lines)
