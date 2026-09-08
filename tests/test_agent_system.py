from pathlib import Path

from src.agent_system.action import AgentAction
from src.agent_system.agents import LLMExperimentDesignerAgent, LLMEvidenceAgent, LLMModelCriticAgent, LLMPlannerAgent
from src.agent_system.policy import AgentPolicy
from src.agent_system.runtime import AgentRuntime
from src.agent_system.state import ResearchState
from src.llm_client import LLMClient


def _state(tmp_path: Path) -> ResearchState:
    return ResearchState(
        project_name="agent_test",
        research_goal="test OR research topic",
        domain_profile="optimization",
        run_dir=str(tmp_path),
    )


def test_agent_policy_blocks_code_modification(tmp_path: Path) -> None:
    action = AgentAction(
        action_type="modify_experiment_code",
        rationale="try direct edit",
        content="change code",
        target_artifact="generated_experiment.py",
    )
    decision = AgentPolicy(tmp_path).evaluate(action)
    assert decision.allowed is False


def test_agent_policy_ignores_bad_target_for_state_action(tmp_path: Path) -> None:
    action = AgentAction(
        action_type="propose_hypothesis",
        rationale="local model used a logical target",
        content="hypothesis",
        target_artifact="hypotheses",
    )
    decision = AgentPolicy(tmp_path).evaluate(action)
    assert decision.allowed is True
    assert decision.action.target_artifact is None


def test_agent_runtime_updates_state_and_saves_trace(tmp_path: Path) -> None:
    runtime = AgentRuntime(
        [
            LLMPlannerAgent(LLMClient(use_mock=True)),
            LLMModelCriticAgent(LLMClient(use_mock=True)),
            LLMExperimentDesignerAgent(LLMClient(use_mock=True)),
            LLMEvidenceAgent(LLMClient(use_mock=True)),
        ],
        AgentPolicy(tmp_path),
    )
    state = runtime.run_round(_state(tmp_path))
    runtime.save(tmp_path, state)
    assert state.hypotheses
    assert state.human_checks
    assert state.proposed_experiments
    assert state.search_directives
    assert state.evidence
    assert (tmp_path / "llm_agent_trace.json").exists()
    assert (tmp_path / "research_state.json").exists()
