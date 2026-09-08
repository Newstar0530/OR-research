from src.agents.readiness_remediation_agent import ReadinessRemediationAgent
from src.config import AppConfig
from src.llm_client import LLMClient
from src.utils.autonomy_readiness import AutonomyReadinessReport


def _config() -> AppConfig:
    return AppConfig(
        project_name="demo",
        research_goal="robust scheduling",
        max_research_iterations=2,
        max_candidate_branches=2,
    )


def test_readiness_remediation_blocks_human_required_evidence() -> None:
    readiness = AutonomyReadinessReport(
        score=80,
        level="review-ready",
        warnings=[
            "Novelty evidence is weak or mock-generated.",
            "Claim checker flagged 1 unsupported high-risk claim snippets.",
            "LLM agents requested explicit human review gates.",
        ],
    )

    plan = ReadinessRemediationAgent(LLMClient(use_mock=True)).run(readiness, _config())

    assert not plan.can_autorun_without_human
    assert any(action.action_type == "strengthen_novelty_evidence" for action in plan.actions)
    assert any(action.action_type == "verify_claims" for action in plan.actions)


def test_readiness_remediation_suggests_replication_patch() -> None:
    readiness = AutonomyReadinessReport(
        score=72,
        level="needs-human-review",
        warnings=["Few or no repeated seeds were detected."],
    )

    plan = ReadinessRemediationAgent(LLMClient(use_mock=True)).run(readiness, _config())

    assert plan.can_autorun_without_human
    assert plan.next_run_config_patch["max_research_iterations"] == 3
    assert "max_research_iterations: 3" in plan.config_patch_yaml()
