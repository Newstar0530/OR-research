from pathlib import Path

from src.config import AppConfig
from src.orchestrator import ResearchOrchestrator


def test_orchestrator_stage_order_constant() -> None:
    assert ResearchOrchestrator.stage_order[0] == "domain_classification"
    assert "idea_generation" in ResearchOrchestrator.stage_order
    assert ResearchOrchestrator.stage_order[-1] == "automated_review"


def test_config_loading_shape() -> None:
    config = AppConfig(project_name="x", research_goal="goal")
    assert config.use_mock_llm is True
