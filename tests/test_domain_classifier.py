from src.agents.domain_classifier_agent import DomainClassifierAgent
from src.llm_client import LLMClient


def test_domain_classifier_scheduling() -> None:
    agent = DomainClassifierAgent(LLMClient(use_mock=True))
    result = agent.run("robust scheduling with due date tightness")
    assert result["profile"] == "scheduling"


def test_domain_classifier_network_reliability() -> None:
    agent = DomainClassifierAgent(LLMClient(use_mock=True))
    result = agent.run("network reliability and supply chain disruption")
    assert result["profile"] == "network_analysis"
