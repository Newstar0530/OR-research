from src.agents.mutation_agent import MutationAgent
from src.core.domain_tools import get_domain_tool_spec
from src.core.research_journal import ResearchJournal
from src.llm_client import LLMClient


def test_domain_tool_spec_for_network() -> None:
    spec = get_domain_tool_spec("network_analysis")
    assert spec.template_key == "network_analysis"
    assert "path_cost" in spec.metrics
    assert any("density" in hint for hint in spec.mutation_hints)


def test_scheduling_stress_mutation_is_domain_specific() -> None:
    mutation = MutationAgent(LLMClient(use_mock=True)).run(
        ResearchJournal(),
        iteration=0,
        branch_index=0,
        plan="Adversarial stress test",
        domain_profile="scheduling",
    )
    assert mutation.metadata["domain_profile"] == "scheduling"
    assert "N_JOBS_LIST = [10, 20, 40]" in mutation.replacements


def test_inventory_replication_mutation_is_domain_specific() -> None:
    mutation = MutationAgent(LLMClient(use_mock=True)).run(
        ResearchJournal(),
        iteration=0,
        branch_index=0,
        plan="Replication branch",
        domain_profile="inventory",
    )
    assert mutation.metadata["domain_profile"] == "inventory"
    assert "DEMAND_SAMPLE_SIZE = 5000" in mutation.replacements
