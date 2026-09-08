from pathlib import Path

import pandas as pd

from src.core.budget import ResearchBudget
from src.core.decision_engine import DecisionEngine
from src.core.experiment_contract import validate_experiment_contract
from src.core.experiment_search import ExperimentSearchController
from src.core.research_journal import ResearchJournal
from src.core.research_node import ResearchNode
from src.agents.experiment_refiner_agent import ExperimentRefinerAgent
from src.agents.mutation_agent import MutationAgent
from src.agents.code_patch_agent import CodePatchAgent
from src.agents.planner_agent import PlannerAgent
from src.agents.strategy_agent import StrategyAgent
from src.agent_system.state import ResearchState
from src.agent_system.agents import LLMExperimentDesignerAgent, LLMEvidenceAgent
from src.agent_system.policy import AgentPolicy
from src.agent_system.runtime import AgentRuntime
from src.agent_system.patch import PatchOperation, PatchProposal, SafePatchApplier
from src.execution.sandbox_runner import SandboxRunner
from src.llm_client import LLMClient
from src.schemas import AlgorithmPlan, ResearchIdea


def _idea() -> ResearchIdea:
    return ResearchIdea(
        title="Generic OR autonomous test",
        problem_context="test",
        core_hypothesis="proposed method improves the metric",
        expected_contribution="test",
        proposed_model_type="generic",
        proposed_algorithm_type="generic",
        experimental_plan="run variants",
        interestingness_score=5,
        novelty_score=5,
        feasibility_score=8,
        risk_score=3,
        assumptions=[],
        required_data=[],
        expected_outputs=[],
    )


def _algorithm() -> AlgorithmPlan:
    return AlgorithmPlan(
        exact_solver_plan="baseline",
        heuristic_plan="proposed",
        baseline_methods=["baseline"],
        pseudocode="run",
        complexity_discussion="small",
        stopping_criteria="budget",
        required_packages=["pandas"],
        evaluation_metrics=["objective"],
    )


def test_decision_engine_scores_contract_results(tmp_path: Path) -> None:
    pd.DataFrame(
        [
            {"instance_id": "i1", "method": "baseline", "objective": 10.0, "runtime_seconds": 0.1, "seed": 1},
            {"instance_id": "i1", "method": "proposed", "objective": 8.0, "runtime_seconds": 0.2, "seed": 1},
        ]
    ).to_csv(tmp_path / "results.csv", index=False)
    status, metric, analysis = DecisionEngine("objective", "minimize").evaluate(
        tmp_path / "results.csv",
        execution=type("Execution", (), {"status": "success"})(),
        contract=validate_experiment_contract(tmp_path),
    )
    assert status == "success"
    assert metric == 8.0
    assert "proposed" in analysis


def test_research_journal_best_node() -> None:
    journal = ResearchJournal()
    journal.append(ResearchNode(iteration=0, branch_index=0, plan="a", status="success", metric_name="objective", metric_value=10.0))
    journal.append(ResearchNode(iteration=0, branch_index=1, plan="b", status="success", metric_name="objective", metric_value=8.0))
    assert journal.best_node().metric_value == 8.0


def test_experiment_search_controller_runs_variants(tmp_path: Path) -> None:
    code = """
from pathlib import Path
import json
import pandas as pd
out = Path(__file__).resolve().parent
pd.DataFrame([
    {"instance_id": "i1", "method": "baseline", "objective": 10.0, "runtime_seconds": 0.1, "seed": 42},
    {"instance_id": "i1", "method": "proposed", "objective": 9.0, "runtime_seconds": 0.2, "seed": 42},
]).to_csv(out / "results.csv", index=False)
print("SUMMARY_JSON:" + json.dumps({"status": "success"}))
"""
    controller = ExperimentSearchController(
        LLMClient(use_mock=True),
        SandboxRunner(timeout_seconds=10),
        ResearchBudget(max_iterations=1, max_branches=2, timeout_seconds=10),
        DecisionEngine("objective", "minimize"),
    )
    state = ResearchState(
        project_name="x",
        research_goal="goal",
        domain_profile="optimization",
        run_dir=str(tmp_path),
        search_directives=["Baseline strengthening branch; Ablation branch; Stress test branch"],
    )
    runtime = AgentRuntime(
        [LLMExperimentDesignerAgent(LLMClient(use_mock=True)), LLMEvidenceAgent(LLMClient(use_mock=True))],
        AgentPolicy(tmp_path),
    )
    journal = controller.run(tmp_path, _idea(), _algorithm(), code, research_state=state, agent_runtime=runtime)
    assert len(journal.nodes) == 2
    assert (tmp_path / "autonomous_journal.json").exists()
    assert (tmp_path / "strategy_trace.json").exists()
    assert (tmp_path / "mutation_trace.json").exists()
    assert (tmp_path / "patch_trace.json").exists()
    assert (tmp_path / "experiment_workspace" / "prompt.json").exists()
    assert (tmp_path / "workspace_lineage.json").exists()
    assert (tmp_path / "bfts_frontier.json").exists()
    assert (tmp_path / "best_experiment.py").exists()
    assert (tmp_path / "best_prompt.json").exists()
    assert (tmp_path / "results.csv").exists()
    assert journal.best_node() is not None
    assert state.metadata["autonomous_nodes"] == 2
    assert any("Best autonomous node" in item for item in state.evidence)
    assert (tmp_path / "agent_feedback_rounds.json").exists()
    assert (tmp_path / "state_snapshots" / "after_iteration_00.json").exists()
    assert runtime.message_count() == 0


def test_experiment_search_runs_in_loop_agent_feedback(tmp_path: Path) -> None:
    code = """
from pathlib import Path
import json
import pandas as pd
out = Path(__file__).resolve().parent
pd.DataFrame([
    {"instance_id": "i1", "method": "baseline", "objective": 10.0, "runtime_seconds": 0.1, "seed": 42},
    {"instance_id": "i1", "method": "proposed", "objective": 9.0, "runtime_seconds": 0.2, "seed": 42},
]).to_csv(out / "results.csv", index=False)
print("SUMMARY_JSON:" + json.dumps({"status": "success"}))
"""
    state = ResearchState(project_name="x", research_goal="goal", domain_profile="optimization", run_dir=str(tmp_path))
    runtime = AgentRuntime(
        [LLMExperimentDesignerAgent(LLMClient(use_mock=True)), LLMEvidenceAgent(LLMClient(use_mock=True))],
        AgentPolicy(tmp_path),
    )
    controller = ExperimentSearchController(
        LLMClient(use_mock=True),
        SandboxRunner(timeout_seconds=10),
        ResearchBudget(max_iterations=2, max_branches=1, timeout_seconds=10),
        DecisionEngine("objective", "minimize"),
    )
    journal = controller.run(tmp_path, _idea(), _algorithm(), code, research_state=state, agent_runtime=runtime)
    assert len(journal.nodes) == 2
    assert runtime.message_count() == 2
    feedback = (tmp_path / "agent_feedback_rounds.md").read_text(encoding="utf-8")
    assert "After Iteration 0" in feedback
    lineage = (tmp_path / "workspace_lineage.md").read_text(encoding="utf-8")
    assert "source_workspace" in lineage


def test_planner_uses_llm_agent_directives() -> None:
    branches = PlannerAgent(LLMClient(use_mock=True)).run(
        _idea(),
        max_branches=3,
        directives=["Baseline strengthening branch; Ablation branch; Adversarial stress test branch"],
    )
    assert "Baseline strengthening" in branches[0]
    assert any("Ablation" in branch for branch in branches)


def test_strategy_agent_adapts_after_initial_iteration() -> None:
    journal = ResearchJournal()
    journal.append(ResearchNode(iteration=0, branch_index=0, plan="a", status="success", metric_name="objective", metric_value=10.0))
    strategy = StrategyAgent(LLMClient(use_mock=True)).run(
        journal,
        iteration=1,
        initial_branches=["initial"],
        max_branches=3,
        patience=2,
        min_improvement=0.0,
    )
    assert strategy.should_stop is False
    assert any("effect_range" in plan for plan in strategy.branch_plans)


def test_refiner_applies_effect_range() -> None:
    code = "PROPOSED_EFFECT_LOW = 0.5\nPROPOSED_EFFECT_HIGH = 3.0\n"
    refined = ExperimentRefinerAgent(LLMClient(use_mock=True)).run(code, "effect_range=1.25,4.75", 1, 0)
    assert "PROPOSED_EFFECT_LOW = 1.25" in refined
    assert "PROPOSED_EFFECT_HIGH = 4.75" in refined


def test_refiner_preserves_future_import_position() -> None:
    code = '"""old header"""\nfrom __future__ import annotations\nVALUE = 1\n'
    refined = ExperimentRefinerAgent(LLMClient(use_mock=True)).run(code, "plan", 1, 0)
    lines = refined.splitlines()
    future_index = lines.index("from __future__ import annotations")
    header_index = next(index for index, line in enumerate(lines) if line.startswith('"""Autonomous variant:'))
    assert future_index < header_index


def test_mutation_agent_selects_stress_operator() -> None:
    mutation = MutationAgent(LLMClient(use_mock=True)).run(ResearchJournal(), 1, 2, "Adversarial stress test")
    assert mutation.kind == "stress_test"
    assert "NOISE_SCALE = 1.0" in mutation.replacements


def test_refiner_applies_mutation_replacements() -> None:
    code = "PROBLEM_SIZES = [10, 25, 50]\nNOISE_SCALE = 1.0\n"
    mutation = MutationAgent(LLMClient(use_mock=True)).run(ResearchJournal(), 1, 2, "Adversarial stress test")
    refined = ExperimentRefinerAgent(LLMClient(use_mock=True)).run(code, "Adversarial stress test", 1, 2, mutation)
    assert "PROBLEM_SIZES = [25, 50, 100]" in refined
    assert "NOISE_SCALE = 2.0" in refined


def test_code_patch_agent_marks_stress_rows() -> None:
    code = '"feasible": True,\nprint("SUMMARY_JSON:{}")\nopen("results.csv", "w")\n'
    mutation = MutationAgent(LLMClient(use_mock=True)).run(ResearchJournal(), 1, 2, "Adversarial stress test")
    proposal = CodePatchAgent(LLMClient(use_mock=True)).run(code, "Adversarial stress test", mutation)
    assert proposal is not None
    assert "stress_tested" in proposal.operations[0].replace


def test_safe_patch_applier_allows_contract_preserving_patch(tmp_path: Path) -> None:
    script = tmp_path / "experiment.py"
    script.write_text(
        'BASELINE_EFFECT = 0.0\nopen("results.csv", "w").write("x")\nprint("SUMMARY_JSON:{}")\n',
        encoding="utf-8",
    )
    proposal = PatchProposal(
        rationale="strengthen baseline",
        operations=[PatchOperation(find="BASELINE_EFFECT = 0.0", replace="BASELINE_EFFECT = 0.3")],
        expected_effect="stronger baseline",
    )
    result = SafePatchApplier(tmp_path).apply(proposal)
    assert result.applied is True
    assert "BASELINE_EFFECT = 0.3" in script.read_text(encoding="utf-8")


def test_safe_patch_applier_rejects_contract_breaking_patch(tmp_path: Path) -> None:
    script = tmp_path / "experiment.py"
    script.write_text(
        'open("results.csv", "w").write("x")\nprint("SUMMARY_JSON:{}")\n',
        encoding="utf-8",
    )
    proposal = PatchProposal(
        rationale="break contract",
        operations=[PatchOperation(find="SUMMARY_JSON", replace="NO_SUMMARY")],
        expected_effect="bad",
    )
    result = SafePatchApplier(tmp_path).apply(proposal)
    assert result.applied is False
