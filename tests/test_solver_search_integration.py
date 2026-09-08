"""The autonomous search loop, driven by real solving instead of a synthetic formula."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.core.budget import ResearchBudget
from src.core.decision_engine import DecisionEngine
from src.core.experiment_contract import validate_experiment_contract
from src.core.experiment_search import ExperimentSearchController
from src.execution.sandbox_runner import SandboxRunner
from src.llm_client import LLMClient
from src.schemas import AlgorithmPlan, ResearchIdea


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = PROJECT_ROOT / "templates" / "binary_program_template.py"

SHRINK = {
    "PROBLEM_SIZES = [14, 18, 22]": "PROBLEM_SIZES = [8, 10]",
    "SEED_LIST = [0, 1, 2, 3, 4]": "SEED_LIST = [0, 1, 2]",
    "GROUND_TRUTH_TIME_LIMIT_SECONDS = 20.0": "GROUND_TRUTH_TIME_LIMIT_SECONDS = 10.0",
}


def _idea() -> ResearchIdea:
    return ResearchIdea(
        title="MI 0-1 local search quality",
        problem_context="multidimensional knapsack",
        core_hypothesis="restart-augmented local search stays closer to the proved optimum than greedy",
        expected_contribution="verified gap measurements",
        proposed_model_type="0-1 integer program",
        proposed_algorithm_type="local search",
        experimental_plan="compare against CP-SAT reference",
        interestingness_score=6,
        novelty_score=4,
        feasibility_score=9,
        risk_score=3,
        assumptions=[],
        required_data=[],
        expected_outputs=["results.csv"],
    )


def _algorithm() -> AlgorithmPlan:
    return AlgorithmPlan(
        exact_solver_plan="CP-SAT",
        heuristic_plan="greedy + local search",
        baseline_methods=["greedy"],
        pseudocode="solve",
        complexity_discussion="small instances",
        stopping_criteria="budget",
        required_packages=["ortools"],
        evaluation_metrics=["gap_to_known_optimum"],
    )


@pytest.fixture(scope="module")
def searched(tmp_path_factory) -> Path:
    pytest.importorskip("ortools")
    run_dir = tmp_path_factory.mktemp("solver_search")
    code = TEMPLATE.read_text(encoding="utf-8")
    for source, target in SHRINK.items():
        assert source in code
        code = code.replace(source, target)

    controller = ExperimentSearchController(
        LLMClient(use_mock=True),
        SandboxRunner(timeout_seconds=600),
        ResearchBudget(max_iterations=2, max_branches=2, timeout_seconds=600),
        DecisionEngine(
            "gap_to_known_optimum",
            "minimize",
            metric_method="local_search",
            baseline_method="greedy",
            proposed_method="local_search",
        ),
        domain_profile="binary_program",
    )
    controller.run(run_dir, _idea(), _algorithm(), code)
    return run_dir


def test_every_node_solved_and_was_scored(searched: Path) -> None:
    journal = json.loads((searched / "autonomous_journal.json").read_text(encoding="utf-8"))
    nodes = journal["nodes"]
    assert len(nodes) >= 2
    for node in nodes:
        assert node["status"] == "success", node.get("analysis")
        assert node["metric_name"] == "gap_to_known_optimum"
        assert node["metric_value"] is not None
        assert node["metric_value"] >= 0.0


def test_nodes_are_genuinely_different_experiments(searched: Path) -> None:
    """A mutation that matched nothing would produce identical sibling nodes."""

    mutations = json.loads((searched / "mutation_trace.json").read_text(encoding="utf-8"))
    assert mutations
    assert all(item["replacements"] for item in mutations)

    knobs = set()
    for node_dir in (searched / "autonomous_search").iterdir():
        code = (node_dir / "experiment.py").read_text(encoding="utf-8")
        knobs.add(
            tuple(
                line.strip()
                for line in code.splitlines()
                if line.startswith(("DIFFICULTY", "LOCAL_SEARCH_RESTARTS", "SEED_LIST", "SOLVER_TIME_LIMIT"))
            )
        )
    assert len(knobs) > 1, "the search explored only one configuration"


def test_promoted_results_pass_the_solver_contract(searched: Path) -> None:
    assert (searched / "results.csv").exists()
    assert (searched / "verification_report.md").exists()
    report = validate_experiment_contract(searched)
    assert report.contract_name == "solver"
    assert report.passed is True, report.failures


def test_promoted_results_are_verified_solver_output(searched: Path) -> None:
    df = pd.read_csv(searched / "results.csv")
    assert df["verification_ok"].all()
    assert df["feasible"].all()
    assert set(df["method"]) == {"exact_cp_sat", "greedy", "local_search"}
    assert (df.loc[df["method"] == "exact_cp_sat", "solver_status"] == "optimal").all()


def test_the_experiment_log_survives_the_plot_run(searched: Path) -> None:
    """Regression: the plot run used to overwrite the experiment's stdout log."""

    for node_dir in (searched / "autonomous_search").iterdir():
        log = node_dir / "experiment_last.log"
        assert log.exists()
        assert "SUMMARY_JSON:" in log.read_text(encoding="utf-8")
