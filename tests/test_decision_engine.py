from pathlib import Path

import pandas as pd

from src.core.decision_engine import DecisionEngine
from src.core.experiment_contract import ContractValidationReport
from src.schemas import ExecutionResult


PASSED = ContractValidationReport(passed=True)
OK = ExecutionResult(status="success", returncode=0, stdout="", stderr="", duration_seconds=0.1)


def _write(tmp_path: Path, rows: list[dict]) -> Path:
    path = tmp_path / "results.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _solver_rows() -> list[dict]:
    rows = []
    for seed in range(4):
        rows.append(
            {
                "instance_id": f"i{seed}",
                "method": "exact_cp_sat",
                "gap_to_known_optimum": 0.0,
                "feasible": True,
                "verification_ok": True,
                "solver_status": "optimal",
            }
        )
        rows.append(
            {
                "instance_id": f"i{seed}",
                "method": "greedy",
                "gap_to_known_optimum": 0.10,
                "feasible": True,
                "verification_ok": True,
                "solver_status": "feasible",
            }
        )
        rows.append(
            {
                "instance_id": f"i{seed}",
                "method": "local_search",
                "gap_to_known_optimum": 0.04,
                "feasible": True,
                "verification_ok": True,
                "solver_status": "feasible",
            }
        )
    return rows


def test_metric_scoped_to_one_method_measures_that_method(tmp_path: Path) -> None:
    path = _write(tmp_path, _solver_rows())
    engine = DecisionEngine("gap_to_known_optimum", "minimize", metric_method="local_search")
    status, value, analysis = engine.evaluate(path, OK, PASSED)
    assert status == "success"
    assert value == 0.04, "the node metric must track the method under test"
    assert "local_search" in analysis


def test_unscoped_metric_is_dominated_by_the_exact_solver(tmp_path: Path) -> None:
    """Why scoping matters: without it every node scores the reference solver."""

    path = _write(tmp_path, _solver_rows())
    engine = DecisionEngine("gap_to_known_optimum", "minimize")
    status, value, _ = engine.evaluate(path, OK, PASSED)
    assert status == "success" and value == 0.0


def test_verification_failure_fails_the_node(tmp_path: Path) -> None:
    rows = _solver_rows()
    rows[1]["verification_ok"] = False
    path = _write(tmp_path, rows)
    engine = DecisionEngine("gap_to_known_optimum", "minimize", metric_method="local_search")
    status, value, analysis = engine.evaluate(path, OK, PASSED)
    assert status == "contract_failed"
    assert value is None
    assert "verification" in analysis


def test_infeasible_rows_are_excluded_from_the_metric(tmp_path: Path) -> None:
    rows = _solver_rows()
    rows.append(
        {
            "instance_id": "i9",
            "method": "local_search",
            "gap_to_known_optimum": 99.0,
            "feasible": False,
            "verification_ok": True,
            "solver_status": "no_solution",
        }
    )
    path = _write(tmp_path, rows)
    engine = DecisionEngine("gap_to_known_optimum", "minimize", metric_method="local_search")
    _, value, _ = engine.evaluate(path, OK, PASSED)
    assert value == 0.04


def test_missing_scoped_method_is_a_contract_failure(tmp_path: Path) -> None:
    path = _write(tmp_path, _solver_rows())
    engine = DecisionEngine("gap_to_known_optimum", "minimize", metric_method="tabu_search")
    status, _, analysis = engine.evaluate(path, OK, PASSED)
    assert status == "contract_failed"
    assert "tabu_search" in analysis


def test_explicit_method_pair_drives_the_comparison(tmp_path: Path) -> None:
    path = _write(tmp_path, _solver_rows())
    engine = DecisionEngine(
        "gap_to_known_optimum",
        "minimize",
        baseline_method="greedy",
        proposed_method="local_search",
    )
    comparison = engine.method_comparison(path)
    assert comparison["baseline_method"] == "greedy"
    assert comparison["proposed_method"] == "local_search"
    assert comparison["improvement"] > 0
    assert comparison["supports_hypothesis"] is True


def test_comparison_reports_verification_failure_instead_of_a_number(tmp_path: Path) -> None:
    rows = _solver_rows()
    rows[2]["verification_ok"] = False
    path = _write(tmp_path, rows)
    engine = DecisionEngine("gap_to_known_optimum", "minimize")
    comparison = engine.method_comparison(path)
    assert comparison["verification_failed"] is True
    assert comparison["supports_hypothesis"] is False


def test_legacy_results_keep_working(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        [
            {"instance_id": "i1", "method": "baseline", "objective": 10.0},
            {"instance_id": "i1", "method": "proposed", "objective": 8.0},
        ],
    )
    engine = DecisionEngine("objective", "minimize")
    status, value, _ = engine.evaluate(path, OK, PASSED)
    assert status == "success" and value == 8.0
    comparison = engine.method_comparison(path)
    assert comparison["supports_hypothesis"] is True
