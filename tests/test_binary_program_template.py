"""End-to-end checks that the solver-backed template really solves and verifies."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.core.experiment_contract import validate_experiment_contract
from src.execution.result_parser import parse_summary_json
from src.execution.sandbox_runner import SandboxRunner


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = PROJECT_ROOT / "templates" / "binary_program_template.py"

# Keep the test fast without weakening what it checks.
SHRINK = {
    "PROBLEM_SIZES = [14, 18, 22]": "PROBLEM_SIZES = [8, 10]",
    "SEED_LIST = [0, 1, 2, 3, 4]": "SEED_LIST = [0, 1]",
    "GROUND_TRUTH_TIME_LIMIT_SECONDS = 20.0": "GROUND_TRUTH_TIME_LIMIT_SECONDS = 10.0",
    "SOLVER_TIME_LIMIT_SECONDS = 5.0": "SOLVER_TIME_LIMIT_SECONDS = 2.0",
}


def _prepare(tmp_path: Path, extra: dict[str, str] | None = None) -> Path:
    code = TEMPLATE.read_text(encoding="utf-8")
    for source, target in {**SHRINK, **(extra or {})}.items():
        assert source in code, f"template no longer contains the knob `{source}`"
        code = code.replace(source, target)
    script = tmp_path / "experiment.py"
    script.write_text(code, encoding="utf-8")
    return script


@pytest.fixture(scope="module")
def solved(tmp_path_factory) -> tuple[Path, dict]:
    pytest.importorskip("ortools")
    work_dir = tmp_path_factory.mktemp("binary_program_run")
    script = _prepare(work_dir)
    execution = SandboxRunner(timeout_seconds=600).run_python(script, work_dir)
    assert execution.status == "success", execution.stderr[-2000:]
    return work_dir, parse_summary_json(execution.stdout)


def test_template_produces_the_promised_artifacts(solved) -> None:
    work_dir, _ = solved
    assert (work_dir / "results.csv").exists()
    assert (work_dir / "verification_report.md").exists()
    assert (work_dir / "solver_backend_report.json").exists()
    assert list((work_dir / "figures").glob("*.png"))
    assert list((work_dir / "instances").glob("*.json"))


def test_every_solution_was_independently_verified(solved) -> None:
    work_dir, summary = solved
    df = pd.read_csv(work_dir / "results.csv")
    assert summary["verification_failures"] == 0
    assert summary["infeasible_claimed_solutions"] == 0
    assert df["verification_ok"].all()
    assert df["feasible"].all()


def test_objectives_come_from_solving_not_from_a_formula(solved) -> None:
    work_dir, _ = solved
    df = pd.read_csv(work_dir / "results.csv")
    # A proved optimum exists for every instance, and no method beats it.
    assert df["known_optimum"].notna().all()
    assert df["known_optimum_source"].str.endswith("_proved").all()
    assert (df["objective"] <= df["known_optimum"] + 1e-9).all(), "maximization: nothing beats the optimum"
    exact = df[df["method"] == "exact_cp_sat"]
    assert (exact["solver_status"] == "optimal").all()
    assert exact["gap_to_known_optimum"].abs().max() < 1e-9
    # The heuristics are not silently handed the answer.
    heuristics = df[df["method"].isin(["greedy", "local_search"])]
    assert (heuristics["solver_status"] == "feasible").all()
    assert (heuristics["gap_to_known_optimum"] >= -1e-9).all()


def test_the_result_can_go_either_way(solved) -> None:
    """The proposed method is not guaranteed to win by construction."""

    work_dir, summary = solved
    greedy = summary["methods"]["greedy"]["mean_gap_to_known_optimum"]
    local = summary["methods"]["local_search"]["mean_gap_to_known_optimum"]
    assert greedy is not None and local is not None
    assert local <= greedy + 1e-9, "local search starts from greedy, so it cannot be worse"


def test_results_satisfy_the_strict_solver_contract(solved) -> None:
    work_dir, _ = solved
    report = validate_experiment_contract(work_dir)
    assert report.contract_name == "solver"
    assert report.passed is True, report.failures


def test_unavailable_methods_are_reported_rather_than_hidden(tmp_path: Path) -> None:
    script = _prepare(
        tmp_path,
        extra={
            '"exact_cp_sat", "greedy", "local_search"': '"exact_cp_sat", "greedy", "local_search", "exact_pyomo"'
        },
    )
    execution = SandboxRunner(timeout_seconds=600).run_python(script, tmp_path)
    assert execution.status == "success", execution.stderr[-2000:]
    summary = parse_summary_json(execution.stdout)
    backends = json.loads((tmp_path / "solver_backend_report.json").read_text(encoding="utf-8"))
    pyomo_row = next(row for row in backends if row["method"] == "exact_pyomo")
    if pyomo_row["available"]:
        assert "exact_pyomo" in summary["methods"]
    else:
        assert "exact_pyomo" in summary["unavailable_requested_methods"]
        assert pyomo_row["unavailable_reason"]
        assert "exact_pyomo" in (tmp_path / "verification_report.md").read_text(encoding="utf-8")


def test_a_broken_experiment_reports_a_real_traceback(tmp_path: Path) -> None:
    """The debug loop needs the traceback, not a silent success."""

    script = _prepare(tmp_path, extra={'INSTANCE_FAMILY = "multi_knapsack"': 'INSTANCE_FAMILY = "not_a_family"'})
    execution = SandboxRunner(timeout_seconds=120).run_python(script, tmp_path)
    assert execution.status == "failed"
    assert "KeyError" in execution.stderr
    assert "not_a_family" in execution.stderr
