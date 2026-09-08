from pathlib import Path

import pandas as pd

from src.core.experiment_contract import (
    LEGACY_CONTRACT,
    SOLVER_CONTRACT,
    detect_contract,
    validate_experiment_contract,
)


def _solver_rows(**overrides) -> list[dict]:
    base = {
        "instance_id": "i1",
        "method": "exact_cp_sat",
        "objective": 10.0,
        "runtime_seconds": 0.1,
        "seed": 0,
        "objective_sense": "maximize",
        "solver_status": "optimal",
        "feasible": True,
        "verification_ok": True,
        "n_vars": 5,
        "n_constraints": 1,
        "known_optimum": 10.0,
    }
    heuristic = {**base, "method": "greedy", "solver_status": "feasible", "objective": 9.0}
    rows = [base, heuristic]
    return [{**row, **overrides} for row in rows] if overrides else rows


def _write(tmp_path: Path, rows: list[dict], stdout: str | None = "SUMMARY_JSON:{}") -> Path:
    pd.DataFrame(rows).to_csv(tmp_path / "results.csv", index=False)
    if stdout is not None:
        (tmp_path / "experiment_last.log").write_text(f"STDOUT\n{stdout}\n", encoding="utf-8")
    return tmp_path


def test_contract_tier_is_detected_from_the_columns() -> None:
    assert detect_contract(["instance_id", "method"]).name == LEGACY_CONTRACT.name
    assert detect_contract(["instance_id", "solver_status"]).name == SOLVER_CONTRACT.name


def test_legacy_results_still_pass_unchanged(tmp_path: Path) -> None:
    pd.DataFrame(
        [
            {"instance_id": "i1", "method": "baseline", "objective": 1.0, "runtime_seconds": 0.1, "seed": 42},
            {"instance_id": "i1", "method": "proposed", "objective": 0.8, "runtime_seconds": 0.2, "seed": 42},
        ]
    ).to_csv(tmp_path / "results.csv", index=False)
    report = validate_experiment_contract(tmp_path)
    assert report.passed is True
    assert report.contract_name == "legacy"
    assert "Status: passed" in report.to_markdown()


def test_solver_results_pass_the_stricter_contract(tmp_path: Path) -> None:
    report = validate_experiment_contract(_write(tmp_path, _solver_rows()))
    assert report.contract_name == "solver"
    assert report.passed is True
    assert report.failures == []


def test_verification_failure_fails_the_contract(tmp_path: Path) -> None:
    rows = _solver_rows()
    rows[1]["verification_ok"] = False
    report = validate_experiment_contract(_write(tmp_path, rows))
    assert report.passed is False
    assert any("independent solution verification" in item for item in report.failures)


def test_claimed_solution_that_is_infeasible_fails_the_contract(tmp_path: Path) -> None:
    rows = _solver_rows()
    rows[1]["feasible"] = False
    report = validate_experiment_contract(_write(tmp_path, rows))
    assert report.passed is False
    assert any("infeasible" in item for item in report.failures)


def test_no_usable_solution_fails_the_contract(tmp_path: Path) -> None:
    rows = _solver_rows()
    for row in rows:
        row["solver_status"] = "no_solution"
        row["feasible"] = False
    report = validate_experiment_contract(_write(tmp_path, rows))
    assert report.passed is False
    assert any("nothing to analyze" in item for item in report.failures)


def test_missing_summary_line_fails_the_solver_contract(tmp_path: Path) -> None:
    report = validate_experiment_contract(_write(tmp_path, _solver_rows(), stdout="nothing here"))
    assert report.passed is False
    assert any("SUMMARY_JSON" in item for item in report.failures)


def test_missing_summary_line_only_warns_under_the_legacy_contract(tmp_path: Path) -> None:
    pd.DataFrame(
        [
            {"instance_id": "i1", "method": "a", "objective": 1.0, "runtime_seconds": 0.1, "seed": 1},
            {"instance_id": "i1", "method": "b", "objective": 2.0, "runtime_seconds": 0.1, "seed": 1},
        ]
    ).to_csv(tmp_path / "results.csv", index=False)
    (tmp_path / "experiment_last.log").write_text("STDOUT\nno summary\n", encoding="utf-8")
    report = validate_experiment_contract(tmp_path)
    assert report.passed is True
    assert any("SUMMARY_JSON" in item for item in report.warnings)


def test_uncaptured_stdout_is_a_warning_not_a_failure(tmp_path: Path) -> None:
    report = validate_experiment_contract(_write(tmp_path, _solver_rows(), stdout=None))
    assert report.passed is True
    assert any("not captured" in item for item in report.warnings)


def test_summary_line_is_accepted_from_any_captured_log(tmp_path: Path) -> None:
    _write(tmp_path, _solver_rows(), stdout=None)
    (tmp_path / "subprocess_last.log").write_text("STDOUT\nSUMMARY_JSON:{}\n", encoding="utf-8")
    assert validate_experiment_contract(tmp_path).passed is True


def test_unproved_reference_is_only_a_warning(tmp_path: Path) -> None:
    rows = _solver_rows()
    rows[0]["solver_status"] = "time_limit"
    rows[0]["known_optimum"] = None
    rows[1]["known_optimum"] = None
    report = validate_experiment_contract(_write(tmp_path, rows))
    assert report.passed is True
    assert any("proved optimality" in item for item in report.warnings)
    assert any("proved optimum" in item for item in report.warnings)
