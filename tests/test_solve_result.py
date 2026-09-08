from src.core.solve_result import (
    SOLVE_RESULT_COLUMNS,
    SolveResult,
    compute_mip_gap,
    relative_shortfall,
    summarize_results,
    write_results_csv,
)


def test_csv_columns_start_with_the_legacy_contract_columns() -> None:
    assert SOLVE_RESULT_COLUMNS[:5] == [
        "instance_id",
        "method",
        "objective",
        "runtime_seconds",
        "seed",
    ]
    for column in ("solver_status", "feasible", "verification_ok", "gap_to_known_optimum"):
        assert column in SOLVE_RESULT_COLUMNS


def test_mip_gap_is_direction_aware_and_never_negative() -> None:
    assert compute_mip_gap(100.0, 90.0, "minimize") == 0.1
    assert compute_mip_gap(100.0, 110.0, "maximize") == 0.1
    # A bound on the wrong side of the incumbent is clipped, not reported negative.
    assert compute_mip_gap(100.0, 110.0, "minimize") == 0.0
    assert compute_mip_gap(None, 10.0, "minimize") is None
    assert compute_mip_gap(10.0, None, "minimize") is None


def test_relative_shortfall_is_positive_when_worse_than_reference() -> None:
    assert relative_shortfall(90.0, 100.0, "maximize") == 0.1
    assert relative_shortfall(110.0, 100.0, "minimize") == 0.1
    # Keeps the sign: beating a "known optimum" is a finding, not a rounding artifact.
    assert relative_shortfall(110.0, 100.0, "maximize") < 0


def test_trust_flags_require_feasibility_and_verification() -> None:
    good = SolveResult(
        instance_id="i", method="m", solver_status="optimal", feasible=True, verification_ok=True
    )
    assert good.is_trustworthy and good.proved_optimal

    infeasible = SolveResult(
        instance_id="i", method="m", solver_status="optimal", feasible=False, verification_ok=True
    )
    assert not infeasible.is_trustworthy and not infeasible.proved_optimal

    unverified = SolveResult(
        instance_id="i", method="m", solver_status="optimal", feasible=True, verification_ok=False
    )
    assert not unverified.is_trustworthy

    heuristic = SolveResult(
        instance_id="i", method="m", solver_status="feasible", feasible=True, verification_ok=True
    )
    assert heuristic.is_trustworthy and not heuristic.proved_optimal


def test_derived_metrics_and_notes() -> None:
    result = SolveResult(
        instance_id="i",
        method="greedy",
        objective=90.0,
        dual_bound=100.0,
        known_optimum=100.0,
        objective_sense="maximize",
    ).with_derived_metrics()
    assert result.mip_gap is not None and round(result.mip_gap, 6) == 0.111111
    assert result.gap_to_known_optimum == 0.1
    result.add_note("first")
    result.add_note("")
    result.add_note("second")
    assert result.notes == "first | second"


def test_write_results_csv_and_summary(tmp_path) -> None:
    results = [
        SolveResult(
            instance_id="i1",
            method="exact",
            solver_status="optimal",
            objective=10.0,
            feasible=True,
            known_optimum=10.0,
            gap_to_known_optimum=0.0,
        ),
        SolveResult(
            instance_id="i1",
            method="broken",
            solver_status="optimal",
            objective=99.0,
            feasible=False,
            verification_ok=False,
        ),
    ]
    path = write_results_csv(results, tmp_path / "results.csv")
    header = path.read_text(encoding="utf-8").splitlines()[0].split(",")
    assert header == SOLVE_RESULT_COLUMNS

    summary = summarize_results(results)
    assert summary["rows"] == 2
    assert summary["verification_failures"] == 1
    assert summary["status"] == "attention_required"
    assert summary["methods"]["exact"]["proved_optimal_rows"] == 1
    assert summary["methods"]["broken"]["infeasible_rows"] == 1
