from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.statistical_evidence import evaluate_statistical_evidence


def _write(tmp_path: Path, rows: list[dict]) -> Path:
    path = tmp_path / "results.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _noisy_pair(baseline_mean: float, proposed_mean: float, sd: float, n: int) -> list[dict]:
    """Two methods with the given exact means, so the effect is deterministic."""

    rng = np.random.default_rng(0)
    rows: list[dict] = []
    for method, mean in (("baseline", baseline_mean), ("proposed", proposed_mean)):
        values = rng.normal(mean, sd, n)
        values = values - values.mean() + mean
        rows.extend({"method": method, "objective": float(v), "seed": i} for i, v in enumerate(values))
    return rows


def test_large_but_noisy_effect_is_not_called_strong_evidence(tmp_path: Path) -> None:
    """Regression test for the false-confidence bug.

    A 1.16 improvement on a standard deviation of 13.5 with n=24 used to be
    labelled `strong_preliminary` because the sample was big and the point
    estimate had the right sign. Its p-value is ~0.74.
    """

    path = _write(tmp_path, _noisy_pair(23.227, 22.063, 13.5, 24))
    report = evaluate_statistical_evidence(
        path, tmp_path, objective_direction="minimize",
        baseline_method="baseline", proposed_method="proposed",
    )
    assert report.best_vs_baseline_effect > 0
    assert report.approximate_p_value > 0.05
    assert report.is_statistically_significant is False
    assert report.evidence_strength == "inconclusive_not_significant"
    assert report.supports_improvement_claim is False
    assert any("NOT" in warning for warning in report.warnings)


def test_a_clean_separation_is_reported_as_promising(tmp_path: Path) -> None:
    path = _write(tmp_path, _noisy_pair(20.0, 10.0, 1.0, 12))
    report = evaluate_statistical_evidence(
        path, tmp_path, objective_direction="minimize",
        baseline_method="baseline", proposed_method="proposed",
    )
    assert report.is_statistically_significant is True
    assert report.evidence_strength == "statistically_promising"
    assert report.supports_improvement_claim is True
    assert report.standardized_effect_size > 2.0


def test_significant_but_tiny_sample_is_labelled_underpowered(tmp_path: Path) -> None:
    path = _write(tmp_path, _noisy_pair(20.0, 10.0, 0.5, 3))
    report = evaluate_statistical_evidence(
        path, tmp_path, objective_direction="minimize",
        baseline_method="baseline", proposed_method="proposed",
    )
    assert report.evidence_strength == "significant_but_underpowered"
    assert any("underpowered" in warning for warning in report.warnings)


def test_a_worse_proposed_method_is_not_dressed_up(tmp_path: Path) -> None:
    path = _write(tmp_path, _noisy_pair(10.0, 20.0, 1.0, 12))
    report = evaluate_statistical_evidence(
        path, tmp_path, objective_direction="minimize",
        baseline_method="baseline", proposed_method="proposed",
    )
    assert report.best_vs_baseline_effect < 0
    assert report.evidence_strength == "no_improvement_observed"
    assert report.supports_improvement_claim is False
    assert any("contradicted" in warning for warning in report.warnings)


def test_unverified_and_infeasible_rows_are_excluded(tmp_path: Path) -> None:
    rows = _noisy_pair(20.0, 10.0, 1.0, 6)
    for row in rows:
        row["feasible"] = True
        row["verification_ok"] = True
    rows.append(
        {"method": "proposed", "objective": -1000.0, "seed": 99, "feasible": True, "verification_ok": False}
    )
    rows.append(
        {"method": "proposed", "objective": -500.0, "seed": 98, "feasible": False, "verification_ok": True}
    )
    report = evaluate_statistical_evidence(
        path := _write(tmp_path, rows), tmp_path, objective_direction="minimize",
        baseline_method="baseline", proposed_method="proposed",
    )
    assert path.exists()
    proposed = next(stat for stat in report.method_statistics if stat.method == "proposed")
    assert proposed.n == 6, "excluded rows must not inflate the sample"
    assert proposed.mean == 10.0
    assert any("verification" in warning for warning in report.warnings)


def test_single_method_cannot_support_a_comparison(tmp_path: Path) -> None:
    path = _write(tmp_path, [{"method": "only", "objective": float(i)} for i in range(8)])
    report = evaluate_statistical_evidence(path, tmp_path, objective_direction="minimize")
    assert report.evidence_strength == "single_method_no_comparison"
    assert report.supports_improvement_claim is False


def test_missing_results_and_columns_are_handled(tmp_path: Path) -> None:
    missing = evaluate_statistical_evidence(tmp_path / "nope.csv", tmp_path)
    assert missing.evidence_strength == "missing_results"
    path = _write(tmp_path, [{"foo": 1}])
    assert evaluate_statistical_evidence(path, tmp_path).evidence_strength == "insufficient_columns"
