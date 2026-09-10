"""OR benchmarks are paired designs, and the analysis has to know that.

Every method solves the same instances, so instance difficulty is shared
variation -- usually the largest source of spread in a benchmark. An unpaired
test throws that away and assumes an independence the design does not have.

The first test below is the one that matters: the same data, analysed both
ways, gives a different answer. The rest pin the checks that keep the paired
path from claiming a design the data does not support.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.utils.statistical_evidence import (
    detect_paired_sample,
    evaluate_statistical_evidence,
)


def _write(tmp_path: Path, rows: list[dict]) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "results.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _benchmark(n: int = 20, effect: float = 1.0, instance_spread: float = 40.0) -> list[dict]:
    """A realistic benchmark: instances differ wildly, the effect is small.

    Instance hardness dominates the raw numbers, which is exactly the situation
    where pairing is the difference between seeing an effect and not.
    """

    rng = np.random.default_rng(7)
    hardness = rng.normal(100.0, instance_spread, n)
    noise = rng.normal(0.0, 0.5, (2, n))
    rows: list[dict] = []
    for index in range(n):
        rows.append(
            {
                "instance_id": f"i{index:03d}",
                "seed": 1,
                "method": "baseline",
                "objective": float(hardness[index] + noise[0, index]),
            }
        )
        rows.append(
            {
                "instance_id": f"i{index:03d}",
                "seed": 1,
                "method": "proposed",
                "objective": float(hardness[index] - effect + noise[1, index]),
            }
        )
    return rows


def _report(tmp_path: Path, rows: list[dict], **kwargs):
    return evaluate_statistical_evidence(
        _write(tmp_path, rows),
        tmp_path,
        objective_direction=kwargs.pop("objective_direction", "minimize"),
        baseline_method="baseline",
        proposed_method="proposed",
        **kwargs,
    )


# -- the reason this exists ------------------------------------------------


def test_the_paired_design_finds_an_effect_the_unpaired_test_would_miss(tmp_path: Path) -> None:
    """A consistent 1.0 improvement, buried under a spread of 40 between instances.

    Unpaired, the instance spread is noise and swamps the effect. Paired, it
    cancels. Same data, opposite conclusion -- which is why the design has to be
    detected rather than assumed away.
    """

    rows = _benchmark(n=20, effect=1.0, instance_spread=40.0)
    report = _report(tmp_path, rows)

    assert report.design == "paired"
    assert report.is_statistically_significant is True
    assert report.supports_improvement_claim is True

    # The same rows with the pairing destroyed: distinct instances per method,
    # so nothing lines up and the module must refuse to pair.
    unpaired_rows = [dict(row) for row in rows]
    for row in unpaired_rows:
        if row["method"] == "proposed":
            row["instance_id"] = row["instance_id"] + "b"
    unpaired = _report(tmp_path / "unpaired", unpaired_rows)

    assert unpaired.design == "unpaired"
    assert unpaired.is_statistically_significant is False
    assert unpaired.approximate_p_value > report.approximate_p_value


def test_the_paired_effect_is_the_mean_per_instance_difference(tmp_path: Path) -> None:
    report = _report(tmp_path, _benchmark(n=20, effect=1.0))

    assert report.paired_mean_difference == pytest.approx(1.0, abs=0.35)
    assert report.best_vs_baseline_effect == report.paired_mean_difference
    assert report.n_pairs == 20
    assert report.n_wins + report.n_losses + report.n_ties == 20


def test_the_reported_test_matches_the_detected_design(tmp_path: Path) -> None:
    paired = _report(tmp_path, _benchmark(n=12))
    assert paired.test_used in {"paired_t", "paired_normal_approximation"}
    assert paired.pairing_key == ["instance_id", "seed"]

    rows = [
        {"method": "baseline", "objective": float(v)} for v in range(10, 20)
    ] + [{"method": "proposed", "objective": float(v)} for v in range(5, 15)]
    unpaired = _report(tmp_path / "u", rows)
    assert unpaired.design == "unpaired"
    assert unpaired.test_used in {"welch_t", "normal_approximation"}
    assert unpaired.n_pairs is None


# -- pairing is checked, not assumed ---------------------------------------


def test_pairing_is_refused_when_no_identity_column_exists(tmp_path: Path) -> None:
    rows = [{"method": "baseline", "objective": float(v)} for v in range(10, 20)]
    rows += [{"method": "proposed", "objective": float(v)} for v in range(5, 15)]
    report = _report(tmp_path, rows)

    assert report.design == "unpaired"
    assert any("cannot be paired" in warning for warning in report.warnings)


def test_pairing_is_refused_when_a_key_repeats_within_one_method(tmp_path: Path) -> None:
    """An ambiguous pairing is worse than none: it invents a design silently."""

    rows = _benchmark(n=8)
    rows.append({"instance_id": "i000", "seed": 1, "method": "proposed", "objective": 1.0})
    report = _report(tmp_path, rows)

    assert report.design == "unpaired"
    assert any("ambiguous" in warning for warning in report.warnings)


def test_unmatched_instances_are_dropped_and_reported(tmp_path: Path) -> None:
    rows = _benchmark(n=10)
    rows.append({"instance_id": "solo", "seed": 1, "method": "proposed", "objective": 1.0})
    report = _report(tmp_path, rows)

    assert report.design == "paired"
    assert report.n_pairs == 10
    assert any("no counterpart" in warning for warning in report.warnings)
    assert any("hard ones" in warning for warning in report.warnings)


def test_too_few_shared_instances_falls_back_rather_than_pairing_one(tmp_path: Path) -> None:
    rows = [
        {"instance_id": "a", "method": "baseline", "objective": 10.0},
        {"instance_id": "a", "method": "proposed", "objective": 9.0},
        {"instance_id": "b", "method": "baseline", "objective": 11.0},
        {"instance_id": "c", "method": "proposed", "objective": 8.0},
    ]
    report = _report(tmp_path, rows)
    assert report.design == "unpaired"
    assert any("not enough to pair" in warning for warning in report.warnings)


def test_pairing_on_seed_alone_says_that_it_was_inferred(tmp_path: Path) -> None:
    """It is right only if the same seed reproduces the same instance."""

    rows = []
    for seed in range(8):
        rows.append({"seed": seed, "method": "baseline", "objective": 10.0 + seed})
        rows.append({"seed": seed, "method": "proposed", "objective": 9.0 + seed})
    report = _report(tmp_path, rows)

    assert report.design == "paired"
    assert report.pairing_key == ["seed"]
    assert any("inferred from" in warning for warning in report.warnings)


def test_direction_is_respected_when_maximising(tmp_path: Path) -> None:
    rows = []
    for index in range(10):
        rows.append({"instance_id": f"i{index}", "method": "baseline", "objective": 10.0 + index})
        rows.append({"instance_id": f"i{index}", "method": "proposed", "objective": 12.0 + index})
    report = _report(tmp_path, rows, objective_direction="maximize")

    assert report.design == "paired"
    assert report.paired_mean_difference == pytest.approx(2.0)
    assert report.n_wins == 10
    assert report.supports_improvement_claim is False, "a constant gap has no variance to test"


# -- the traps a paired test brings with it --------------------------------


def test_an_identical_difference_on_every_instance_is_flagged_not_tested(tmp_path: Path) -> None:
    """Two real search methods almost never differ by a constant everywhere.

    When they do, the likely cause is that one is computed from the other --
    which was exactly the defect in the original synthetic template.
    """

    rows = []
    for index in range(12):
        rows.append({"instance_id": f"i{index}", "method": "baseline", "objective": 10.0 + index})
        rows.append({"instance_id": f"i{index}", "method": "proposed", "objective": 9.0 + index})
    report = _report(tmp_path, rows)

    assert report.test_used == "not_applicable_zero_variance"
    assert report.is_statistically_significant is None
    assert report.supports_improvement_claim is False
    assert any("check that the experiment is not computing one method" in w for w in report.warnings)


def test_a_benchmark_that_mostly_ties_says_how_little_it_rests_on(tmp_path: Path) -> None:
    """Both methods proving the optimum on easy instances is a tie, not evidence."""

    rows = []
    for index in range(20):
        shared = 100.0 + index
        proposed = shared - (1.0 if index < 4 else 0.0)
        rows.append({"instance_id": f"i{index}", "method": "baseline", "objective": shared})
        rows.append({"instance_id": f"i{index}", "method": "proposed", "objective": proposed})
    report = _report(tmp_path, rows)

    assert report.n_ties == 16
    assert report.n_wins == 4
    assert any("exact ties" in warning for warning in report.warnings)
    assert any("rests on 4 instance" in warning for warning in report.warnings)


def test_the_underpowered_warning_counts_pairs_not_rows(tmp_path: Path) -> None:
    """Twenty rows over three instances is three pieces of evidence, not twenty."""

    rows = []
    for index in range(3):
        rows.append({"instance_id": f"i{index}", "method": "baseline", "objective": 10.0 + index})
        rows.append({"instance_id": f"i{index}", "method": "proposed", "objective": 8.0 + index * 1.5})
    report = _report(tmp_path, rows)

    assert report.n_pairs == 3
    assert any("3 paired instance(s)" in warning for warning in report.warnings)


# -- the unit under the entry point ----------------------------------------


def test_detect_paired_sample_orients_differences_so_positive_means_better() -> None:
    df = pd.DataFrame(
        [
            {"instance_id": "a", "method": "baseline", "objective": 10.0},
            {"instance_id": "a", "method": "proposed", "objective": 7.0},
            {"instance_id": "b", "method": "baseline", "objective": 20.0},
            {"instance_id": "b", "method": "proposed", "objective": 25.0},
        ]
    )
    minimising, _ = detect_paired_sample(df, "objective", "baseline", "proposed", maximize=False)
    assert minimising.differences == [3.0, -5.0]

    maximising, _ = detect_paired_sample(df, "objective", "baseline", "proposed", maximize=True)
    assert maximising.differences == [-3.0, 5.0]
