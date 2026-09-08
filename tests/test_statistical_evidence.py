from pathlib import Path

import pandas as pd

from src.utils.statistical_evidence import evaluate_statistical_evidence


def test_statistical_evidence_computes_method_stats(tmp_path: Path) -> None:
    path = tmp_path / "results.csv"
    pd.DataFrame(
        [
            {"method": "baseline", "objective": 10, "seed": 1},
            {"method": "baseline", "objective": 12, "seed": 2},
            {"method": "proposed", "objective": 8, "seed": 1},
            {"method": "proposed", "objective": 9, "seed": 2},
        ]
    ).to_csv(path, index=False)

    report = evaluate_statistical_evidence(path, tmp_path, objective_direction="minimize")

    assert report.best_method == "proposed"
    assert report.best_vs_baseline_effect and report.best_vs_baseline_effect > 0
    assert report.approximate_p_value is not None
    assert (tmp_path / "statistical_evidence.md").exists()
