from pathlib import Path

import pandas as pd

from src.utils.result_evaluator import evaluate_results


def test_generic_result_evaluator(tmp_path: Path) -> None:
    path = tmp_path / "results.csv"
    pd.DataFrame(
        [
            {"method": "baseline", "objective": 10.0, "runtime_seconds": 0.1, "seed": 1},
            {"method": "proposed", "objective": 8.0, "runtime_seconds": 0.2, "seed": 1},
        ]
    ).to_csv(path, index=False)
    report = evaluate_results(path)
    assert "Hypothesis Checks" in report
    assert "baseline" in report
