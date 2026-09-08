from pathlib import Path

import pandas as pd

from src.utils.autonomy_readiness import evaluate_autonomy_readiness


def _write_core_artifacts(root: Path) -> None:
    for name in [
        "selected_idea.json",
        "novelty_report.md",
        "model_draft.md",
        "model_critique.md",
        "algorithm_plan.md",
        "sensitivity_report.md",
        "final_report.md",
    ]:
        (root / name).write_text("ok\n", encoding="utf-8")
    (root / "automated_review.md").write_text("## Scores\n- Overall: 7/10\n", encoding="utf-8")
    (root / "claim_check.md").write_text("No high-risk claim patterns were detected.\n", encoding="utf-8")
    (root / "novelty_report.md").write_text("Verified local literature evidence was indexed.\n", encoding="utf-8")
    (root / "experiment_contract.md").write_text("# Experiment Contract Validation\n\nStatus: passed\n", encoding="utf-8")


def test_autonomy_readiness_review_ready(tmp_path: Path) -> None:
    _write_core_artifacts(tmp_path)
    pd.DataFrame(
        [
            {"instance_id": "a", "method": "baseline", "objective": 10, "runtime_seconds": 0.1, "seed": 1, "gap": 0.0},
            {"instance_id": "a", "method": "proposed", "objective": 9, "runtime_seconds": 0.2, "seed": 2, "gap": -0.1},
        ]
    ).to_csv(tmp_path / "results.csv", index=False)

    report = evaluate_autonomy_readiness(tmp_path)

    assert report.score >= 80
    assert report.level == "review-ready"
    assert not report.blockers


def test_autonomy_readiness_blocks_missing_results(tmp_path: Path) -> None:
    _write_core_artifacts(tmp_path)

    report = evaluate_autonomy_readiness(tmp_path)

    assert report.level == "blocked"
    assert any("results.csv" in item for item in report.blockers)
