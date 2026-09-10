"""End-to-end: the QUBO study runs, and its verdicts distinguish cost from defect."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.core.experiment_contract import validate_experiment_contract
from src.execution.result_parser import parse_summary_json
from src.execution.sandbox_runner import SandboxRunner


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = PROJECT_ROOT / "templates" / "qubo_transformation_template.py"

SHRINK = {
    'INSTANCE_FAMILIES = ["knapsack", "multi_knapsack", "set_cover"]':
        'INSTANCE_FAMILIES = ["knapsack", "set_cover"]',
    "INSTANCE_SIZES = [6, 8]": "INSTANCE_SIZES = [6]",
    "SEED_LIST = [0, 1, 2]": "SEED_LIST = [0, 1]",
    "PENALTY_SCALES = [0.001, 0.01, 1.0]": "PENALTY_SCALES = [0.0001, 1.0]",
}


@pytest.fixture(scope="module")
def study(tmp_path_factory) -> tuple[Path, dict, pd.DataFrame]:
    pytest.importorskip("scipy")
    work_dir = tmp_path_factory.mktemp("qubo_study")
    code = TEMPLATE.read_text(encoding="utf-8")
    for source, target in SHRINK.items():
        assert source in code, f"the template no longer has the knob `{source}`"
        code = code.replace(source, target)
    script = work_dir / "experiment.py"
    script.write_text(code, encoding="utf-8")
    execution = SandboxRunner(timeout_seconds=1800).run_python(script, work_dir)
    assert execution.status == "success", execution.stderr[-3000:]
    return work_dir, parse_summary_json(execution.stdout), pd.read_csv(work_dir / "results.csv")


def test_the_study_produces_its_artifacts(study) -> None:
    work_dir, _, _ = study
    assert (work_dir / "results.csv").exists()
    assert (work_dir / "qubo_report.md").exists()
    assert list((work_dir / "figures").glob("*.png"))
    assert list((work_dir / "instances").glob("*.json"))


def test_the_derived_penalty_encoding_is_sound(study) -> None:
    _, summary, df = study
    assert summary["load_bearing_defects"] == 0
    assert summary["derived_penalty_sound_share"] == 1.0
    derived = df[df["method"] == "qubo_derived_penalty"]
    assert not derived.empty
    assert (~derived["is_defect"].astype(bool)).all()
    assert derived["verification_ok"].all()


def test_the_control_penalty_actually_breaks_the_encoding(study) -> None:
    _, _, df = study
    controls = df[df["is_control_condition"].astype(bool)]
    assert not controls.empty
    assert (controls["qubo_verdict"] == "penalty_too_small").any(), (
        "a penalty ten thousand times below the bound must break something"
    )


def test_an_infeasible_bitstring_is_never_reported_as_a_solution(study) -> None:
    _, _, df = study
    claims = df[df["solver_status"].isin(["optimal", "feasible"])]
    assert claims["feasible"].all()
    broken = df[df["qubo_verdict"] == "penalty_too_small"]
    assert (broken["solver_status"] == "no_solution").all()
    assert not broken["feasible"].any()


def test_optimality_is_only_claimed_from_a_proved_ground_state(study) -> None:
    _, _, df = study
    optimal = df[df["solver_status"] == "optimal"]
    assert optimal["ground_state_proved"].astype(bool).all()
    assert (df["qubo_verdict"] != "energy_encoding_mismatch").all() | df[
        "ground_state_proved"
    ].astype(bool).all()


def test_the_encoding_cost_is_reported(study) -> None:
    _, summary, df = study
    assert summary["max_bits_observed"] > 0
    assert (df["n_bits"] >= df["n_grid_bits"]).all()
    assert (df["n_slack_bits"] >= 0).all()
    # Slack registers exist because the source model has inequalities.
    assert (df["n_slack_bits"] > 0).any()


def test_results_satisfy_the_strict_solver_contract(study) -> None:
    work_dir, _, _ = study
    report = validate_experiment_contract(work_dir)
    assert report.contract_name == "solver"
    assert report.passed is True, report.failures


def test_domain_columns_are_present(study) -> None:
    _, _, df = study
    for column in (
        "qubo_verdict",
        "penalty",
        "penalty_to_objective_ratio",
        "n_bits",
        "n_slack_bits",
        "ground_state_proved",
        "discretisation_loss",
        "is_control_condition",
    ):
        assert column in df.columns


def test_a_broken_template_reports_a_real_traceback(tmp_path: Path) -> None:
    code = TEMPLATE.read_text(encoding="utf-8").replace(
        'INSTANCE_FAMILIES = ["knapsack", "multi_knapsack", "set_cover"]',
        'INSTANCE_FAMILIES = ["not_a_family"]',
    )
    script = tmp_path / "experiment.py"
    script.write_text(code, encoding="utf-8")
    execution = SandboxRunner(timeout_seconds=600).run_python(script, tmp_path)
    assert execution.status == "failed"
    assert "not_a_family" in execution.stderr
