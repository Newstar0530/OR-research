"""End-to-end: the transformation study runs, and its verdicts mean something."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.core.experiment_contract import validate_experiment_contract
from src.execution.result_parser import parse_summary_json
from src.execution.sandbox_runner import SandboxRunner


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = PROJECT_ROOT / "templates" / "bilevel_transformation_template.py"

SHRINK = {
    "INSTANCE_SIZES = [2, 3]": "INSTANCE_SIZES = [2]",
    "SEED_LIST = [0, 1, 2]": "SEED_LIST = [0, 1]",
    "BIG_M_SCALES = [0.1, 0.5, 1.0, 10.0]": "BIG_M_SCALES = [0.1, 1.0]",
    "UNIFORM_BIG_M_VALUES = [1000.0, 1000000.0]": "UNIFORM_BIG_M_VALUES = [1000000.0]",
}


@pytest.fixture(scope="module")
def study(tmp_path_factory) -> tuple[Path, dict, pd.DataFrame]:
    pytest.importorskip("scipy")
    work_dir = tmp_path_factory.mktemp("bilevel_study")
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
    assert (work_dir / "equivalence_report.md").exists()
    assert list((work_dir / "figures").glob("*.png"))
    assert list((work_dir / "instances").glob("*.json"))


def test_every_instance_has_a_proved_reference_optimum(study) -> None:
    _, summary, df = study
    assert summary["oracle_unavailable"] == 0
    assert df["known_optimum"].notna().all()
    assert (df["known_optimum_source"] == "vertex_enumeration_proved").all()


def test_the_load_bearing_transformations_hold(study) -> None:
    """KKT and the rigorously derived Big-M must be exact on every instance."""

    _, summary, df = study
    assert summary["load_bearing_failures"] == 0
    assert summary["derived_bigM_equivalent_share"] == 1.0
    load_bearing = df[~df["is_control_condition"].astype(bool)]
    assert (load_bearing["equivalence_verdict"] == "equivalent").all()
    assert load_bearing["verification_ok"].all()
    assert set(load_bearing["method"]) == {"kkt_pattern_enumeration", "mi01_derived_bigM"}


def test_the_control_conditions_actually_break(study) -> None:
    """If a deliberately bad Big-M never fails, the experiment measures nothing."""

    _, _, df = study
    controls = df[df["is_control_condition"].astype(bool)]
    assert not controls.empty
    verdicts = set(controls["equivalence_verdict"])
    assert verdicts != {"equivalent"}, "no control condition failed; the study has no signal"
    too_small = controls[controls["big_m_setting"] == "derived x 0.1"]
    assert (too_small["equivalence_verdict"] != "equivalent").all()


def test_a_model_that_is_not_bilevel_feasible_is_never_marked_feasible(study) -> None:
    _, _, df = study
    claims_a_solution = df[df["solver_status"].isin(["optimal", "feasible"])]
    assert claims_a_solution["solution_bilevel_feasible"].all()
    assert claims_a_solution["feasible"].all()


def test_results_satisfy_the_strict_solver_contract(study) -> None:
    work_dir, _, _ = study
    report = validate_experiment_contract(work_dir)
    assert report.contract_name == "solver"
    assert report.passed is True, report.failures


def test_domain_columns_are_present_without_shadowing_the_schema(study) -> None:
    _, _, df = study
    for column in (
        "equivalence_verdict",
        "big_m_setting",
        "big_m_slack",
        "big_m_dual",
        "n_binaries",
        "is_control_condition",
        "difficulty",
    ):
        assert column in df.columns
    assert df["objective"].notna().any()
    assert df["gap_to_known_optimum"].notna().any()


def test_a_broken_template_reports_a_real_traceback(tmp_path: Path) -> None:
    code = TEMPLATE.read_text(encoding="utf-8").replace(
        'DIFFICULTIES = ["balanced", "coupled", "degenerate", "large_dual"]',
        'DIFFICULTIES = ["not_a_difficulty"]',
    )
    script = tmp_path / "experiment.py"
    script.write_text(code, encoding="utf-8")
    execution = SandboxRunner(timeout_seconds=600).run_python(script, tmp_path)
    assert execution.status == "failed"
    assert "not_a_difficulty" in execution.stderr
