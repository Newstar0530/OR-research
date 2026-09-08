from pathlib import Path

import pandas as pd

from src.core.experiment_contract import validate_experiment_contract
from src.core.method_registry import build_method_registry
from src.core.metric_registry import build_metric_registry
from src.core.research_protocol import default_or_protocol
from src.core.template_registry import select_template


def test_default_protocol_is_inspectable() -> None:
    protocol = default_or_protocol()
    assert protocol.stage_names()[0] == "domain_classification"
    assert "automated_review" in protocol.stage_names()
    assert "Human" not in protocol.name


def test_registries_have_generic_entries() -> None:
    assert "milp_baseline" in build_method_registry().keys()
    assert "objective" in build_metric_registry().keys()


def test_template_selection_falls_back_to_generic() -> None:
    spec = select_template("unknown_profile", Path.cwd())
    assert spec.key == "generic_or"
    assert spec.path.exists()


def test_experiment_contract_validation(tmp_path: Path) -> None:
    pd.DataFrame(
        [
            {
                "instance_id": "i1",
                "method": "baseline",
                "objective": 1.0,
                "runtime_seconds": 0.1,
                "seed": 42,
            },
            {
                "instance_id": "i1",
                "method": "proposed",
                "objective": 0.8,
                "runtime_seconds": 0.2,
                "seed": 42,
            },
        ]
    ).to_csv(tmp_path / "results.csv", index=False)
    report = validate_experiment_contract(tmp_path)
    assert report.passed is True
