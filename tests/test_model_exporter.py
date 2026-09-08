from pathlib import Path

from src.core.model_exporter import export_model_skeletons
from src.core.model_ir import build_model_ir


def test_model_exporter_writes_solver_skeletons(tmp_path: Path) -> None:
    ir = build_model_ir("## Decision Variables\n- x_i: binary choice\n## Objective\n- Minimize cost\n## Constraints\n- Capacity: sum x <= 1")

    outputs = export_model_skeletons(ir, tmp_path)

    assert Path(outputs["pyomo"]).exists()
    assert Path(outputs["ortools"]).exists()
    assert (tmp_path / "model_export_manifest.json").exists()
