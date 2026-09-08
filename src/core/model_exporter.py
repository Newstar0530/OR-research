from __future__ import annotations

import json
from pathlib import Path

from src.core.model_ir import OptimizationModelIR


def export_model_skeletons(ir: OptimizationModelIR, output_dir: str | Path) -> dict[str, str]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    pyomo_path = root / "model_export_pyomo.py"
    ortools_path = root / "model_export_ortools.py"
    diagnostics_path = root / "model_export_diagnostics.md"
    pyomo_path.write_text(_pyomo_skeleton(ir), encoding="utf-8")
    ortools_path.write_text(_ortools_skeleton(ir), encoding="utf-8")
    diagnostics_path.write_text(_diagnostics(ir), encoding="utf-8")
    (root / "model_export_manifest.json").write_text(
        json.dumps(
            {
                "pyomo": pyomo_path.name,
                "ortools": ortools_path.name,
                "diagnostics": diagnostics_path.name,
                "status": "skeleton_requires_human_completion",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {"pyomo": str(pyomo_path), "ortools": str(ortools_path), "diagnostics": str(diagnostics_path)}


def _pyomo_skeleton(ir: OptimizationModelIR) -> str:
    lines = [
        '"""Pyomo skeleton generated from model_ir.json.',
        "Needs human verification before solving.",
        '"""',
        "",
        "try:",
        "    import pyomo.environ as pyo",
        "except ModuleNotFoundError as exc:",
        '    raise RuntimeError("Pyomo is not installed. Install pyomo or use another exporter.") from exc',
        "",
        "",
        "def build_model(data: dict | None = None):",
        "    data = data or {}",
        "    model = pyo.ConcreteModel()",
    ]
    for component in ir.sets:
        lines.append(f"    # Set {component.name}: {component.description}")
    for component in ir.parameters:
        lines.append(f"    # Param {component.name}: {component.description}")
    for component in ir.decision_variables:
        lines.append(f"    # Var {component.name}: {component.description}")
    lines.extend(
        [
            f"    # Objective sense from IR: {ir.objective_sense}",
            f"    # Objective draft: {ir.objective or 'not extracted'}",
            "    # TODO: declare sets, parameters, variables, objective, and constraints.",
            "    return model",
            "",
        ]
    )
    return "\n".join(lines)


def _ortools_skeleton(ir: OptimizationModelIR) -> str:
    return "\n".join(
        [
            '"""OR-Tools skeleton generated from model_ir.json.',
            "Needs human verification before solving.",
            '"""',
            "",
            "try:",
            "    from ortools.sat.python import cp_model",
            "except ModuleNotFoundError as exc:",
            '    raise RuntimeError("OR-Tools is not installed. Install ortools or use another exporter.") from exc',
            "",
            "",
            "def build_model(data: dict | None = None):",
            "    data = data or {}",
            "    model = cp_model.CpModel()",
            f"    # Objective sense from IR: {ir.objective_sense}",
            "    # TODO: map IR variables and constraints into CP-SAT constructs.",
            "    return model",
            "",
        ]
    )


def _diagnostics(ir: OptimizationModelIR) -> str:
    lines = ["# Model Export Diagnostics", "", "- status: skeleton_requires_human_completion", ""]
    lines.append("## Extracted Counts")
    lines.append(f"- sets: {len(ir.sets)}")
    lines.append(f"- parameters: {len(ir.parameters)}")
    lines.append(f"- decision_variables: {len(ir.decision_variables)}")
    lines.append(f"- constraints: {len(ir.constraints)}")
    lines.append("")
    lines.append("## Warnings")
    warnings = list(ir.extraction_warnings)
    if ir.objective_sense == "unknown":
        warnings.append("Objective sense is unknown or ambiguous.")
    lines.extend(f"- {item}" for item in warnings or ["None."])
    lines.append("")
    lines.append("## Human Tasks")
    lines.append("- Verify index dimensions and domains.")
    lines.append("- Implement algebraic expressions manually before solving.")
    lines.append("- Confirm solver status, infeasibility, Big-M values, and optimality gaps.")
    return "\n".join(lines) + "\n"
