from __future__ import annotations

import json
from pathlib import Path

from src.core.or_ir import OptimizationModel, VariableSpec


_PYOMO_DOMAIN = {
    "binary": "pyo.Binary",
    "integer": "pyo.Integers",
    "continuous": "pyo.Reals",
}


def export_model_skeletons(ir: OptimizationModel, output_dir: str | Path) -> dict[str, str]:
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
                "model_source": ir.source,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {"pyomo": str(pyomo_path), "ortools": str(ortools_path), "diagnostics": str(diagnostics_path)}


def _pyomo_skeleton(ir: OptimizationModel) -> str:
    """Declare what the IR states; leave the algebra to a human.

    Variables now carry a domain and bounds, so those become real
    declarations rather than comments. Constraints stay commented: their
    expressions are strings the IR never parsed, and emitting them as code
    would produce something that looks executable and is not.
    """

    lines = [
        '"""Pyomo skeleton generated from model_ir.json.',
        "",
        f"Model source: {ir.source}. Variable domains and bounds below are declared by the IR;",
        "objective and constraint expressions are comments, because the IR holds them as text",
        "and never parsed them. Needs human verification before solving.",
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
        index = f" indexed by {component.index}" if component.index else ""
        lines.append(f"    # Set {component.name}:{index} {component.description}".rstrip())
    for component in ir.parameters:
        unit = component.unit or "UNIT NOT STATED"
        lines.append(f"    # Param {component.name} [{unit}]: {component.description}".rstrip())
    if ir.decision_variables:
        lines.append("")
    for component in ir.decision_variables:
        lines.append(f"    # Var {component.name}: {component.description}".rstrip())
        lines.append(f"    {_pyomo_var(component)}")
    lines.append("")
    if ir.objective:
        lines.append(f"    # Objective ({ir.objective.sense}): {ir.objective.rendered()}")
        for term in ir.objective.terms:
            lines.append(f"    #   term {term.expression}: {term.business_meaning}".rstrip())
        sense = "pyo.minimize" if ir.objective.sense == "minimize" else "pyo.maximize"
        lines.append(f"    # model.obj = pyo.Objective(expr=..., sense={sense})")
    else:
        lines.append("    # Objective: none declared, so no direction was chosen.")
    lines.append("")
    for component in ir.constraints:
        tag = f"[{component.source_requirement}] " if component.source_requirement else ""
        body = component.expression or "NOT WRITTEN AS AN EXPRESSION"
        lines.append(f"    # Constraint {tag}{component.name} ({component.enforcement}): {body}")
    lines.extend(
        [
            "    # TODO: write the objective and constraint expressions against the declarations above.",
            "    return model",
            "",
        ]
    )
    return "\n".join(lines)


def _pyomo_var(component: VariableSpec) -> str:
    domain = _PYOMO_DOMAIN.get(component.domain)
    if domain is None:
        return (
            f"# model.{component.name} = pyo.Var(...)  # domain not declared in the IR"
        )
    bounds = ""
    if component.domain != "binary" and (component.lower_bound is not None or component.upper_bound is not None):
        low = component.lower_bound if component.lower_bound is not None else "None"
        high = component.upper_bound if component.upper_bound is not None else "None"
        bounds = f", bounds=({low}, {high})"
    index = f"model.{component.indexed_by[0]}, " if component.indexed_by else ""
    return f"model.{component.name} = pyo.Var({index}domain={domain}{bounds})"


def _ortools_skeleton(ir: OptimizationModel) -> str:
    lines = [
        '"""OR-Tools skeleton generated from model_ir.json.',
        "",
        f"Model source: {ir.source}. Needs human verification before solving.",
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
    ]
    for component in ir.decision_variables:
        if component.domain == "binary":
            lines.append(f'    {component.name} = model.NewBoolVar("{component.name}")')
        elif component.domain == "integer":
            low = component.lower_bound if component.lower_bound is not None else "0"
            high = component.upper_bound if component.upper_bound is not None else "UPPER_BOUND"
            lines.append(f'    # {component.name} = model.NewIntVar({low}, {high}, "{component.name}")')
        else:
            lines.append(
                f"    # {component.name}: {component.domain}; CP-SAT needs an integer encoding for this."
            )
    lines.append(f"    # Objective sense from IR: {ir.objective_sense}")
    lines.append("    # TODO: map IR constraints into CP-SAT constructs.")
    lines.extend(["    return model", ""])
    return "\n".join(lines)


def _diagnostics(ir: OptimizationModel) -> str:
    report = ir.structural_report()
    lines = [
        "# Model Export Diagnostics",
        "",
        "- status: skeleton_requires_human_completion",
        f"- model source: {ir.source}",
        f"- implementable as declared: {report.is_implementable}",
        "",
        "## Declared Counts",
        f"- sets: {len(ir.sets)}",
        f"- parameters: {len(ir.parameters)} ({sum(1 for p in ir.parameters if p.unit)} with a unit)",
        f"- decision_variables: {len(ir.decision_variables)} "
        f"({sum(1 for v in ir.decision_variables if v.domain != 'unspecified')} with a declared domain)",
        f"- constraints: {len(ir.constraints)} "
        f"({sum(1 for c in ir.constraints if c.expression.strip())} written as expressions, "
        f"{sum(1 for c in ir.constraints if c.source_requirement)} citing a requirement)",
        f"- uncertainty entries: {len(ir.uncertainty)}",
        "",
        "## Blocking Gaps",
    ]
    lines.extend(f"- {item}" for item in report.missing or ["None."])
    lines.extend(["", "## Warnings"])
    warnings = report.warnings + list(ir.extraction_warnings)
    if ir.objective_sense == "unknown":
        warnings.append("Objective sense is unknown or ambiguous.")
    lines.extend(f"- {item}" for item in warnings or ["None."])
    lines.extend(
        [
            "",
            "## Human Tasks",
            "- Verify index dimensions and domains.",
            "- Write the algebraic expressions against the declarations before solving.",
            "- Confirm solver status, infeasibility, Big-M values, and optimality gaps.",
        ]
    )
    return "\n".join(lines) + "\n"
