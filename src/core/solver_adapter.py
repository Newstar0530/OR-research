from __future__ import annotations

import importlib.util
from typing import Literal

from pydantic import BaseModel, Field


ToolStatus = Literal["available", "missing"]


class SolverToolCapability(BaseModel):
    name: str
    package: str
    status: ToolStatus
    use_cases: list[str] = Field(default_factory=list)
    fallback: str
    human_notes: list[str] = Field(default_factory=list)


class SolverToolReport(BaseModel):
    tools: list[SolverToolCapability]

    def to_markdown(self) -> str:
        lines = ["# Solver Tool Report", ""]
        for tool in self.tools:
            lines.append(f"## {tool.name}")
            lines.append(f"- package: `{tool.package}`")
            lines.append(f"- status: {tool.status}")
            lines.append(f"- fallback: {tool.fallback}")
            lines.append("- use_cases:")
            lines.extend(f"  - {item}" for item in tool.use_cases)
            if tool.human_notes:
                lines.append("- human_notes:")
                lines.extend(f"  - {item}" for item in tool.human_notes)
            lines.append("")
        return "\n".join(lines)


TOOL_SPECS = [
    {
        "name": "Pyomo",
        "package": "pyomo",
        "use_cases": ["linear/integer/nonlinear optimization modeling", "solver-independent algebraic models"],
        "fallback": "Use scipy.optimize for small continuous models or template baselines for mock experiments.",
        "human_notes": ["External solvers such as GLPK, CBC, Gurobi, or IPOPT may still be required."],
    },
    {
        "name": "Gurobi",
        "package": "gurobipy",
        "use_cases": ["MILP/MIQP baseline", "high-quality exact optimization benchmark"],
        "fallback": "Use open-source solvers through Pyomo or smaller brute-force/dynamic-programming baselines.",
        "human_notes": ["License configuration must be verified outside the assistant."],
    },
    {
        "name": "OR-Tools",
        "package": "ortools",
        "use_cases": ["routing", "CP-SAT scheduling", "assignment", "vehicle routing"],
        "fallback": "Use networkx/scipy heuristics or local template dispatching baselines.",
        "human_notes": ["Check problem-specific modeling assumptions and time limits."],
    },
    {
        "name": "SciPy",
        "package": "scipy",
        "use_cases": ["continuous optimization", "statistics", "numerical sensitivity analysis"],
        "fallback": "Use numpy grid search or closed-form baselines for simple experiments.",
        "human_notes": ["SciPy solvers do not directly replace MILP solvers."],
    },
    {
        "name": "NetworkX",
        "package": "networkx",
        "use_cases": ["network reliability proxies", "shortest paths", "centrality", "graph experiments"],
        "fallback": "Use simple adjacency-list algorithms for small graphs.",
        "human_notes": ["Exact network reliability is often computationally hard; distinguish proxy metrics from exact reliability."],
    },
]


def detect_solver_tools() -> SolverToolReport:
    tools = []
    for spec in TOOL_SPECS:
        status: ToolStatus = "available" if importlib.util.find_spec(spec["package"]) else "missing"
        tools.append(SolverToolCapability(status=status, **spec))
    return SolverToolReport(tools=tools)


def recommended_tools_for_profile(profile: str, report: SolverToolReport | None = None) -> list[str]:
    report = report or detect_solver_tools()
    available = {tool.name.lower(): tool for tool in report.tools if tool.status == "available"}
    if profile == "network_analysis":
        return _available_or_fallback(["NetworkX", "SciPy"], available)
    if profile == "scheduling":
        return _available_or_fallback(["OR-Tools", "Pyomo", "Gurobi"], available)
    if profile == "inventory":
        return _available_or_fallback(["SciPy", "Pyomo"], available)
    if profile == "stochastic_models":
        return _available_or_fallback(["SciPy", "NetworkX"], available)
    return _available_or_fallback(["Pyomo", "Gurobi", "SciPy"], available)


def _available_or_fallback(names: list[str], available: dict[str, SolverToolCapability]) -> list[str]:
    selected = [name for name in names if name.lower() in available]
    return selected or ["template_baseline"]
