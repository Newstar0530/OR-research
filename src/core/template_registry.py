from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from src.core.registry import Registry


class TemplateSpec(BaseModel):
    key: str
    path: Path
    problem_families: list[str]
    description: str


def build_template_registry(project_root: Path) -> Registry[TemplateSpec]:
    template_dir = project_root / "templates"
    registry: Registry[TemplateSpec] = Registry()
    registry.register(
        "generic_or",
        TemplateSpec(
            key="generic_or",
            path=template_dir / "generic_or_template.py",
            problem_families=["optimization", "heuristics_metaheuristics", "ai_for_ie"],
            description="Generic baseline-vs-proposed computational OR experiment.",
        ),
    )
    registry.register(
        "scheduling",
        TemplateSpec(
            key="scheduling",
            path=template_dir / "scheduling_template.py",
            problem_families=["scheduling"],
            description="Generic scheduling experiment scaffold.",
        ),
    )
    registry.register(
        "inventory",
        TemplateSpec(
            key="inventory",
            path=template_dir / "inventory_template.py",
            problem_families=["inventory"],
            description="Generic inventory experiment scaffold.",
        ),
    )
    registry.register(
        "network_analysis",
        TemplateSpec(
            key="network_analysis",
            path=template_dir / "network_analysis_template.py",
            problem_families=["network_analysis"],
            description="Generic network experiment scaffold.",
        ),
    )
    registry.register(
        "binary_program",
        TemplateSpec(
            key="binary_program",
            path=template_dir / "binary_program_template.py",
            problem_families=["binary_programming", "mixed_integer_programming"],
            description=(
                "Real MI 0-1 experiment: reproducible instances, exact and heuristic solvers, "
                "independently verified solutions, gap to a proved optimum."
            ),
        ),
    )
    registry.register(
        "bilevel_transformation",
        TemplateSpec(
            key="bilevel_transformation",
            path=template_dir / "bilevel_transformation_template.py",
            problem_families=["bilevel_programming", "transformation_verification"],
            description=(
                "Verifies that the bilevel -> KKT -> MI 0-1 chain preserves the optimum, against "
                "an oracle that uses neither KKT nor Big-M, and locates the Big-M threshold."
            ),
        ),
    )
    registry.register(
        "qubo_transformation",
        TemplateSpec(
            key="qubo_transformation",
            path=template_dir / "qubo_transformation_template.py",
            problem_families=["qubo", "quantum_optimization"],
            description=(
                "Checks that the QUBO encoding preserves the optimum, separating the cost of "
                "discretisation from the correctness of the penalty, and reports the bit cost."
            ),
        ),
    )
    registry.register(
        "stochastic_models",
        TemplateSpec(
            key="stochastic_models",
            path=template_dir / "stochastic_template.py",
            problem_families=["stochastic_models"],
            description="Generic stochastic simulation scaffold.",
        ),
    )
    return registry


def select_template(
    profile_name: str,
    project_root: Path,
    template_key: str | None = None,
) -> TemplateSpec:
    """Pick an experiment scaffold.

    An explicit `template_key` from the config always wins, so a run can opt
    into the solver-backed experiment without changing domain classification.
    """

    registry = build_template_registry(project_root)
    if template_key:
        try:
            return registry.get(template_key)
        except KeyError as exc:
            raise KeyError(
                f"Unknown experiment_template `{template_key}`. Known: {registry.keys()}"
            ) from exc
    for spec in registry.values():
        if profile_name in spec.problem_families:
            return spec
    return registry.get("generic_or")

