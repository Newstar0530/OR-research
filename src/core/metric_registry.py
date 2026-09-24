from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from src.core.registry import Registry


Direction = Literal["minimize", "maximize", "target"]


class MetricSpec(BaseModel):
    key: str
    direction: Direction
    description: str
    required_column: str | None = None


def build_metric_registry() -> Registry[MetricSpec]:
    registry: Registry[MetricSpec] = Registry()
    for spec in [
        MetricSpec(key="objective", direction="minimize", required_column="objective", description="Primary objective value."),
        MetricSpec(key="runtime_seconds", direction="minimize", required_column="runtime_seconds", description="Wall-clock runtime."),
        MetricSpec(key="gap", direction="minimize", required_column="gap", description="Relative or absolute gap to baseline or exact solution."),
        MetricSpec(key="feasibility", direction="maximize", required_column="feasible", description="Share of feasible solutions."),
        MetricSpec(key="robustness", direction="maximize", required_column="robustness_metric", description="Stability under perturbation or uncertainty."),
        # Formulation metrics. These are not columns in results.csv: they score
        # the model against the requirements it was supposed to encode, before
        # anything is solved. A run can be perfect on every metric above while
        # failing these, because solving the wrong problem well is still wrong.
        MetricSpec(key="missing_constraint_rate", direction="minimize", description="Share of stated requirements that no constraint in the formulation imposes."),
        MetricSpec(key="requirement_coverage", direction="maximize", description="Share of stated requirements the formulation appears to impose."),
        MetricSpec(key="wrong_objective_rate", direction="minimize", description="1 when the declared objective direction contradicts the required one, else 0."),
    ]:
        registry.register(spec.key, spec)
    return registry

