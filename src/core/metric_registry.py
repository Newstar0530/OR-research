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
    ]:
        registry.register(spec.key, spec)
    return registry

