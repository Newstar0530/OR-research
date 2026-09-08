from __future__ import annotations

from pydantic import BaseModel, Field

from src.core.registry import Registry


class MethodSpec(BaseModel):
    key: str
    family: str
    description: str
    requires_packages: list[str] = Field(default_factory=list)
    suitable_for: list[str] = Field(default_factory=list)
    baseline_role: bool = False


def build_method_registry() -> Registry[MethodSpec]:
    registry: Registry[MethodSpec] = Registry()
    for spec in [
        MethodSpec(
            key="milp_baseline",
            family="exact_optimization",
            description="MILP/MINLP baseline when a mathematical formulation and solver are available.",
            requires_packages=["pyomo or scipy.optimize where applicable"],
            suitable_for=["optimization", "scheduling", "inventory", "network_analysis"],
            baseline_role=True,
        ),
        MethodSpec(
            key="simulation_baseline",
            family="simulation",
            description="Monte Carlo or discrete-event simulation baseline for uncertain systems.",
            requires_packages=["numpy", "pandas"],
            suitable_for=["stochastic_models", "inventory", "network_analysis", "ai_for_ie"],
            baseline_role=True,
        ),
        MethodSpec(
            key="greedy_heuristic",
            family="heuristic",
            description="Constructive greedy heuristic with transparent ranking rule.",
            requires_packages=["numpy", "pandas"],
            suitable_for=["optimization", "scheduling", "inventory", "network_analysis"],
            baseline_role=True,
        ),
        MethodSpec(
            key="local_search",
            family="metaheuristic",
            description="Neighborhood improvement procedure for combinatorial solutions.",
            requires_packages=["numpy"],
            suitable_for=["optimization", "scheduling", "heuristics_metaheuristics"],
        ),
        MethodSpec(
            key="sensitivity_sweep",
            family="analysis",
            description="Controlled parameter sweep with repeated seeds and robustness metrics.",
            requires_packages=["numpy", "pandas", "matplotlib"],
            suitable_for=["optimization", "scheduling", "inventory", "network_analysis", "stochastic_models"],
        ),
    ]:
        registry.register(spec.key, spec)
    return registry

