from __future__ import annotations

from pydantic import BaseModel, Field


class DomainToolSpec(BaseModel):
    profile: str
    template_key: str
    primary_metric: str = "objective"
    objective_direction: str = "minimize"
    metrics: list[str] = Field(default_factory=list)
    mutation_hints: list[str] = Field(default_factory=list)
    human_checks: list[str] = Field(default_factory=list)


DOMAIN_TOOL_SPECS = {
    "optimization": DomainToolSpec(
        profile="optimization",
        template_key="generic_or",
        metrics=["objective", "runtime_seconds", "gap", "feasible"],
        mutation_hints=["constraint tightness", "problem size", "stronger baseline", "ablation"],
        human_checks=["verify objective direction", "check feasibility and boundedness"],
    ),
    "heuristics_metaheuristics": DomainToolSpec(
        profile="heuristics_metaheuristics",
        template_key="generic_or",
        metrics=["objective", "runtime_seconds", "feasible"],
        mutation_hints=["local search intensity", "random seed", "ablation", "stress test"],
        human_checks=["compare against exact small-instance baseline", "report multiple seeds"],
    ),
    "binary_program": DomainToolSpec(
        profile="binary_program",
        template_key="binary_program",
        primary_metric="gap_to_known_optimum",
        objective_direction="minimize",
        metrics=[
            "gap_to_known_optimum",
            "mip_gap",
            "objective",
            "runtime_seconds",
            "feasible",
            "solver_status",
        ],
        mutation_hints=[
            "instance difficulty / profit-weight correlation",
            "number of binary variables",
            "number of resource constraints",
            "solver time limit",
            "local-search restarts (ablation)",
            "seed count (replication)",
        ],
        human_checks=[
            "confirm the reference optimum was PROVED, not time-limited",
            "confirm every reported solution passed independent feasibility verification",
            "confirm the baseline heuristic is not artificially weakened",
            "check Big-M / penalty coefficients before trusting any reformulation",
        ],
    ),
    "scheduling": DomainToolSpec(
        profile="scheduling",
        template_key="scheduling",
        metrics=["objective", "total_tardiness", "makespan", "runtime_seconds"],
        mutation_hints=["n_jobs", "due-date tightness", "processing-time variability", "dispatching rule baseline"],
        human_checks=["verify precedence and machine assumptions", "compare dispatching rules fairly"],
    ),
    "inventory": DomainToolSpec(
        profile="inventory",
        template_key="inventory",
        metrics=["objective", "mean_cost", "service_level", "runtime_seconds"],
        mutation_hints=["demand variance", "shortage cost", "order quantity grid", "service-level target"],
        human_checks=["verify demand distribution", "check cost units and shortage assumptions"],
    ),
    "network_analysis": DomainToolSpec(
        profile="network_analysis",
        template_key="network_analysis",
        metrics=["objective", "path_cost", "feasible", "runtime_seconds"],
        mutation_hints=["node count", "edge density", "edge failure/disruption", "connectivity"],
        human_checks=["verify graph directedness", "check disconnected cases and reliability definition"],
    ),
    "stochastic_models": DomainToolSpec(
        profile="stochastic_models",
        template_key="stochastic_models",
        metrics=["objective", "mean_wait", "utilization", "runtime_seconds"],
        mutation_hints=["arrival rate", "service rate", "simulation horizon", "replications"],
        human_checks=["report confidence intervals", "check warm-up and simulation length"],
    ),
}


def get_domain_tool_spec(profile: str) -> DomainToolSpec:
    return DOMAIN_TOOL_SPECS.get(profile, DOMAIN_TOOL_SPECS["optimization"])

