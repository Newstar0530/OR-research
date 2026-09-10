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
    "bilevel_transformation": DomainToolSpec(
        profile="bilevel_transformation",
        template_key="bilevel_transformation",
        primary_metric="gap_to_known_optimum",
        objective_direction="minimize",
        metrics=[
            "equivalence_verdict",
            "gap_to_known_optimum",
            "big_m_slack",
            "big_m_dual",
            "n_binaries",
            "solution_bilevel_feasible",
        ],
        mutation_hints=[
            "instance difficulty (dual magnitude, degeneracy, coupling)",
            "problem size and number of complementarity pairs",
            "Big-M scale relative to the derived bounds",
            "flat Big-M values as used in the literature",
            "seed count (replication)",
        ],
        human_checks=[
            "confirm the oracle proved the optimum rather than timing out",
            "confirm every reported point was checked for follower optimality, not just feasibility",
            "a `model_better_than_truth` verdict is a modelling error, never a tuning problem",
            "equivalence on these instances is not a proof of equivalence in general",
        ],
    ),
    "qubo_transformation": DomainToolSpec(
        profile="qubo_transformation",
        template_key="qubo_transformation",
        primary_metric="gap_to_known_optimum",
        objective_direction="minimize",
        metrics=[
            "qubo_verdict",
            "gap_to_known_optimum",
            "penalty",
            "penalty_to_objective_ratio",
            "n_bits",
            "n_slack_bits",
            "ground_state_proved",
            "discretisation_loss",
        ],
        mutation_hints=[
            "penalty scale relative to the derived bound",
            "discretisation precision",
            "number of inequality constraints (drives the slack bit count)",
            "problem size",
            "the bit budget below which a ground state can still be proved",
        ],
        human_checks=[
            "`not_proved` means a sampler's best draw, never a ground state",
            "a feasible decoded point with the wrong energy is sampler non-convergence, "
            "not an encoding defect, unless the ground state was proved",
            "check the coefficient dynamic range before assuming a correct QUBO is programmable",
            "discretisation loss is a measured cost of the grid, not a bug",
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

