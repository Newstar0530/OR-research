from __future__ import annotations

from src.core.mutation import MutationSpec
from src.core.research_journal import ResearchJournal
from src.llm_client import LLMClient


class MutationAgent:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def run(self, journal: ResearchJournal, iteration: int, branch_index: int, plan: str, domain_profile: str = "optimization") -> MutationSpec:
        plan_lower = plan.lower()
        domain_specific = self._domain_mutation(domain_profile, plan_lower)
        if domain_specific:
            return domain_specific
        if "adversarial" in plan_lower or "stress" in plan_lower:
            return MutationSpec(
                kind="stress_test",
                description="Increase instance sizes and noise to test whether the result survives harder scenarios.",
                replacements={
                    "PROBLEM_SIZES = [10, 25, 50]": "PROBLEM_SIZES = [25, 50, 100]",
                    "NOISE_SCALE = 1.0": "NOISE_SCALE = 2.0",
                    "REPLICATES = 10": "REPLICATES = 8",
                },
                metadata={"stress": True},
            )
        if "ablation" in plan_lower or "weak" in plan_lower:
            return MutationSpec(
                kind="component_ablation",
                description="Weaken the proposed component to estimate its contribution.",
                replacements={
                    "PROPOSED_EFFECT_LOW = 0.5": "PROPOSED_EFFECT_LOW = 0.0",
                    "PROPOSED_EFFECT_HIGH = 3.0": "PROPOSED_EFFECT_HIGH = 0.4",
                },
                metadata={"ablation": True},
            )
        if "robustness" in plan_lower or "repeat" in plan_lower:
            return MutationSpec(
                kind="replication_check",
                description="Increase repetitions with moderate effect settings to check stability.",
                replacements={
                    "REPLICATES = 10": "REPLICATES = 20",
                    "PROPOSED_EFFECT_LOW = 0.5": "PROPOSED_EFFECT_LOW = 0.6",
                    "PROPOSED_EFFECT_HIGH = 3.0": "PROPOSED_EFFECT_HIGH = 3.2",
                },
                metadata={"replication": True},
            )
        if "baseline" in plan_lower and "strength" in plan_lower:
            return MutationSpec(
                kind="baseline_strengthening",
                description="Strengthen the baseline by giving it a small improvement allowance.",
                replacements={
                    "BASELINE_EFFECT = 0.0": "BASELINE_EFFECT = 0.3",
                },
                metadata={"stronger_baseline": True},
            )
        return MutationSpec(
            kind="proposed_intensification",
            description="Increase proposed-method effect to explore the promising branch.",
            replacements={
                "PROPOSED_EFFECT_LOW = 0.5": "PROPOSED_EFFECT_LOW = 1.25",
                "PROPOSED_EFFECT_HIGH = 3.0": "PROPOSED_EFFECT_HIGH = 4.75",
            },
            metadata={"intensification": True},
        )

    def _domain_mutation(self, domain_profile: str, plan_lower: str) -> MutationSpec | None:
        if domain_profile == "binary_program":
            return self._binary_program_mutation(plan_lower)
        if domain_profile == "scheduling":
            if "stress" in plan_lower or "adversarial" in plan_lower:
                return MutationSpec(
                    kind="stress_test",
                    description="Increase scheduling problem size and processing-time variability.",
                    replacements={
                        "N_JOBS_LIST = [10, 20, 40]": "N_JOBS_LIST = [20, 40, 80]",
                        "PROCESSING_TIME_MAX = 20": "PROCESSING_TIME_MAX = 40",
                        "REPLICATES = 5": "REPLICATES = 4",
                    },
                    metadata={"domain_profile": domain_profile},
                )
            if "ablation" in plan_lower or "weak" in plan_lower:
                return MutationSpec(
                    kind="component_ablation",
                    description="Tighten due dates to test whether dispatching conclusions survive harsher tardiness pressure.",
                    replacements={"DUE_DATE_TIGHTNESS = [0.7, 1.0, 1.3]": "DUE_DATE_TIGHTNESS = [0.45, 0.7, 0.95]"},
                    metadata={"domain_profile": domain_profile},
                )
        if domain_profile == "inventory":
            if "stress" in plan_lower or "adversarial" in plan_lower:
                return MutationSpec(
                    kind="stress_test",
                    description="Increase demand uncertainty and shortage penalties.",
                    replacements={
                        "DEMAND_SIGMAS = [5, 15, 30]": "DEMAND_SIGMAS = [15, 30, 60]",
                        "SHORTAGE_COSTS = [5, 20, 50]": "SHORTAGE_COSTS = [20, 50, 100]",
                    },
                    metadata={"domain_profile": domain_profile},
                )
            if "replication" in plan_lower or "repeat" in plan_lower:
                return MutationSpec(
                    kind="replication_check",
                    description="Increase inventory demand sample size.",
                    replacements={"DEMAND_SAMPLE_SIZE = 5000": "DEMAND_SAMPLE_SIZE = 10000"},
                    metadata={"domain_profile": domain_profile},
                )
        if domain_profile == "network_analysis":
            if "stress" in plan_lower or "adversarial" in plan_lower:
                return MutationSpec(
                    kind="stress_test",
                    description="Increase network size while lowering density to create harder connectivity cases.",
                    replacements={
                        "N_NODES_LIST = [20, 40, 80]": "N_NODES_LIST = [40, 80, 120]",
                        "EDGE_DENSITIES = [0.08, 0.14, 0.22]": "EDGE_DENSITIES = [0.04, 0.08, 0.14]",
                    },
                    metadata={"domain_profile": domain_profile},
                )
            if "replication" in plan_lower or "repeat" in plan_lower:
                return MutationSpec(
                    kind="replication_check",
                    description="Increase random graph replications.",
                    replacements={"REPLICATES = 5": "REPLICATES = 10"},
                    metadata={"domain_profile": domain_profile},
                )
        if domain_profile == "stochastic_models":
            if "stress" in plan_lower or "adversarial" in plan_lower:
                return MutationSpec(
                    kind="stress_test",
                    description="Push queueing utilization closer to instability.",
                    replacements={"ARRIVAL_RATES = [0.6, 0.8, 0.95]": "ARRIVAL_RATES = [0.85, 0.95, 0.99]"},
                    metadata={"domain_profile": domain_profile},
                )
            if "replication" in plan_lower or "repeat" in plan_lower:
                return MutationSpec(
                    kind="replication_check",
                    description="Increase simulation horizon and replications.",
                    replacements={
                        "SIMULATION_HORIZON = 500": "SIMULATION_HORIZON = 1000",
                        "REPLICATES = 20": "REPLICATES = 40",
                    },
                    metadata={"domain_profile": domain_profile},
                )
        return None

    @staticmethod
    def _binary_program_mutation(plan_lower: str) -> MutationSpec:
        """Mutations for the solver-backed MI 0-1 experiment.

        Every replacement targets a knob that changes the *science* -- instance
        hardness, replication, the proposed method's search budget -- rather
        than a cosmetic constant. A binary-program node always receives one of
        these, because falling through to the generic string replacements would
        silently produce a node identical to its parent.
        """

        if "stress" in plan_lower or "adversarial" in plan_lower:
            return MutationSpec(
                kind="stress_test",
                description=(
                    "Harder instances: strongly correlated profits (which defeat ratio-greedy), "
                    "more variables, and more resource constraints."
                ),
                replacements={
                    'DIFFICULTY = "weakly_correlated"': 'DIFFICULTY = "strongly_correlated"',
                    "PROBLEM_SIZES = [14, 18, 22]": "PROBLEM_SIZES = [18, 24, 30]",
                    "N_RESOURCES = 3": "N_RESOURCES = 5",
                    "GROUND_TRUTH_TIME_LIMIT_SECONDS = 20.0": "GROUND_TRUTH_TIME_LIMIT_SECONDS = 45.0",
                },
                metadata={"domain_profile": "binary_program"},
            )
        if "ablation" in plan_lower or "weak" in plan_lower:
            return MutationSpec(
                kind="component_ablation",
                description=(
                    "Ablate the restart component of the proposed local search to measure how much "
                    "of its advantage comes from restarts rather than from hill climbing."
                ),
                replacements={"LOCAL_SEARCH_RESTARTS = 3": "LOCAL_SEARCH_RESTARTS = 0"},
                metadata={"domain_profile": "binary_program", "ablated_component": "restarts"},
            )
        if "replication" in plan_lower or "repeat" in plan_lower or "robust" in plan_lower or "seed" in plan_lower:
            return MutationSpec(
                kind="replication_check",
                description="Double the number of instance seeds to tighten the confidence intervals.",
                replacements={
                    "SEED_LIST = [0, 1, 2, 3, 4]": "SEED_LIST = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]"
                },
                metadata={"domain_profile": "binary_program"},
            )
        if "baseline" in plan_lower:
            return MutationSpec(
                kind="baseline_strengthening",
                description=(
                    "Make the comparison harder to win: cut the proposed method's time budget and "
                    "restarts so an advantage cannot come from simply spending more compute."
                ),
                replacements={
                    "SOLVER_TIME_LIMIT_SECONDS = 5.0": "SOLVER_TIME_LIMIT_SECONDS = 1.0",
                    "LOCAL_SEARCH_RESTARTS = 3": "LOCAL_SEARCH_RESTARTS = 1",
                },
                metadata={"domain_profile": "binary_program"},
            )
        return MutationSpec(
            kind="proposed_intensification",
            description="Give the proposed local search more restarts and a longer budget.",
            replacements={
                "LOCAL_SEARCH_RESTARTS = 3": "LOCAL_SEARCH_RESTARTS = 12",
                "SOLVER_TIME_LIMIT_SECONDS = 5.0": "SOLVER_TIME_LIMIT_SECONDS = 10.0",
            },
            metadata={"domain_profile": "binary_program"},
        )
