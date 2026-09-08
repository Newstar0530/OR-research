from __future__ import annotations

from src.llm_client import LLMClient
from src.schemas import ModelDraft, ResearchIdea
from src.utils.prompt_utils import human_verification_footer


class ModelingAgent:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def run(self, idea: ResearchIdea, research_goal: str, template_text: str = "") -> ModelDraft:
        markdown = f"""# Mathematical Model Draft

## Problem Definition
This is a generic OR/IE mathematical formulation scaffold for the research goal:

{research_goal}

The selected idea proposes:

- Model type: {idea.proposed_model_type}
- Algorithm type: {idea.proposed_algorithm_type}
- Hypothesis: {idea.core_hypothesis}

## Sets and Indices
- I: set of decision entities, indexed by i
- T: optional set of time periods, indexed by t
- S: optional set of scenarios, indexed by s
- A: optional set of network arcs or relationships, indexed by a

Needs human verification: remove unused sets and replace generic sets with domain-specific notation.

## Parameters
- c_i: cost, weight, processing time, or penalty associated with entity i
- d_t or d_s: demand, workload, disruption, or scenario parameter
- u_i: capacity, availability, or upper bound
- p_s: probability or weight of scenario s, if stochastic analysis is used
- theta: tunable policy, heuristic, or robustness parameter

Needs human verification: units, signs, and parameter meanings must be aligned with the actual research problem.

## Decision Variables
- x_i: primary decision variable for entity i
- y_t: optional state/control variable for period t
- z_s: optional scenario response variable

Variable domains may be binary, integer, continuous, or mixed depending on the model family.

## Objective Function
Generic form:

```text
minimize or maximize F(x, y, z; c, d, theta)
```

Typical OR objectives include cost, tardiness, service level, reliability, robustness, path length, waiting time, or weighted multi-criteria utility.

Needs human verification: objective direction and scale must match the research hypothesis.

## Constraints
Generic constraint families:

```text
resource/capacity constraints
flow, balance, or conservation constraints
assignment, sequencing, or precedence constraints
service-level, reliability, or risk constraints
variable-domain and boundary constraints
```

Needs human verification: add all feasibility, boundary, and coupling constraints required by the actual system.

## Assumptions
- The first executable experiment may use synthetic data when real benchmarks are unavailable.
- Baselines must be selected before interpreting proposed-method performance.
- All generated notation is provisional.

## Possible Extensions
- Robust or stochastic variants.
- Multi-objective formulation.
- Decomposition, simulation-optimization, or metaheuristic search.
- Statistical comparison over repeated seeds.

## Limitations
- This draft is not a proof of correctness.
- This draft does not establish novelty.
- Any claim about real-world validity requires domain data and human review.
"""
        return ModelDraft(markdown=markdown + human_verification_footer())

