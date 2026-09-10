"""Formulate the problem, or hand over a template that admits what it is.

This was the last stage in the front half with no path to an LLM at all. It
interpolated the research goal into a fixed scaffold -- `F(x, y, z; c, d,
theta)`, sets `I / T / S / A`, a *list of constraint families* where the
constraints should be -- and returned it as `model_draft.md`. Every downstream
stage then treated a template as a formulation. Configuring a real model changed
nothing, because nothing here ever called one.

Now it asks, and then it checks. A draft is accepted only if it would survive
the critique that `CriticAgent` is about to write about it: scaffold symbols
gone, every section present, real mathematics somewhere, and an objective
direction chosen. A draft that fails gets the failures back as its next
instruction rather than being written to disk, because a model that produces
prose about modelling is the failure mode this stage already had.

When no model is available the scaffold is still returned -- it is a useful
form to fill in -- but the provenance says `not_generated` and the critique
says which of its elements are untouched. The template is offered as a
template, not as a draft of this problem.
"""

from __future__ import annotations

from src.core.artifact_provenance import ContentSource
from src.core.model_draft_inspection import inspect_model_draft
from src.llm_client import LLMClient
from src.llm_errors import LLMCallError
from src.schemas import ModelDraft, ResearchIdea
from src.utils.prompt_utils import human_verification_footer


MODEL_SYSTEM_PROMPT = (
    "You are an Operations Research modeller. You write formulations that can be implemented "
    "and falsified: explicit sets, explicit parameters with units, explicit variable domains, "
    "a chosen objective direction, and constraints written as expressions rather than named as "
    "families. You never write about modelling in place of modelling. If the problem statement "
    "is too vague to formulate, you say exactly what is missing instead of producing a template."
)

#: How many times a rejected draft may be sent back with its failures.
MAX_MODEL_ATTEMPTS = 2


class ModelingAgent:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def run(
        self,
        idea: ResearchIdea,
        research_goal: str,
        template_text: str = "",
        domain_profile: dict | None = None,
        literature_brief: str = "",
    ) -> tuple[ModelDraft, ContentSource, list[str]]:
        """Returns (draft, content_source, inputs_used)."""

        if not self.llm.use_mock:
            drafted, notes = self._llm_draft(
                idea, research_goal, domain_profile or {}, literature_brief, template_text
            )
            if drafted is not None:
                inputs = ["research_goal", "selected_idea"]
                if domain_profile:
                    inputs.append("domain_profile")
                if literature_brief:
                    inputs.append("literature_index")
                if template_text:
                    inputs.append("template_path")
                return ModelDraft(markdown=drafted + human_verification_footer()), "llm", inputs
            # The attempts are recorded in the scaffold so a reader can see the
            # model was asked and what it failed to produce.
            return (
                ModelDraft(markdown=self._scaffold(idea, research_goal, notes) + human_verification_footer()),
                "not_generated",
                ["research_goal", "selected_idea"],
            )
        return (
            ModelDraft(markdown=self._scaffold(idea, research_goal) + human_verification_footer()),
            "not_generated",
            ["research_goal", "selected_idea"],
        )

    def _llm_draft(
        self,
        idea: ResearchIdea,
        research_goal: str,
        domain_profile: dict,
        literature_brief: str,
        template_text: str,
    ) -> tuple[str | None, list[str]]:
        """Ask for a formulation, and refuse anything that is still a template."""

        notes: list[str] = []
        feedback = ""
        for attempt in range(MAX_MODEL_ATTEMPTS):
            prompt = self._prompt(
                idea, research_goal, domain_profile, literature_brief, template_text, feedback
            )
            try:
                draft = self.llm.chat(MODEL_SYSTEM_PROMPT, prompt)
            except Exception as exc:
                if isinstance(exc, LLMCallError) and exc.is_permanent:
                    raise
                reason = exc.summary() if isinstance(exc, LLMCallError) else str(exc)
                notes.append(f"Attempt {attempt + 1}: the language model call failed ({reason}).")
                return None, notes
            body = _strip_fences(draft)
            inspection = inspect_model_draft(body)
            if inspection.is_specific:
                return body, notes
            failures = inspection.failures()
            notes.append(f"Attempt {attempt + 1} was rejected: " + " ".join(failures))
            feedback = (
                "Your previous draft was rejected because it is not yet a formulation of this "
                "problem. Fix all of these:\n" + "\n".join(f"- {item}" for item in failures)
            )
        return None, notes

    @staticmethod
    def _prompt(
        idea: ResearchIdea,
        research_goal: str,
        domain_profile: dict,
        literature_brief: str,
        template_text: str,
        feedback: str = "",
    ) -> str:
        sections = [
            "Write a mathematical formulation for this research problem, as markdown.",
            "",
            f"Research goal: {research_goal}",
            f"Hypothesis: {idea.core_hypothesis}",
            f"Proposed model type: {idea.proposed_model_type}",
            f"Proposed algorithm type: {idea.proposed_algorithm_type}",
        ]
        typical = [str(item) for item in domain_profile.get("typical_models", []) or []]
        if typical:
            sections.append(f"Model families used in this domain: {', '.join(typical)}")
        if literature_brief:
            sections.extend(["", literature_brief[:4000]])
        if template_text:
            sections.extend(
                ["", "A starter template was supplied; follow its conventions where they apply:", template_text[:2000]]
            )
        sections.extend(
            [
                "",
                "Required sections, with these exact headings:",
                "## Problem Definition",
                "## Sets and Indices",
                "## Parameters",
                "## Decision Variables",
                "## Objective Function",
                "## Constraints",
                "## Assumptions",
                "## Limitations",
                "",
                "Rules:",
                "- Name this problem's own sets, parameters and variables. Do not use placeholder "
                "symbols such as `F(x, y, z)` or `I: set of decision entities`.",
                "- State every variable's domain (binary, integer, continuous) and every "
                "parameter's units.",
                "- Begin the objective with `minimize` or `maximize`. Not both.",
                "- Write the constraints as expressions, with summations and inequalities. Naming "
                "a family such as `capacity constraints` is not a constraint.",
                "- State what result would falsify the hypothesis.",
            ]
        )
        if feedback:
            sections.extend(["", feedback])
        return "\n".join(sections)

    @staticmethod
    def _scaffold(idea: ResearchIdea, research_goal: str, notes: list[str] | None = None) -> str:
        attempt_block = ""
        if notes:
            attempt_block = (
                "\n> **NOT GENERATED: this problem's formulation**\n>\n"
                "> A language model was asked to formulate this problem and its draft was rejected "
                "for not being specific enough to implement. What follows is the generic scaffold.\n>\n"
                + "\n".join(f"> - {note}" for note in notes)
                + "\n"
            )
        else:
            attempt_block = (
                "\n> **NOT GENERATED: this problem's formulation**\n>\n"
                "> No language model was configured for this run, so nothing formulated this "
                "problem. What follows is a generic scaffold with the research goal interpolated "
                "into it -- a form to fill in, not a draft of your model. `model_critique.md` "
                "lists which of its elements are still untouched.\n"
            )
        markdown = f"""# Mathematical Model Draft
{attempt_block}

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
        return markdown



def _strip_fences(text: str) -> str:
    """Models wrap markdown in a fence about half the time."""

    body = (text or "").strip()
    if body.startswith("```"):
        lines = body.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        body = "\n".join(lines).strip()
    return body
