"""Write down what the formulation will be held to, before it is written.

Nothing in this pipeline previously stated what the model was *supposed* to
contain. `ResearchProblem.constraint_candidates` existed as a field and was
never populated, so the critique had nothing to compare a draft against and
said so: automated checking cannot tell whether a constraint set is complete.
That is true only while the requirements are unwritten. It is a statement about
the missing list, not about the limits of checking.

So this stage produces the list first. It runs before modelling, and it asks a
different question than the modelling stage does -- what must be true of a
solution, rather than how to express it -- so that the draft is later measured
against something that was not derived from the draft.

The list is not authoritative. An extractor can miss a requirement the goal
only implies, and can invent one the goal does not state. Every requirement
carries the text it came from so a reader can check it, and a requirement the
author rejects should be removed from the goal, not from the check.
"""

from __future__ import annotations

import re

from src.core.artifact_provenance import ContentSource
from src.core.requirement_coverage import Requirement, RequirementSet
from src.llm_client import LLMClient
from src.llm_errors import LLMCallError
from src.schemas import ResearchIdea


REQUIREMENT_SYSTEM_PROMPT = (
    "You read a research goal and list the conditions any acceptable solution must satisfy. "
    "You do not formulate the problem, choose notation, or propose a method. You list "
    "requirements: capacity limits, balance conditions, coverage or service levels, "
    "precedence, budget, timing, and the direction the objective must take. Each requirement "
    "is one checkable condition stated in the problem's own terms. If the goal does not state "
    "a condition, you do not invent one."
)

#: Words that mark a clause as stating a condition rather than a motivation.
_REQUIREMENT_CUES = (
    "must", "cannot", "can not", "shall", "required", "require", "requires",
    "at least", "at most", "no more than", "no fewer", "not exceed", "exceed",
    "more than", "fewer than", "less than", "may not", "must not",
    "subject to", "constrained", "constraint", "limit", "limited", "capacity",
    "budget", "deadline", "due date", "minimum", "maximum", "ensure", "guarantee",
    "within", "feasible", "availability", "available", "precedence", "balance",
)

_MINIMIZE_CUES = ("minimize", "minimise", "reduce", "lower", "decrease", "cost", "tardiness", "waiting", "makespan")
_MAXIMIZE_CUES = ("maximize", "maximise", "increase", "improve", "throughput", "profit", "utilization", "utilisation", "coverage")


class RequirementAgent:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def run(
        self,
        research_goal: str,
        idea: ResearchIdea | None = None,
        domain_profile: dict | None = None,
    ) -> tuple[RequirementSet, ContentSource]:
        if not self.llm.use_mock:
            extracted = self._llm_requirements(research_goal, idea, domain_profile or {})
            if extracted is not None:
                return extracted, "llm"
        derived = self._derived_requirements(research_goal)
        if derived.requirements:
            return derived, "derived"
        return (
            RequirementSet(
                extracted_by="not_generated",
                notes=[
                    "No requirement could be extracted from the research goal. The goal states no "
                    "condition in checkable form, so the formulation cannot be checked for "
                    "completeness against it."
                ],
            ),
            "not_generated",
        )

    def _llm_requirements(
        self, research_goal: str, idea: ResearchIdea | None, domain_profile: dict
    ) -> RequirementSet | None:
        prompt_parts = [f"Research goal: {research_goal}"]
        if idea is not None:
            prompt_parts.append(f"Selected hypothesis: {idea.core_hypothesis}")
            prompt_parts.append(f"Proposed model type: {idea.proposed_model_type}")
        typical = [str(item) for item in domain_profile.get("typical_models", []) or []]
        if typical:
            prompt_parts.append(f"Model families common in this domain: {', '.join(typical)}")
        prompt_parts.append(
            "\nReturn JSON: {\"requirements\": [{\"text\": \"<one checkable condition>\", "
            "\"source\": \"<the phrase in the goal that states it>\"}], "
            "\"objective_sense\": \"minimize\" | \"maximize\" | \"unspecified\"}. "
            "List only conditions the goal actually states. An empty list is a valid answer."
        )
        try:
            payload = self.llm.chat_json(
                REQUIREMENT_SYSTEM_PROMPT, "\n".join(prompt_parts), stage="requirement_extraction"
            )
        except Exception as exc:
            if isinstance(exc, LLMCallError) and exc.is_permanent:
                raise
            return None
        raw = payload.get("requirements") or []
        requirements = []
        for index, entry in enumerate(raw, start=1):
            text = (entry.get("text") if isinstance(entry, dict) else str(entry)) or ""
            if not text.strip():
                continue
            source = entry.get("source", "research_goal") if isinstance(entry, dict) else "research_goal"
            requirements.append(
                Requirement(id=f"R{index}", text=text.strip(), source=str(source)[:200])
            )
        if not requirements:
            return None
        sense = str(payload.get("objective_sense", "unspecified")).lower()
        if sense not in {"minimize", "maximize", "unspecified"}:
            sense = "unspecified"
        return RequirementSet(
            requirements=requirements,
            required_objective_sense=sense,  # type: ignore[arg-type]
            extracted_by="llm",
        )

    @staticmethod
    def _derived_requirements(research_goal: str) -> RequirementSet:
        """Pull condition-bearing clauses out of the goal without a model.

        Deliberately shallow: it selects sentences that state a condition and
        keeps them verbatim. It cannot paraphrase, disambiguate, or notice a
        requirement stated as background. Its value is that the coverage check
        runs at all when no model is configured, with provenance that says how
        the list was built.
        """

        goal = (research_goal or "").strip()
        if not goal:
            return RequirementSet(extracted_by="not_generated")
        clauses = [item.strip() for item in re.split(r"(?<=[.;!?])\s+|\n+", goal) if item.strip()]
        requirements = []
        for clause in clauses:
            lowered = clause.lower()
            if not any(cue in lowered for cue in _REQUIREMENT_CUES):
                continue
            requirements.append(
                Requirement(
                    id=f"R{len(requirements) + 1}",
                    text=clause[:400],
                    source="research_goal (clause selected by cue word)",
                )
            )
        notes = []
        if requirements:
            notes.append(
                "These clauses were selected from the research goal by cue words such as `must`, "
                "`capacity` or `at least`. They are the goal's own wording, not an interpretation "
                "of it, and a condition the goal only implies was not selected."
            )
        return RequirementSet(
            requirements=requirements,
            required_objective_sense=_derived_sense(goal),
            extracted_by="derived" if requirements else "not_generated",
            notes=notes,
        )


def _derived_sense(goal: str) -> str:
    """The direction the goal implies, when it implies one unambiguously."""

    lowered = goal.lower()
    wants_min = any(cue in lowered for cue in _MINIMIZE_CUES)
    wants_max = any(cue in lowered for cue in _MAXIMIZE_CUES)
    if wants_min and not wants_max:
        return "minimize"
    if wants_max and not wants_min:
        return "maximize"
    return "unspecified"
