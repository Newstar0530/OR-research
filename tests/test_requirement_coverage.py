"""A formulation is checked against what it was supposed to encode.

Every verification already in this project compares one model to another: a
QUBO to the MI 0-1 problem behind it, a KKT reformulation to the bilevel
problem it replaced, a returned vector to the constraints it must satisfy. All
of it is blind to the same thing -- a chain that began from the wrong problem.
Dropping a requirement enlarges the feasible region, so the solver returns a
*better* objective and every downstream check passes.

The tests that matter here are therefore the ones where the draft is fluent,
complete, internally consistent, and wrong.
"""

from src.agents.critic_agent import CriticAgent
from src.agents.modeling_agent import ModelingAgent
from src.agents.requirement_agent import RequirementAgent
from src.core.requirement_coverage import (
    Requirement,
    RequirementChecker,
    RequirementSet,
    observed_objective_sense,
)
from src.llm_client import LLMClient
from src.schemas import ResearchIdea


COMPLETE_DRAFT = """# Mathematical Model Draft

## Problem Definition
Assign nurses to shifts.

## Sets and Indices
- N: nurses, indexed by n
- S: shifts, indexed by s

## Parameters
- c_ns: cost of assigning nurse n to shift s, in currency units
- r_s: required headcount on shift s, in nurses

## Decision Variables
- x_ns in {0, 1}: 1 if nurse n works shift s

## Objective Function
minimize sum_n sum_s c_ns * x_ns

## Constraints
[R1] sum_n x_ns >= r_s for all s in S
[R2] sum_s x_ns <= 5 for all n in N

## Assumptions
- Every nurse is qualified for every shift.

## Limitations
- Rest periods between consecutive shifts are not modelled.
"""

#: The same model with the weekly maximum removed. Still specific, still
#: solvable, still passes every structural check -- and its optimum is cheaper
#: than the real problem's, because a dropped constraint only ever enlarges the
#: feasible region.
DRAFT_MISSING_A_REQUIREMENT = COMPLETE_DRAFT.replace(
    "[R2] sum_s x_ns <= 5 for all n in N\n", ""
)

REQUIREMENTS = RequirementSet(
    requirements=[
        Requirement(id="R1", text="each shift must be covered by its required headcount"),
        Requirement(id="R2", text="no nurse works more than five shifts per week"),
    ],
    required_objective_sense="minimize",
    extracted_by="derived",
)


def _checker() -> RequirementChecker:
    """No model configured, so coverage is decided lexically."""

    return RequirementChecker(LLMClient(use_mock=True))


# -- the check itself ------------------------------------------------------


def test_a_complete_draft_covers_every_requirement() -> None:
    verdict = _checker().check(COMPLETE_DRAFT, REQUIREMENTS)

    assert verdict.checked
    assert verdict.is_complete
    assert verdict.missing == []
    assert verdict.missing_constraint_rate == 0.0
    assert verdict.coverage_ratio == 1.0


def test_a_dropped_constraint_is_named_not_merely_counted() -> None:
    """The point of the check is the sentence a reader can act on."""

    verdict = _checker().check(DRAFT_MISSING_A_REQUIREMENT, REQUIREMENTS)

    assert verdict.checked
    assert not verdict.is_complete
    assert verdict.blocks_acceptance()
    assert [item.requirement_id for item in verdict.missing] == ["R2"]
    assert "five shifts" in verdict.missing[0].requirement_text
    assert verdict.missing_constraint_rate == 0.5


def test_the_dropped_constraint_survives_every_other_check() -> None:
    """Why this module exists: nothing else in the pipeline notices."""

    from src.core.model_draft_inspection import inspect_model_draft

    assert inspect_model_draft(DRAFT_MISSING_A_REQUIREMENT).is_specific
    assert inspect_model_draft(DRAFT_MISSING_A_REQUIREMENT).failures() == []


def test_a_reversed_objective_is_caught_even_when_every_constraint_is_present() -> None:
    reversed_draft = COMPLETE_DRAFT.replace(
        "minimize sum_n sum_s c_ns * x_ns", "maximize sum_n sum_s c_ns * x_ns"
    )
    verdict = _checker().check(reversed_draft, REQUIREMENTS)

    assert verdict.missing == [], "the constraints are all still there"
    assert not verdict.objective_sense_matches
    assert verdict.blocks_acceptance()
    assert any("direction" in reason for reason in verdict.failures())


def test_prose_in_the_problem_definition_does_not_count_as_a_constraint() -> None:
    """A requirement described is not a requirement imposed."""

    described = DRAFT_MISSING_A_REQUIREMENT.replace(
        "Assign nurses to shifts.",
        "Assign nurses to shifts. No nurse works more than five shifts per week.",
    )
    verdict = _checker().check(described, REQUIREMENTS)

    assert [item.requirement_id for item in verdict.missing] == ["R2"]
    assert verdict.searched_section == "## Constraints"


def test_an_untagged_draft_falls_back_and_says_the_fallback_is_weak() -> None:
    """Algebra and prose share no words, so the fallback under-reports coverage.

    Recorded rather than hidden: a reader who sees every requirement marked
    missing needs to know it is the method talking, not the model.
    """

    untagged = COMPLETE_DRAFT.replace("[R1] ", "").replace("[R2] ", "")
    verdict = _checker().check(untagged, REQUIREMENTS)

    assert verdict.checked
    assert all(item.method == "lexical" for item in verdict.matches)
    assert any("prompt to look, not as a verdict" in note for note in verdict.notes)


def test_a_tag_is_a_claim_about_a_constraint_not_a_proof_about_one() -> None:
    """The known hole in declared coverage, kept visible.

    Tagging a constraint `[R2]` asserts that it imposes R2; nothing here
    checks the algebra actually does. This narrows the failure from "a
    requirement silently vanished" to "a specific line claims something a
    reader can check", which is the whole gain -- not a correctness proof.
    """

    mislabelled = COMPLETE_DRAFT.replace(
        "[R2] sum_s x_ns <= 5 for all n in N", "[R2] x_ns >= 0 for all n in N, s in S"
    )
    verdict = _checker().check(mislabelled, REQUIREMENTS)

    assert verdict.is_complete, "the claim is accepted at face value"
    assert verdict.matches[1].evidence.startswith("[R2] x_ns >= 0"), "but it is recorded verbatim"


def test_an_empty_requirement_list_is_not_a_pass() -> None:
    verdict = _checker().check(COMPLETE_DRAFT, RequirementSet())

    assert not verdict.checked
    assert not verdict.is_complete
    assert not verdict.blocks_acceptance(), "an unrun check must not stop the run"


def test_the_declared_direction_is_read_from_the_line_not_the_document() -> None:
    both_words = "The tradeoff is whether to maximize service.\n\nminimize total cost\n"
    assert observed_objective_sense(both_words) == "minimize"
    assert observed_objective_sense("no direction here") == "unknown"


# -- extraction ------------------------------------------------------------


def test_requirements_are_pulled_from_the_goal_without_a_model() -> None:
    goal = (
        "Schedule nurses across shifts. Each shift must be covered by its required headcount. "
        "No nurse may work more than five shifts per week. We hope to reduce total cost."
    )
    extracted, source = RequirementAgent(LLMClient(use_mock=True)).run(goal)

    assert source == "derived"
    assert len(extracted.requirements) == 2
    assert any("headcount" in item.text for item in extracted.requirements)
    assert extracted.required_objective_sense == "minimize"


def test_a_goal_stating_no_condition_yields_no_requirements_and_says_so() -> None:
    extracted, source = RequirementAgent(LLMClient(use_mock=True)).run(
        "Investigate scheduling heuristics."
    )

    assert source == "not_generated"
    assert extracted.requirements == []
    assert any("cannot be checked" in note for note in extracted.notes)


# -- the critique reports it -----------------------------------------------


def test_the_critique_names_the_missing_requirement_instead_of_declining_to_judge() -> None:
    verdict = _checker().check(DRAFT_MISSING_A_REQUIREMENT, REQUIREMENTS)
    critique, _ = CriticAgent(LLMClient(use_mock=True)).run(DRAFT_MISSING_A_REQUIREMENT, verdict)

    assert any("five shifts" in item for item in critique.missing_constraints)
    assert any("not imposed by any constraint" in issue for issue in critique.critical_issues)
    assert not any("Automated checking cannot tell" in item for item in critique.missing_constraints)


def test_without_a_requirement_list_the_critique_says_unchecked_not_complete() -> None:
    critique, _ = CriticAgent(LLMClient(use_mock=True)).run(COMPLETE_DRAFT)

    assert any("unchecked, not confirmed" in item for item in critique.missing_constraints)


# -- the modelling stage enforces it ---------------------------------------


class ScriptedLLM(LLMClient):
    """A non-mock client returning canned drafts, recording every prompt."""

    def __init__(self, *responses: str) -> None:
        super().__init__(use_mock=False)
        object.__setattr__(self, "responses", list(responses))
        object.__setattr__(self, "prompts", [])

    def chat(self, system_prompt: str, user_prompt: str) -> str:
        self.prompts.append(user_prompt)
        return self.responses[min(len(self.prompts) - 1, len(self.responses) - 1)]


def _idea() -> ResearchIdea:
    return ResearchIdea(
        title="Nurse rostering",
        problem_context="shift assignment",
        core_hypothesis="a matheuristic closes the gap",
        expected_contribution="a reproducible comparison",
        proposed_model_type="set covering MILP",
        proposed_algorithm_type="matheuristic",
        experimental_plan="compare against the exact solver",
        interestingness_score=6, novelty_score=4, feasibility_score=8, risk_score=3,
        assumptions=[], required_data=[], expected_outputs=[],
    )


def test_a_draft_missing_a_requirement_is_rejected_and_told_which_one() -> None:
    """The draft is specific, so only the coverage check can refuse it."""

    llm = ScriptedLLM(DRAFT_MISSING_A_REQUIREMENT, COMPLETE_DRAFT)
    draft, source, inputs, coverage = ModelingAgent(llm).run(
        _idea(), "goal", requirements=REQUIREMENTS
    )

    assert len(llm.prompts) == 2, "the first draft should have been sent back"
    assert "five shifts" in llm.prompts[1]
    assert "not imposed by any constraint" in llm.prompts[1]
    assert source == "llm"
    assert coverage.is_complete
    assert "requirement_set" in inputs


def test_the_requirements_are_in_the_first_prompt_not_only_the_retry() -> None:
    llm = ScriptedLLM(COMPLETE_DRAFT)
    ModelingAgent(llm).run(_idea(), "goal", requirements=REQUIREMENTS)

    assert "no nurse works more than five shifts per week" in llm.prompts[0]
    assert "must be a minimize problem" in llm.prompts[0]


def test_without_requirements_the_modelling_stage_is_unchanged() -> None:
    llm = ScriptedLLM(DRAFT_MISSING_A_REQUIREMENT)
    draft, source, _, coverage = ModelingAgent(llm).run(_idea(), "goal")

    assert source == "llm", "nothing to check against, so nothing to reject for"
    assert len(llm.prompts) == 1
    assert not coverage.checked
