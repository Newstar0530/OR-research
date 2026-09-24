"""The modeling stage asks for a formulation, then refuses anything that isn't one.

This was the last front-half stage with no path to a language model at all: it
interpolated the research goal into a fixed scaffold and returned it as
`model_draft.md`, so configuring a real model changed nothing.

The interesting tests here are the rejections. A model asked to formulate a
problem will often write *about* modelling instead -- naming constraint
families, describing an objective without writing one -- and that output is
indistinguishable from the old scaffold in every way that matters. It has to be
refused, with the reasons handed back, rather than written to disk.
"""

from pathlib import Path

import pytest

from src.agents.critic_agent import CriticAgent
from src.agents.modeling_agent import ModelingAgent
from src.core.model_draft_inspection import inspect_model_draft
from src.llm_client import LLMClient
from src.schemas import ResearchIdea


GOOD_DRAFT = """# Mathematical Model Draft

## Problem Definition
Single-machine weighted tardiness.

## Sets and Indices
- J: jobs, indexed by j and k

## Parameters
- p_j: processing time of job j, in minutes
- d_j: due date of job j, in minutes from release
- w_j: tardiness weight of job j, dimensionless

## Decision Variables
- s_j >= 0: start time of job j, continuous
- y_jk in {0, 1}: 1 if job j precedes job k

## Objective Function
minimize sum_j w_j * T_j

## Constraints
T_j >= s_j + p_j - d_j for all j in J
T_j >= 0 for all j in J
s_k >= s_j + p_j - M * (1 - y_jk) for all j, k in J
y_jk + y_kj = 1 for all j < k

## Assumptions
- One machine, no preemption, all jobs released at time zero.

## Limitations
- Big-M must be derived, not guessed.
"""

#: Fluent, well-formatted, and not a formulation: families instead of
#: constraints, no expression anywhere, no chosen direction.
PROSE_ABOUT_MODELLING = """# Mathematical Model Draft

## Problem Definition
We consider a scheduling problem.

## Sets and Indices
- The set of jobs and the set of machines.

## Parameters
- Processing times and due dates.

## Decision Variables
- Start times and sequencing decisions.

## Objective Function
The objective is to optimise weighted tardiness performance.

## Constraints
resource/capacity constraints, precedence constraints, and variable-domain constraints.

## Assumptions
- Deterministic data.

## Limitations
- Needs validation.
"""


def _idea() -> ResearchIdea:
    return ResearchIdea(
        title="Weighted tardiness on one machine",
        problem_context="single machine scheduling",
        core_hypothesis="a dispatching rule closes most of the gap",
        expected_contribution="a reproducible comparison",
        proposed_model_type="time-indexed MILP",
        proposed_algorithm_type="dispatching heuristic",
        experimental_plan="compare against the exact solver",
        interestingness_score=6, novelty_score=4, feasibility_score=8, risk_score=3,
        assumptions=[], required_data=[], expected_outputs=[],
    )


class ScriptedLLM(LLMClient):
    """A non-mock client returning canned drafts, recording every prompt."""

    def __init__(self, *responses: str) -> None:
        super().__init__(use_mock=False)
        object.__setattr__(self, "responses", list(responses))
        object.__setattr__(self, "prompts", [])

    def chat(self, system_prompt: str, user_prompt: str) -> str:
        self.prompts.append(user_prompt)
        return self.responses[min(len(self.prompts) - 1, len(self.responses) - 1)]


class BrokenLLM(LLMClient):
    def __init__(self) -> None:
        super().__init__(use_mock=False)

    def chat(self, system_prompt: str, user_prompt: str) -> str:
        raise RuntimeError("provider is down")


# -- the inspection both stages share --------------------------------------


def test_a_real_formulation_passes_the_inspection() -> None:
    inspection = inspect_model_draft(GOOD_DRAFT)
    assert inspection.is_specific
    assert inspection.failures() == []


def test_prose_about_modelling_fails_it() -> None:
    """It has every heading and reads well. That is exactly the danger."""

    inspection = inspect_model_draft(PROSE_ABOUT_MODELLING)
    assert not inspection.is_specific
    assert not inspection.has_math
    assert not inspection.declares_direction
    assert any("constraint" in reason.lower() for reason in inspection.failures())


@pytest.mark.parametrize(
    "mutation,expected",
    [
        (lambda d: d.replace("## Constraints", "## Restrictions"), "missing sections"),
        (lambda d: d.replace("minimize sum_j w_j * T_j", "minimize or maximize weighted tardiness"), "direction"),
        (
            # Every mathematical marker removed, not just the obvious ones:
            # `for all` and `in {0, 1}` are quantifiers and count as maths too.
            lambda d: d.replace(">=", "is at least")
            .replace("sum_j", "the sum over j")
            .replace("for all", "for every")
            .replace("in {0, 1}", "is binary"),
            "expressions",
        ),
    ],
)
def test_each_requirement_is_enforced_separately(mutation, expected: str) -> None:
    inspection = inspect_model_draft(mutation(GOOD_DRAFT))
    assert not inspection.is_specific
    assert any(expected.split()[0].lower() in reason.lower() for reason in inspection.failures())


def test_an_empty_draft_is_not_specific() -> None:
    assert inspect_model_draft("").failures() == ["The document is empty."]


def test_the_critic_and_the_modeling_stage_agree_on_what_a_scaffold_is() -> None:
    """Kept as one inspection so they cannot drift apart."""

    scaffold, _, _, _ = ModelingAgent(LLMClient(use_mock=True)).run(_idea(), "goal")
    critique, _ = CriticAgent(LLMClient(use_mock=True)).run(scaffold.markdown)

    assert not inspect_model_draft(scaffold.markdown).is_specific
    assert any("untouched scaffold" in issue for issue in critique.critical_issues)

    accepted_critique, _ = CriticAgent(LLMClient(use_mock=True)).run(GOOD_DRAFT)
    assert inspect_model_draft(GOOD_DRAFT).is_specific
    assert any("No scaffold placeholder was detected" in issue for issue in accepted_critique.critical_issues)


# -- the modeling stage ----------------------------------------------------


def test_a_specific_formulation_is_accepted_and_recorded_as_llm_written() -> None:
    llm = ScriptedLLM(f"```markdown\n{GOOD_DRAFT}```")
    draft, source, inputs, _ = ModelingAgent(llm).run(
        _idea(), "minimise weighted tardiness", literature_brief="Indexed literature: ..."
    )

    assert source == "llm"
    assert "minimize sum_j w_j * T_j" in draft.markdown
    assert "NOT GENERATED" not in draft.markdown
    assert "literature_index" in inputs and "research_goal" in inputs


def test_a_template_shaped_answer_is_rejected_and_the_reasons_fed_back() -> None:
    llm = ScriptedLLM(PROSE_ABOUT_MODELLING, GOOD_DRAFT)
    draft, source, _, _ = ModelingAgent(llm).run(_idea(), "goal")

    assert source == "llm", "the second attempt should have been accepted"
    assert len(llm.prompts) == 2
    assert "rejected" in llm.prompts[1]
    assert "summation" in llm.prompts[1] or "expressions" in llm.prompts[1]
    assert "minimize sum_j" in draft.markdown


def test_a_model_that_never_formulates_falls_back_and_says_what_it_tried() -> None:
    llm = ScriptedLLM(PROSE_ABOUT_MODELLING)
    draft, source, _, _ = ModelingAgent(llm).run(_idea(), "goal")

    assert source == "not_generated"
    assert len(llm.prompts) == 2, "both attempts should have been spent"
    assert "NOT GENERATED: this problem's formulation" in draft.markdown
    assert "its draft was rejected" in draft.markdown
    assert "Attempt 1 was rejected" in draft.markdown
    # The scaffold is still handed over -- it is a useful form to fill in.
    assert "## Sets and Indices" in draft.markdown


def test_a_provider_outage_does_not_take_the_run_down() -> None:
    draft, source, _, _ = ModelingAgent(BrokenLLM()).run(_idea(), "goal")
    assert source == "not_generated"
    assert "provider is down" in draft.markdown


def test_the_prompt_carries_the_study_and_forbids_placeholder_symbols() -> None:
    llm = ScriptedLLM(GOOD_DRAFT)
    ModelingAgent(llm).run(
        _idea(),
        "minimise weighted tardiness on one machine",
        template_text="# starter",
        domain_profile={"typical_models": ["time-indexed MILP", "disjunctive MILP"]},
        literature_brief="Indexed literature for this study: a paper on dispatching",
    )
    prompt = llm.prompts[0]

    assert "minimise weighted tardiness on one machine" in prompt
    assert "a dispatching rule closes most of the gap" in prompt
    assert "time-indexed MILP" in prompt
    assert "a paper on dispatching" in prompt, "the literature must reach modeling too"
    assert "# starter" in prompt
    assert "F(x, y, z)" in prompt, "the placeholder symbols must be named as forbidden"
    assert "would falsify the hypothesis" in prompt


def test_the_mock_path_offers_the_scaffold_as_a_template_not_a_draft() -> None:
    draft, source, _, _ = ModelingAgent(LLMClient(use_mock=True)).run(_idea(), "goal")
    assert source == "not_generated"
    assert "a form to fill in, not a draft of your model" in draft.markdown
    assert "No language model was configured" in draft.markdown


def test_a_fenced_response_is_unwrapped() -> None:
    llm = ScriptedLLM(f"```\n{GOOD_DRAFT}\n```")
    draft, source, _, _ = ModelingAgent(llm).run(_idea(), "goal")
    assert source == "llm"
    assert draft.markdown.lstrip().startswith("# Mathematical Model Draft")
