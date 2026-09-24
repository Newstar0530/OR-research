"""The formulation is an object, and the document is rendered from it.

While the formulation was a string, everything that mattered about it -- what a
parameter is measured in, which requirement a constraint exists to satisfy,
whether a variable is binary -- survived only as English inside a paragraph.
Prose can describe a variable without ever saying it is binary and name a
constraint without writing one, and nothing could tell the difference.

These tests are mostly about what the type refuses to let through.
"""

import json

from src.agents.modeling_agent import ModelingAgent
from src.core.model_exporter import export_model_skeletons
from src.core.model_ir import build_model_ir, write_model_ir
from src.core.or_ir import (
    ConstraintSpec,
    ObjectiveSpec,
    ObjectiveTerm,
    OptimizationModel,
    ParameterSpec,
    SetSpec,
    VariableSpec,
)
from src.llm_client import LLMClient
from src.schemas import ResearchIdea


def _model(**overrides) -> OptimizationModel:
    base = dict(
        problem_name="nurse_rostering",
        sets=[SetSpec(name="N", description="nurses", index="n")],
        parameters=[ParameterSpec(name="c_ns", description="assignment cost", unit="currency", source="R1")],
        decision_variables=[
            VariableSpec(name="x_ns", description="assignment", domain="binary", indexed_by=["n", "s"])
        ],
        objective=ObjectiveSpec(
            sense="minimize",
            expression="sum_n sum_s c_ns * x_ns",
            terms=[ObjectiveTerm(expression="c_ns * x_ns", business_meaning="cost of staffing a shift")],
        ),
        constraints=[
            ConstraintSpec(
                name="coverage",
                expression="sum_n x_ns >= r_s",
                source_requirement="R1",
                indexed_by=["s"],
            )
        ],
        assumptions=["Every nurse is qualified for every shift."],
        source="llm_structured",
    )
    base.update(overrides)
    return OptimizationModel(**base)


# -- what the structure refuses --------------------------------------------


def test_a_complete_model_is_implementable() -> None:
    report = _model().structural_report()

    assert report.is_implementable
    assert report.failures() == []


def test_a_variable_without_a_domain_blocks_implementation() -> None:
    """Prose hides this; a field cannot."""

    report = _model(
        decision_variables=[VariableSpec(name="x_ns", description="assignment")]
    ).structural_report()

    assert not report.is_implementable
    assert any("no declared domain" in item for item in report.failures())
    assert any("x_ns" in item for item in report.failures())


def test_a_constraint_named_but_not_written_blocks_implementation() -> None:
    report = _model(
        constraints=[ConstraintSpec(name="capacity constraints", source_requirement="R1")]
    ).structural_report()

    assert not report.is_implementable
    assert any("named but none is written as an expression" in item for item in report.failures())


def test_an_objective_without_an_expression_blocks_implementation() -> None:
    report = _model(objective=ObjectiveSpec(sense="minimize")).structural_report()

    assert not report.is_implementable
    assert any("nothing to optimise" in item for item in report.failures())


def test_a_missing_unit_warns_but_does_not_block() -> None:
    """A unitless parameter is how minutes get added to hours and called an optimum."""

    report = _model(
        parameters=[ParameterSpec(name="c_ns", description="assignment cost")]
    ).structural_report()

    assert report.is_implementable, "it can still be built, just not checked for scale"
    assert any("no unit" in item for item in report.warnings)


def test_a_soft_constraint_with_no_penalty_is_flagged_as_free_to_violate() -> None:
    report = _model(
        constraints=[
            ConstraintSpec(name="rest", expression="s_j >= 11", enforcement="soft", source_requirement="R2")
        ]
    ).structural_report()

    assert any("free to violate" in item for item in report.warnings)


def test_a_constraint_citing_nothing_is_flagged() -> None:
    report = _model(
        constraints=[ConstraintSpec(name="coverage", expression="sum_n x_ns >= r_s")]
    ).structural_report()

    assert any("cite no requirement" in item for item in report.warnings)


# -- the document is downstream of the object ------------------------------


def test_the_rendered_document_carries_the_requirement_tag() -> None:
    """So the coverage check reads the recorded link rather than guessing it."""

    rendered = _model().render_markdown()

    assert "[R1] `coverage`: sum_n x_ns >= r_s for all s" in rendered
    assert "## Constraints" in rendered and "## Parameters" in rendered


def test_a_parameter_without_a_unit_says_so_in_the_document() -> None:
    rendered = _model(parameters=[ParameterSpec(name="c_ns", description="cost")]).render_markdown()

    assert "[unit: **not stated**]" in rendered


def test_the_rendered_document_passes_the_draft_inspection() -> None:
    """Rendering must not produce something the pipeline then rejects."""

    from src.core.model_draft_inspection import inspect_model_draft

    assert inspect_model_draft(_model().render_markdown()).is_specific


def test_business_meaning_survives_into_the_document() -> None:
    assert "cost of staffing a shift" in _model().render_markdown()


# -- the degraded path -----------------------------------------------------


def test_prose_recovers_into_the_same_type_but_marked_as_recovered() -> None:
    markdown = """
## Sets and Indices
- J: set of jobs, indexed by j
## Parameters
- p_j: processing time, in minutes
## Decision Variables
- x_j: 1 if selected, binary
## Objective Function
- Minimize total cost
## Constraints
- Capacity: sum p_j x_j <= B
"""
    recovered = build_model_ir(markdown, "demo")

    assert recovered.source == "markdown_extraction"
    assert recovered.problem_name == "demo"
    assert recovered.objective_sense == "minimize"
    assert recovered.parameters[0].unit == "minutes", "the one unit shape prose states often enough"
    assert recovered.decision_variables[0].domain == "binary"
    assert any("recovered from a document" in item for item in recovered.extraction_warnings)


def test_recovery_leaves_unstated_fields_empty_rather_than_inventing_them() -> None:
    recovered = build_model_ir(
        "## Decision Variables\n- x_i: the choice\n## Constraints\n- Capacity: sum x <= 1\n"
    )

    assert recovered.decision_variables[0].domain == "unspecified"
    assert recovered.constraints[0].source_requirement is None
    assert not recovered.structural_report().is_implementable


def test_an_ambiguous_direction_yields_no_objective_rather_than_a_guess() -> None:
    """Guessing here is how a model silently optimises the wrong way."""

    recovered = build_model_ir("## Objective\n- minimize or maximize F(x)")

    assert recovered.objective is None
    assert recovered.objective_sense == "unknown"


# -- artifacts -------------------------------------------------------------


def test_the_written_ir_records_its_structural_verdict(tmp_path) -> None:
    write_model_ir(_model(), tmp_path)
    payload = json.loads((tmp_path / "model_ir.json").read_text(encoding="utf-8"))

    assert payload["source"] == "llm_structured"
    assert payload["objective_sense"] == "minimize"
    assert payload["structural_report"]["missing"] == []
    assert (tmp_path / "model_ir.md").exists()


def test_the_exporter_declares_variables_it_has_domains_for(tmp_path) -> None:
    export_model_skeletons(_model(), tmp_path)
    pyomo = (tmp_path / "model_export_pyomo.py").read_text(encoding="utf-8")
    ortools = (tmp_path / "model_export_ortools.py").read_text(encoding="utf-8")

    assert "model.x_ns = pyo.Var(model.n, domain=pyo.Binary)" in pyomo
    assert 'x_ns = model.NewBoolVar("x_ns")' in ortools
    assert "# Constraint [R1] coverage (hard): sum_n x_ns >= r_s" in pyomo
    assert "# Param c_ns [currency]" in pyomo


def test_the_exporter_will_not_declare_a_variable_whose_domain_is_unknown(tmp_path) -> None:
    export_model_skeletons(
        _model(decision_variables=[VariableSpec(name="x_ns", description="assignment")]), tmp_path
    )
    pyomo = (tmp_path / "model_export_pyomo.py").read_text(encoding="utf-8")

    assert "# model.x_ns = pyo.Var(...)  # domain not declared in the IR" in pyomo


def test_the_diagnostics_count_what_is_declared_not_merely_what_exists(tmp_path) -> None:
    export_model_skeletons(
        _model(parameters=[ParameterSpec(name="c_ns", description="cost")]), tmp_path
    )
    diagnostics = (tmp_path / "model_export_diagnostics.md").read_text(encoding="utf-8")

    assert "- parameters: 1 (0 with a unit)" in diagnostics
    assert "- decision_variables: 1 (1 with a declared domain)" in diagnostics
    assert "1 citing a requirement" in diagnostics


# -- the modelling stage asks for fields first -----------------------------


class StructuredLLM(LLMClient):
    """A client that answers `chat_json`, the way a real provider would."""

    def __init__(self, payload: dict) -> None:
        super().__init__(use_mock=False)
        object.__setattr__(self, "payload", payload)
        object.__setattr__(self, "json_prompts", [])

    def chat_json(self, system_prompt: str, user_prompt: str, stage: str = "") -> dict:
        self.json_prompts.append(user_prompt)
        return self.payload

    def chat(self, system_prompt: str, user_prompt: str) -> str:
        raise AssertionError("the structured answer should have been accepted")


def _idea() -> ResearchIdea:
    return ResearchIdea(
        title="Nurse rostering", problem_context="shifts",
        core_hypothesis="a matheuristic closes the gap",
        expected_contribution="a comparison", proposed_model_type="MILP",
        proposed_algorithm_type="matheuristic", experimental_plan="compare",
        interestingness_score=6, novelty_score=4, feasibility_score=8, risk_score=3,
        assumptions=[], required_data=[], expected_outputs=[],
    )


def test_a_structured_answer_becomes_the_model_and_the_document() -> None:
    llm = StructuredLLM(json.loads(_model().model_dump_json()))
    draft, source, _, _ = ModelingAgent(llm).run(_idea(), "goal")

    assert source == "llm"
    assert draft.model is not None, "the typed formulation is kept, not only its rendering"
    assert draft.model.source == "llm_structured"
    assert draft.model.decision_variables[0].domain == "binary"
    assert "[R1] `coverage`" in draft.markdown, "the document is rendered from the object"


def test_the_structured_request_states_the_shape_it_wants() -> None:
    llm = StructuredLLM(json.loads(_model().model_dump_json()))
    ModelingAgent(llm).run(_idea(), "goal")

    prompt = llm.json_prompts[0]
    assert '"domain": "binary|integer|continuous"' in prompt
    assert '"source_requirement"' in prompt
    assert "naming a family is not a constraint" in prompt


def test_a_structurally_incomplete_answer_is_sent_back_with_the_gap_named() -> None:
    incomplete = json.loads(_model().model_dump_json())
    incomplete["decision_variables"][0]["domain"] = "unspecified"
    llm = StructuredLLM(incomplete)
    draft, source, _, _ = ModelingAgent(llm).run(_idea(), "goal")

    assert source == "not_generated", "two attempts, both structurally incomplete"
    assert draft.model is None
    assert "no declared domain" in draft.markdown
