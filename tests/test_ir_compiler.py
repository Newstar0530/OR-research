"""The declared formulation becomes a model the solver layer can run.

The project has had a rigorous solver path for a while -- verified solutions,
exhaustive cross-checks, oracle-compared transformations -- and a research
pipeline that never reached it. This is the bridge, so the tests that matter
most are the ones where it refuses to cross: a parser that guesses would
produce a model that solves cleanly while encoding something else, which is the
failure the requirement check exists to prevent, reintroduced one layer down.
"""

import pytest

from src.core.ir_compiler import CompileRefusal, _parse_constraint, compile_to_mip
from src.core.mixed_integer_program import solve_mip, verify_mip_solution
from src.core.or_ir import (
    ConstraintSpec,
    ObjectiveSpec,
    OptimizationModel,
    ParameterSpec,
    SetSpec,
    VariableSpec,
)


def _knapsackish(**overrides) -> OptimizationModel:
    """Two binaries, one budget, written out in ground form."""

    base = dict(
        problem_name="tiny_selection",
        parameters=[
            ParameterSpec(name="budget", description="spend cap", unit="currency", value=10.0),
            ParameterSpec(name="cost_a", description="cost of a", unit="currency", value=6.0),
            ParameterSpec(name="cost_b", description="cost of b", unit="currency", value=5.0),
        ],
        decision_variables=[
            VariableSpec(name="x_a", description="take a", domain="binary"),
            VariableSpec(name="x_b", description="take b", domain="binary"),
        ],
        objective=ObjectiveSpec(sense="maximize", expression="3 * x_a + 4 * x_b"),
        constraints=[
            ConstraintSpec(
                name="budget_limit",
                expression="cost_a * x_a + cost_b * x_b <= budget",
                source_requirement="R1",
            )
        ],
        source="llm_structured",
    )
    base.update(overrides)
    return OptimizationModel(**base)


# -- what compiles ---------------------------------------------------------


def test_a_ground_linear_model_compiles() -> None:
    report = compile_to_mip(_knapsackish())

    assert report.compiled
    assert report.refusals == []
    assert report.variable_order == ["x_a", "x_b"]
    assert report.mip.sense == "maximize"
    assert report.mip.n_binaries == 2
    assert report.mip.objective == [3.0, 4.0]
    assert report.mip.constraints[0].coefficients == [6.0, 5.0]
    assert report.mip.constraints[0].rhs == 10.0
    assert report.mip.constraints[0].sense == "<="


def test_the_compiled_model_solves_and_verifies() -> None:
    """The point of the bridge: the research model reaches the rigorous path."""

    report = compile_to_mip(_knapsackish())
    solution = solve_mip(report.mip, time_limit=10.0)

    assert solution.status == "optimal"
    # Only one of the two fits in a budget of 10, and b is worth more.
    assert solution.objective == pytest.approx(4.0)
    assert verify_mip_solution(report.mip, solution.values).feasible


def test_the_requirement_link_survives_into_the_executable_model() -> None:
    """So a solved model can still say which requirement each row came from."""

    report = compile_to_mip(_knapsackish())

    assert report.mip.metadata["requirement_links"] == {"budget_limit": "R1"}
    assert report.mip.metadata["model_source"] == "llm_structured"


def test_variables_on_both_sides_are_collected() -> None:
    report = compile_to_mip(
        _knapsackish(
            constraints=[ConstraintSpec(name="ordering", expression="2 * x_a - 3 >= x_b + 1")]
        )
    )

    assert report.compiled
    assert report.mip.constraints[0].coefficients == [2.0, -1.0]
    assert report.mip.constraints[0].rhs == 4.0


def test_brackets_and_unary_minus_are_handled() -> None:
    report = compile_to_mip(
        _knapsackish(
            constraints=[ConstraintSpec(name="grouped", expression="-2 * (x_a - x_b) <= 1")]
        )
    )

    assert report.mip.constraints[0].coefficients == [-2.0, 2.0]


def test_a_constant_in_the_objective_becomes_an_offset_not_a_dropped_term() -> None:
    report = compile_to_mip(
        _knapsackish(objective=ObjectiveSpec(sense="maximize", expression="3 * x_a + 4 * x_b + 7"))
    )

    assert report.mip.objective_offset == 7.0
    assert report.mip.objective_value([1.0, 0.0]) == pytest.approx(10.0)


def test_coefficients_follow_declaration_order_not_alphabetical_order() -> None:
    """A positional vector built in the wrong order is silently the wrong model.

    Nothing downstream could catch it: the model would solve, verify against
    its own scrambled constraints, and report an optimum for a problem nobody
    wrote.
    """

    report = compile_to_mip(
        _knapsackish(
            decision_variables=[
                VariableSpec(name="z_first", description="declared first", domain="binary"),
                VariableSpec(name="a_second", description="declared second", domain="binary"),
            ],
            objective=ObjectiveSpec(sense="maximize", expression="z_first"),
            constraints=[ConstraintSpec(name="only_a", expression="a_second <= 1")],
        )
    )

    assert report.variable_order == ["z_first", "a_second"]
    assert report.mip.objective == [1.0, 0.0]
    assert report.mip.constraints[0].coefficients == [0.0, 1.0]


# -- what it refuses -------------------------------------------------------


def test_an_indexed_model_is_refused_with_what_is_missing() -> None:
    report = compile_to_mip(
        _knapsackish(
            sets=[SetSpec(name="J", description="items", index="j")],
            decision_variables=[
                VariableSpec(name="x_j", description="take j", domain="binary", indexed_by=["J"])
            ],
            objective=ObjectiveSpec(sense="maximize", expression="x_j"),
            constraints=[ConstraintSpec(name="budget_limit", expression="x_j <= 1")],
        )
    )

    assert not report.compiled and report.mip is None
    assert "needs the members of those sets" in report.refusals[0].reason
    assert "`J`" in report.refusals[0].reason


def test_a_summation_is_refused_rather_than_approximated() -> None:
    report = compile_to_mip(
        _knapsackish(
            constraints=[ConstraintSpec(name="budget_limit", expression="sum_j c_j x_j <= budget")]
        )
    )

    assert not report.compiled
    assert "quantifies over a set" in report.refusals[0].reason


def test_a_symbol_with_no_value_is_refused_by_name() -> None:
    report = compile_to_mip(
        _knapsackish(
            constraints=[ConstraintSpec(name="budget_limit", expression="theta * x_a <= budget")]
        )
    )

    assert not report.compiled
    assert "`theta` is neither a declared decision variable nor a parameter with a value" in (
        report.refusals[0].reason
    )


def test_a_nonlinear_product_is_refused() -> None:
    report = compile_to_mip(
        _knapsackish(constraints=[ConstraintSpec(name="pairing", expression="x_a * x_b <= 1")])
    )

    assert not report.compiled
    assert "not linear" in report.refusals[0].reason


def test_an_integer_variable_is_refused_rather_than_re_encoded() -> None:
    report = compile_to_mip(
        _knapsackish(
            decision_variables=[VariableSpec(name="n_a", description="count", domain="integer")]
        )
    )

    assert not report.compiled
    assert "would have to be re-encoded" in report.refusals[0].reason


def test_a_soft_constraint_is_refused_because_moving_it_is_a_modelling_decision() -> None:
    report = compile_to_mip(
        _knapsackish(
            constraints=[
                ConstraintSpec(
                    name="preference",
                    expression="x_a <= 1",
                    enforcement="soft",
                    penalty="100 per violation",
                )
            ]
        )
    )

    assert not report.compiled
    assert "belongs in the objective with its penalty" in report.refusals[0].reason


def test_a_strict_inequality_is_refused() -> None:
    with pytest.raises(CompileRefusal, match="no strict inequalities"):
        _parse_constraint("c", "x_a < 1", ["x_a"], {})


def test_a_double_inequality_is_refused_as_two_constraints() -> None:
    with pytest.raises(CompileRefusal, match="two constraints"):
        _parse_constraint("c", "0 <= x_a <= 1", ["x_a"], {})


def test_an_expression_with_no_comparison_is_not_a_constraint() -> None:
    with pytest.raises(CompileRefusal, match="not a constraint"):
        _parse_constraint("c", "x_a + 1", ["x_a"], {})


def test_a_constraint_mentioning_no_variable_constrains_nothing() -> None:
    with pytest.raises(CompileRefusal, match="constrains nothing"):
        _parse_constraint("c", "3 <= 4", ["x_a"], {})


def test_a_symbolic_bound_without_a_value_is_refused() -> None:
    report = compile_to_mip(
        _knapsackish(
            decision_variables=[
                VariableSpec(name="y", description="level", domain="continuous", upper_bound="u_max")
            ]
        )
    )

    assert not report.compiled
    assert "`u_max` is a symbol with no value" in report.refusals[0].reason


def test_a_symbolic_bound_that_names_a_valued_parameter_resolves() -> None:
    report = compile_to_mip(
        _knapsackish(
            decision_variables=[
                VariableSpec(name="y", description="level", domain="continuous", upper_bound="budget")
            ],
            objective=ObjectiveSpec(sense="maximize", expression="y"),
            constraints=[ConstraintSpec(name="cap", expression="y <= budget")],
        )
    )

    assert report.compiled
    assert report.mip.variables[0].upper == 10.0


# -- the report ------------------------------------------------------------


def test_the_report_says_a_refusal_is_not_a_verdict_on_the_model() -> None:
    report = compile_to_mip(
        _knapsackish(
            constraints=[ConstraintSpec(name="budget_limit", expression="sum_j c_j x_j <= budget")]
        )
    )
    document = report.markdown()

    assert "not a judgement of the model" in document
    assert "budget_limit" in document
    assert "quantifies over a set" in document


def test_the_report_refuses_to_claim_correctness_from_compiling() -> None:
    document = compile_to_mip(_knapsackish()).markdown()

    assert "does not establish that the model is the right one" in document


# -- the artifact the run leaves behind ------------------------------------


def test_the_run_records_the_recomputed_objective_not_the_reported_one(tmp_path) -> None:
    """The bridge, end to end: a declared formulation becomes a checked result."""

    import json
    from pathlib import Path

    from src.config import load_config
    from src.orchestrator import ResearchOrchestrator

    config = load_config(Path.cwd() / "configs" / "default.yaml")
    config.output_dir = str(tmp_path)
    orchestrator = ResearchOrchestrator(config, project_root=Path.cwd())

    report = compile_to_mip(_knapsackish())
    orchestrator._solve_compiled_model(report.mip, tmp_path)
    recorded = json.loads((tmp_path / "model_solution.json").read_text(encoding="utf-8"))

    assert recorded["status"] == "optimal"
    assert recorded["feasible"] is True
    assert recorded["recomputed_objective"] == pytest.approx(4.0)
    assert recorded["max_violation"] == pytest.approx(0.0)
    assert recorded["values"] == {"x_a": 0.0, "x_b": 1.0}
    document = (tmp_path / "model_solution.md").read_text(encoding="utf-8")
    assert "template with its own numbers" in document
    assert "cross-checked against exhaustive enumeration" in document, (
        "the second opinion is what makes the recomputed objective worth recording"
    )
