"""The QUBO detector must separate a cost from a defect, and catch the defect."""

import pytest

from src.core import qubo_transformations as qt
from src.core.mixed_integer_program import (
    MipConstraint,
    MipVariable,
    MixedIntegerProgram,
    solve_mip,
)
from src.core.qubo_equivalence import sweep_penalty, verify_qubo_chain


def _knapsack() -> MixedIntegerProgram:
    return MixedIntegerProgram(
        name="knap",
        sense="maximize",
        variables=[MipVariable(name=f"b{i}", is_binary=True) for i in range(6)],
        objective=[6.0, 5.0, 8.0, 9.0, 6.0, 7.0],
        constraints=[
            MipConstraint(
                name="cap", coefficients=[2.0, 3.0, 6.0, 7.0, 5.0, 9.0], sense="<=", rhs=15.0
            )
        ],
    )


def _off_grid() -> MixedIntegerProgram:
    """max x s.t. 2x <= 7, x in [0, 10]. The optimum is 3.5, off any integer grid."""

    return MixedIntegerProgram(
        name="offgrid",
        sense="maximize",
        variables=[MipVariable(name="x", upper=10.0)],
        objective=[1.0],
        constraints=[MipConstraint(name="cap", coefficients=[2.0], sense="<=", rhs=7.0)],
    )


def test_the_chain_holds_with_a_derived_penalty() -> None:
    report = verify_qubo_chain(_knapsack())
    assert report.chain_is_equivalent
    assert report.penalty_is_sound
    assert report.penalty_is_rigorous
    assert report.ground_state_proved
    assert report.discretisation_loss == pytest.approx(0.0)
    assert report.qubo_value == pytest.approx(report.true_value)
    assert "QUBO Equivalence" in report.to_markdown()


def test_a_penalty_that_is_too_small_is_caught_not_averaged() -> None:
    report = verify_qubo_chain(_knapsack(), penalty=0.05)
    step = report.step("binary_grid_to_qubo")
    assert step is not None
    assert step.verdict == "penalty_too_small"
    assert step.solution_verified is False
    assert step.passed is False
    assert report.penalty_source == "manual"
    # The infeasible bitstring looks *better* than the truth, which is the trap.
    assert report.qubo_value > report.true_value


def test_discretisation_is_a_measured_cost_not_a_defect() -> None:
    coarse = verify_qubo_chain(_off_grid(), precision=1.0)
    grid_step = coarse.step("mip_to_binary_grid")
    penalty_step = coarse.step("binary_grid_to_qubo")
    assert coarse.true_value == pytest.approx(3.5)
    assert coarse.grid_value == pytest.approx(3.0)
    assert grid_step.verdict == "discretisation_loss"
    assert coarse.discretisation_loss == pytest.approx(0.5)
    # The penalty step is untouched by the grid's cost.
    assert penalty_step.verdict == "equivalent"
    assert penalty_step.solution_verified


def test_a_finer_grid_removes_the_discretisation_loss() -> None:
    fine = verify_qubo_chain(_off_grid(), precision=0.5)
    assert fine.grid_value == pytest.approx(3.5)
    assert fine.discretisation_loss == pytest.approx(0.0)
    assert fine.chain_is_equivalent
    assert fine.n_bits > verify_qubo_chain(_off_grid(), precision=1.0).n_bits


def test_an_expansion_that_overshoots_is_caught_as_a_broken_projection(monkeypatch) -> None:
    """A grid that can express values the variable cannot must never look like good news.

    The variable's box is implicit in its expansion, so a wider expansion is a
    direct relaxation of that bound -- but only visible when the bound is what
    binds. Here `max x s.t. 2x <= 30, x <= 10` is optimal at the bound.
    """

    bound_binding = MixedIntegerProgram(
        name="bound_binding",
        sense="maximize",
        variables=[MipVariable(name="x", upper=10.0)],
        objective=[1.0],
        constraints=[MipConstraint(name="loose", coefficients=[2.0], sense="<=", rhs=30.0)],
    )
    assert solve_mip(bound_binding).objective == pytest.approx(10.0)

    original = qt.binary_expansion

    def overshooting(lower, upper, precision=1.0, name="v"):
        expansion = original(lower, upper, precision, name)
        expansion.weights = list(expansion.weights) + [4.0 * max(precision, 1.0)]
        return expansion

    monkeypatch.setattr(qt, "binary_expansion", overshooting)
    report = verify_qubo_chain(bound_binding, precision=1.0)
    step = report.step("mip_to_binary_grid")
    assert step is not None
    assert step.verdict == "grid_better_than_true"
    assert report.grid_value > report.true_value
    assert report.chain_is_equivalent is False


def test_an_unproved_ground_state_is_never_blamed_on_the_encoding() -> None:
    """Sampler non-convergence and a broken encoding are different findings."""

    for penalty in (None, 0.5, 5.0):
        report = verify_qubo_chain(_knapsack(), penalty=penalty, max_bits_for_proof=4)
        step = report.step("binary_grid_to_qubo")
        assert report.ground_state_proved is False
        assert step.verdict != "energy_encoding_mismatch", (
            "an encoding defect may only be diagnosed from a proved ground state"
        )


def test_a_model_too_large_to_prove_says_so_instead_of_claiming_equivalence() -> None:
    report = verify_qubo_chain(_knapsack(), max_bits_for_proof=4)
    step = report.step("binary_grid_to_qubo")
    assert report.ground_state_proved is False
    assert step.verdict in ("not_proved", "penalty_too_small")
    assert report.chain_is_equivalent is False


def test_the_source_model_being_infeasible_is_reported_not_transformed() -> None:
    infeasible = MixedIntegerProgram(
        name="bad",
        variables=[MipVariable(name="b", is_binary=True)],
        objective=[1.0],
        constraints=[MipConstraint(name="c", coefficients=[1.0], sense=">=", rhs=5.0)],
    )
    report = verify_qubo_chain(infeasible)
    assert report.steps == []
    assert any("nothing to preserve" in note for note in report.notes)


def test_the_sweep_locates_the_penalty_threshold() -> None:
    sweep = sweep_penalty(_knapsack(), scales=(0.0001, 0.001, 0.01, 1.0))
    by_scale = {p.scale: p for p in sweep.points}
    assert by_scale[1.0].verdict == "equivalent"
    assert by_scale[0.0001].verdict == "penalty_too_small"
    assert sweep.smallest_sound_scale is not None
    assert sweep.grid_value == pytest.approx(solve_mip(_knapsack()).objective)
    # The ratio that actually moves with the penalty is reported.
    ratios = [p.penalty_to_objective_ratio for p in sweep.points]
    assert ratios == sorted(ratios)
    assert "Penalty Sensitivity" in sweep.to_markdown()


def test_the_derived_bound_is_sufficient_but_reported_as_loose() -> None:
    """It is rigorous; the sweep shows how much slack that rigour costs."""

    sweep = sweep_penalty(_knapsack(), scales=(0.001, 0.01, 0.1, 1.0))
    sound = [p.scale for p in sweep.points if p.verdict == "equivalent"]
    assert sound, "the derived penalty must at least work at full strength"
    assert min(sound) < 1.0, "if nothing below 1.0 works the bound would be tight, not loose"
