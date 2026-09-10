import numpy as np
import pytest

from src.core.mixed_integer_program import (
    MipConstraint,
    MipVariable,
    MixedIntegerProgram,
    solve_mip,
    verify_mip_solution,
)
from src.core.qubo import solve_qubo_by_enumeration
from src.core.qubo_transformations import (
    binary_expansion,
    build_qubo,
    derive_penalty_bound,
    to_binary_grid_mip,
    to_qubo,
)


def _enumerate_values(expansion) -> set[float]:
    expansion.bit_indices = list(range(expansion.n_bits))
    values = set()
    for mask in range(1 << expansion.n_bits):
        bits = [(mask >> k) & 1 for k in range(expansion.n_bits)]
        values.add(round(expansion.value(bits), 9))
    return values


@pytest.mark.parametrize(
    "lower,upper,precision,expected_points",
    [(0, 7, 1, 8), (0, 10, 1, 11), (0, 1, 1, 2), (2, 5, 1, 4), (0, 3.5, 0.5, 8), (0, 100, 1, 101)],
)
def test_expansion_covers_the_grid_without_overshooting(
    lower, upper, precision, expected_points
) -> None:
    """An expansion that can express values the variable cannot is a relaxation."""

    expansion = binary_expansion(lower, upper, precision, "v")
    values = _enumerate_values(expansion)
    assert expansion.representable_upper == pytest.approx(upper)
    assert min(values) == pytest.approx(lower)
    assert max(values) == pytest.approx(upper)
    assert len(values) == expected_points


def test_expansion_rejects_what_it_cannot_encode() -> None:
    with pytest.raises(ValueError):
        binary_expansion(0.0, float("inf"), 1.0, "unbounded")
    with pytest.raises(ValueError):
        binary_expansion(5.0, 1.0, 1.0, "empty")
    with pytest.raises(ValueError):
        binary_expansion(0.0, 5.0, 0.0, "zero_precision")
    fixed = binary_expansion(3.0, 3.0, 1.0, "fixed")
    assert fixed.n_bits == 0 and fixed.value([]) == 3.0


def test_grid_model_keeps_the_objective_through_the_substitution() -> None:
    mip = MixedIntegerProgram(
        name="m",
        sense="maximize",
        variables=[MipVariable(name="x", lower=2.0, upper=6.0), MipVariable(name="z", is_binary=True)],
        objective=[3.0, -1.0],
        objective_offset=7.0,
        constraints=[MipConstraint(name="c", coefficients=[1.0, 1.0], sense="<=", rhs=5.0)],
    )
    grid, encoding = to_binary_grid_mip(mip, precision=1.0)
    assert all(v.is_binary for v in grid.variables)
    rng = np.random.default_rng(0)
    for _ in range(64):
        bits = [int(v) for v in rng.integers(0, 2, grid.n_vars)]
        decoded = encoding.decode(bits)
        assert grid.objective_value(bits) == pytest.approx(mip.objective_value(decoded))
        for original, projected in zip(mip.constraints, grid.constraints):
            assert projected.lhs(bits) - projected.rhs == pytest.approx(
                original.lhs(decoded) - original.rhs
            )


def test_a_binary_source_model_loses_nothing_to_the_grid() -> None:
    mip = MixedIntegerProgram(
        sense="maximize",
        variables=[MipVariable(name=f"b{i}", is_binary=True) for i in range(4)],
        objective=[3.0, 5.0, 2.0, 4.0],
        constraints=[
            MipConstraint(name="c", coefficients=[2.0, 3.0, 1.0, 2.0], sense="<=", rhs=4.0)
        ],
    )
    grid, encoding = to_binary_grid_mip(mip)
    assert encoding.is_lossless
    assert solve_mip(grid).objective == pytest.approx(solve_mip(mip).objective)


def test_penalty_bound_is_rigorous_only_when_the_data_is_integral() -> None:
    integral = MixedIntegerProgram(
        variables=[MipVariable(name="b", is_binary=True)],
        objective=[3.0],
        constraints=[MipConstraint(name="c", coefficients=[1.0], sense="<=", rhs=1.0)],
    )
    bound = derive_penalty_bound(integral)
    assert bound.is_rigorous and bound.penalty > bound.objective_range

    fractional = MixedIntegerProgram(
        variables=[MipVariable(name="b", is_binary=True)],
        objective=[3.0],
        constraints=[MipConstraint(name="c", coefficients=[0.37], sense="<=", rhs=0.5)],
    )
    loose = derive_penalty_bound(fractional)
    assert loose.is_rigorous is False
    assert any("heuristic" in note for note in loose.notes)


def test_to_qubo_refuses_models_it_cannot_encode() -> None:
    grid, _ = to_binary_grid_mip(
        MixedIntegerProgram(
            variables=[MipVariable(name="b", is_binary=True)],
            objective=[1.0],
            constraints=[MipConstraint(name="c", coefficients=[1.0], sense="<=", rhs=1.0)],
        )
    )
    with pytest.raises(ValueError, match="positive finite"):
        to_qubo(grid, penalty=float("inf"))
    with pytest.raises(ValueError, match="positive finite"):
        to_qubo(grid, penalty=0.0)
    not_binary = MixedIntegerProgram(
        variables=[MipVariable(name="x", upper=5.0)], objective=[1.0]
    )
    with pytest.raises(ValueError, match="all-binary"):
        to_qubo(not_binary, penalty=10.0)


def test_an_unsatisfiable_row_is_reported_at_encoding_time() -> None:
    grid = MixedIntegerProgram(
        variables=[MipVariable(name="b", is_binary=True)],
        objective=[1.0],
        constraints=[MipConstraint(name="impossible", coefficients=[1.0], sense=">=", rhs=5.0)],
    )
    with pytest.raises(ValueError, match="cannot be satisfied"):
        to_qubo(grid, penalty=10.0)


def test_the_ground_state_decodes_to_the_optimum_with_a_derived_penalty() -> None:
    mip = MixedIntegerProgram(
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
    qubo, encoding, grid, bound = build_qubo(mip)
    assert bound.is_rigorous
    solution = solve_qubo_by_enumeration(qubo, max_bits=24)
    assert solution.status == "optimal"
    decoded = encoding.grid.decode(solution.bits[: grid.n_vars])
    assert verify_mip_solution(mip, decoded).feasible
    assert mip.objective_value(decoded) == pytest.approx(solve_mip(mip).objective)
    # At a feasible point every residual is zero, so energy is the signed objective.
    assert solution.energy == pytest.approx(-mip.objective_value(decoded))
