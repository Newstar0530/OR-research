import numpy as np
import pytest

from src.core import mixed_integer_program as mip_module
from src.core.mixed_integer_program import (
    MipConstraint,
    MipSolution,
    MipVariable,
    MixedIntegerProgram,
    available_mip_backends,
    solve_lp,
    solve_mip,
    solve_mip_by_enumeration,
    solve_mip_scipy,
    verify_mip_solution,
)


def _demo() -> MixedIntegerProgram:
    return MixedIntegerProgram(
        name="demo",
        sense="maximize",
        variables=[
            MipVariable(name="x", upper=10.0),
            MipVariable(name="z1", is_binary=True),
            MipVariable(name="z2", is_binary=True),
        ],
        objective=[1.0, 7.0, 5.0],
        constraints=[
            MipConstraint(name="budget", coefficients=[1.0, 4.0, 3.0], sense="<=", rhs=8.0),
            MipConstraint(name="pick_one", coefficients=[0.0, 1.0, 1.0], sense="<=", rhs=1.0),
        ],
    )


def test_model_introspection() -> None:
    model = _demo()
    assert model.n_vars == 3 and model.n_binaries == 1 + 1
    assert model.binary_indices == [1, 2]
    assert model.index_of("z2") == 2
    assert model.objective_value([4.0, 1.0, 0.0]) == pytest.approx(11.0)
    with pytest.raises(KeyError):
        model.index_of("nope")
    with pytest.raises(ValueError):
        model.objective_value([1.0])


def test_matrix_views_negate_greater_than_rows() -> None:
    model = MixedIntegerProgram(
        variables=[MipVariable(name="a", upper=5.0)],
        objective=[1.0],
        constraints=[
            MipConstraint(name="lo", coefficients=[1.0], sense=">=", rhs=2.0),
            MipConstraint(name="eq", coefficients=[1.0], sense="==", rhs=3.0),
        ],
    )
    a_ub, b_ub = model.inequality_rows()
    a_eq, b_eq = model.equality_rows()
    assert np.allclose(a_ub, [[-1.0]]) and np.allclose(b_ub, [-2.0])
    assert np.allclose(a_eq, [[1.0]]) and np.allclose(b_eq, [3.0])


def test_verification_catches_every_kind_of_bad_solution() -> None:
    model = _demo()
    assert verify_mip_solution(model, [4.0, 1.0, 0.0]).feasible
    assert verify_mip_solution(model, [5.0, 1.0, 0.0]).n_violated == 1
    assert verify_mip_solution(model, [4.0, 0.5, 0.0]).domain_errors
    assert verify_mip_solution(model, [40.0, 1.0, 0.0]).domain_errors
    assert verify_mip_solution(model, None).feasible is False
    assert verify_mip_solution(model, [1.0]).feasible is False


def test_backends_agree_on_random_models() -> None:
    rng = np.random.default_rng(0)
    for _ in range(8):
        n_cont, n_bin = 2, 4
        variables = [MipVariable(name=f"c{i}", upper=6.0) for i in range(n_cont)]
        variables += [MipVariable(name=f"b{i}", is_binary=True) for i in range(n_bin)]
        objective = [float(v) for v in rng.integers(-5, 6, size=n_cont + n_bin)]
        constraints = [
            MipConstraint(
                name=f"r{k}",
                coefficients=[float(v) for v in rng.integers(-3, 4, size=n_cont + n_bin)],
                sense="<=",
                rhs=float(rng.integers(3, 12)),
            )
            for k in range(3)
        ]
        model = MixedIntegerProgram(
            sense="maximize", variables=variables, objective=objective, constraints=constraints
        )
        highs = solve_mip_scipy(model, time_limit=20)
        exhaustive = solve_mip_by_enumeration(model, time_limit=60)
        assert highs.status == exhaustive.status
        if highs.status == "optimal":
            assert highs.objective == pytest.approx(exhaustive.objective, abs=1e-6)


def test_solve_mip_reports_a_backend_disagreement_instead_of_a_number(monkeypatch) -> None:
    """A backend that stops early but claims optimality must not be believed.

    The solution it returns is perfectly feasible, so verification alone would
    pass it. Only the second, independent solve exposes that a better optimum
    exists -- which is exactly why the cross-check is there.
    """

    model = _demo()

    def stops_early(mip, time_limit=30.0):
        # Feasible (4 + 3 <= 8, one binary set) but worth 9, not the true 11.
        return MipSolution(
            status="optimal", objective=9.0, values=[4.0, 0.0, 1.0], backend="stops_early"
        )

    monkeypatch.setattr(mip_module, "solve_mip_scipy", stops_early)
    solution = solve_mip(model, backend="scipy_milp", time_limit=20)
    assert solution.status == "error"
    assert "CROSS-CHECK FAILURE" in solution.notes
    assert "11" in solution.notes, "the note should name the value enumeration proved"


def test_solve_mip_rejects_a_solution_that_breaks_the_model(monkeypatch) -> None:
    model = _demo()

    def infeasible_backend(mip, time_limit=30.0):
        return MipSolution(
            status="optimal", objective=99.0, values=[10.0, 1.0, 1.0], backend="liar"
        )

    monkeypatch.setattr(mip_module, "solve_mip_scipy", infeasible_backend)
    solution = solve_mip(model, backend="scipy_milp", time_limit=20, cross_check=False)
    assert solution.status == "error"
    assert "VERIFICATION FAILURE" in solution.notes


def test_reported_objective_is_recomputed_from_the_model(monkeypatch) -> None:
    model = _demo()

    def sloppy_backend(mip, time_limit=30.0):
        return MipSolution(
            status="optimal", objective=10.5, values=[4.0, 1.0, 0.0], backend="sloppy"
        )

    monkeypatch.setattr(mip_module, "solve_mip_scipy", sloppy_backend)
    solution = solve_mip(model, backend="scipy_milp", time_limit=20, cross_check=False)
    assert solution.objective == pytest.approx(11.0)
    assert "recomputed" in solution.notes


def test_infeasible_and_unbounded_are_distinguished() -> None:
    infeasible = MixedIntegerProgram(
        variables=[MipVariable(name="x", upper=1.0)],
        objective=[1.0],
        constraints=[MipConstraint(name="a", coefficients=[1.0], sense=">=", rhs=5.0)],
    )
    assert solve_mip(infeasible).status == "infeasible"
    unbounded = MixedIntegerProgram(
        sense="maximize",
        variables=[MipVariable(name="x", upper=None)],
        objective=[1.0],
    )
    assert solve_lp(unbounded).status == "unbounded"


def test_enumeration_refuses_models_with_too_many_binaries() -> None:
    model = MixedIntegerProgram(
        variables=[MipVariable(name=f"b{i}", is_binary=True) for i in range(6)],
        objective=[1.0] * 6,
    )
    solution = solve_mip_by_enumeration(model, max_binaries=3)
    assert solution.status == "not_run"
    assert "max_binaries" in solution.notes


def test_unknown_backend_and_availability_reporting() -> None:
    assert solve_mip(_demo(), backend="nope").status == "error"
    for name, (available, reason) in available_mip_backends().items():
        assert isinstance(available, bool)
        if not available:
            assert reason
