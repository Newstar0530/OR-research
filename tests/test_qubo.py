import numpy as np
import pytest

from src.core.qubo import (
    QUBOModel,
    QuadraticTerm,
    solve_qubo,
    solve_qubo_by_annealing,
    solve_qubo_by_enumeration,
)


def _random_qubo(n: int = 8, seed: int = 0) -> QUBOModel:
    rng = np.random.default_rng(seed)
    return QUBOModel(
        name="random",
        n_bits=n,
        linear=[float(v) for v in rng.integers(-5, 6, n)],
        quadratic=[
            QuadraticTerm(i=i, j=j, coefficient=float(rng.integers(-4, 5)))
            for i in range(n)
            for j in range(i + 1, n)
            if rng.random() < 0.4
        ],
        offset=3.0,
        bit_names=[f"b{i}" for i in range(n)],
    )


def test_energy_matches_the_matrix_form() -> None:
    qubo = _random_qubo()
    rng = np.random.default_rng(1)
    matrix = qubo.to_matrix()
    for _ in range(50):
        bits = rng.integers(0, 2, qubo.n_bits)
        assert qubo.energy(bits) == pytest.approx(bits @ matrix @ bits + qubo.offset)
    with pytest.raises(ValueError):
        qubo.energy([0, 1])


def test_ising_form_preserves_energy() -> None:
    """Hardware speaks Ising; a conversion that shifts energies changes the problem."""

    qubo = _random_qubo(seed=2)
    h, J, offset = qubo.to_ising()
    rng = np.random.default_rng(3)
    for _ in range(200):
        bits = rng.integers(0, 2, qubo.n_bits)
        spins = 2 * bits - 1
        ising = offset + h @ spins + spins @ np.triu(J, 1) @ spins
        assert qubo.energy(bits) == pytest.approx(ising, abs=1e-9)


def test_enumeration_proves_the_ground_state_and_annealing_does_not() -> None:
    qubo = _random_qubo(seed=4)
    exact = solve_qubo_by_enumeration(qubo)
    annealed = solve_qubo_by_annealing(qubo, seed=1)
    assert exact.status == "optimal"
    assert annealed.status == "feasible", "a sampler must never claim optimality"
    assert annealed.energy >= exact.energy - 1e-9
    assert qubo.energy(exact.bits) == pytest.approx(exact.energy)
    assert qubo.energy(annealed.bits) == pytest.approx(annealed.energy)


def test_enumeration_refuses_models_it_cannot_sweep() -> None:
    qubo = QUBOModel(name="big", n_bits=30, linear=[1.0] * 30, bit_names=[f"b{i}" for i in range(30)])
    result = solve_qubo_by_enumeration(qubo, max_bits=10)
    assert result.status == "not_run"
    assert "max_bits" in result.notes


def test_auto_solver_picks_proof_when_it_can_and_says_so_when_it_cannot() -> None:
    small = _random_qubo(n=6, seed=5)
    assert solve_qubo(small).backend == "qubo_enumeration"
    big = QUBOModel(name="big", n_bits=26, linear=[1.0] * 26, bit_names=[f"b{i}" for i in range(26)])
    result = solve_qubo(big, max_bits_for_proof=20, time_limit=5)
    assert result.backend == "qubo_annealing"
    assert result.status == "feasible"
    assert "beyond exhaustive proof" in result.notes


def test_diagnostics_describe_the_model() -> None:
    qubo = _random_qubo(seed=6)
    assert 0.0 <= qubo.density <= 1.0
    assert qubo.max_abs_coefficient > 0
    assert qubo.dynamic_range >= 1.0
    empty = QUBOModel(name="empty", n_bits=0, offset=2.0)
    assert solve_qubo_by_enumeration(empty).energy == 2.0
    assert solve_qubo_by_annealing(empty).energy == 2.0
