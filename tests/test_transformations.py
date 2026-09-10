import numpy as np
import pytest

from src.core.bilevel import BILEVEL_DIFFICULTIES, bard_example, generate_bilevel_lp
from src.core.bilevel_oracle import solve_bilevel_by_vertex_enumeration
from src.core.mixed_integer_program import solve_mip
from src.core.transformations import (
    derive_big_m_bounds,
    solve_kkt_by_pattern_enumeration,
    to_kkt_one_level,
    to_mi01,
)


def test_kkt_program_has_the_expected_shape() -> None:
    blp = generate_bilevel_lp(2, seed=0)
    kkt = to_kkt_one_level(blp)
    assert kkt.n_vars == blp.n_x + blp.n_y + blp.n_lower_rows
    assert kkt.n_lambda == blp.n_lower_rows
    assert len(kkt.complementarity) == blp.n_lower_rows
    names = [c.name for c in kkt.linear_constraints]
    assert sum(1 for n in names if n.startswith("primal_")) == blp.n_lower_rows
    assert sum(1 for n in names if n.startswith("stationarity_")) == blp.n_y


def test_stationarity_rows_encode_G_transpose_lambda_equals_minus_d2() -> None:
    blp = generate_bilevel_lp(2, seed=1)
    kkt = to_kkt_one_level(blp)
    _, G, _ = blp.lower_system()
    offset = blp.n_x + blp.n_y
    for j in range(blp.n_y):
        row = next(c for c in kkt.linear_constraints if c.name == f"stationarity_y{j}")
        assert row.sense == "=="
        assert row.rhs == pytest.approx(-blp.d2[j])
        assert np.allclose(row.coefficients[offset:], G[:, j])
        assert np.allclose(row.coefficients[:offset], 0.0)


def test_kkt_reformulation_reproduces_the_true_optimum() -> None:
    """The load-bearing correctness claim: no Big-M involved, so nothing to tune."""

    for difficulty in BILEVEL_DIFFICULTIES:
        for seed in range(3):
            blp = generate_bilevel_lp(2, seed=seed, difficulty=difficulty)
            oracle = solve_bilevel_by_vertex_enumeration(blp)
            kkt = solve_kkt_by_pattern_enumeration(to_kkt_one_level(blp))
            assert oracle.status == "optimal" and kkt.status == "optimal"
            assert kkt.objective == pytest.approx(oracle.upper_objective, abs=1e-6)


def test_slack_bounds_hold_everywhere_in_the_box() -> None:
    """Interval arithmetic must bound the slack for every point in the box, not just feasible ones."""

    rng = np.random.default_rng(0)
    for seed in range(3):
        blp = generate_bilevel_lp(2, seed=seed)
        E, G, h = blp.lower_system()
        bounds = np.asarray(derive_big_m_bounds(blp).slack_bounds)
        for _ in range(200):
            x = rng.uniform(0.0, blp.x_upper)
            y = rng.uniform(0.0, blp.y_upper)
            slack = h - E @ x - G @ y
            assert np.all(slack <= bounds + 1e-9)


def test_dual_bounds_hold_for_the_multipliers_that_actually_occur() -> None:
    for difficulty in BILEVEL_DIFFICULTIES:
        blp = generate_bilevel_lp(2, seed=0, difficulty=difficulty)
        kkt = to_kkt_one_level(blp)
        bounds = derive_big_m_bounds(blp)
        solution = solve_kkt_by_pattern_enumeration(kkt)
        assert solution.status == "optimal"
        _, _, multipliers = kkt.split(solution.values)
        assert np.all(np.asarray(multipliers) <= np.asarray(bounds.dual_bounds) + 1e-6)


def test_derived_bounds_are_finite_where_the_naive_lp_bound_is_not() -> None:
    """Folding the y bounds in makes the dual set unbounded; vertices are still finite."""

    blp = bard_example()
    bounds = derive_big_m_bounds(blp)
    assert bounds.is_rigorous
    assert bounds.dual_vertices_found > 0
    assert all(np.isfinite(bounds.dual_bounds))
    assert any(bounds.dual_set_unbounded), "the LP bound should be unbounded here"


def test_mi01_with_derived_bounds_reproduces_the_true_optimum() -> None:
    for difficulty in BILEVEL_DIFFICULTIES:
        for seed in range(2):
            blp = generate_bilevel_lp(2, seed=seed, difficulty=difficulty)
            oracle = solve_bilevel_by_vertex_enumeration(blp)
            kkt = to_kkt_one_level(blp)
            bounds = derive_big_m_bounds(blp)
            mi01 = to_mi01(kkt, bounds.slack_bounds, bounds.dual_bounds)
            assert mi01.n_binaries == kkt.n_lambda
            solution = solve_mip(mi01, time_limit=30)
            assert solution.status == "optimal"
            assert solution.objective == pytest.approx(oracle.upper_objective, abs=1e-6)


def test_an_infinite_big_m_is_refused_rather_than_approximated() -> None:
    kkt = to_kkt_one_level(bard_example())
    with pytest.raises(ValueError, match="finite"):
        to_mi01(kkt, 1.0, float("inf"))


def test_pattern_enumeration_refuses_to_run_when_it_would_explode() -> None:
    kkt = to_kkt_one_level(generate_bilevel_lp(3, seed=0))
    solution = solve_kkt_by_pattern_enumeration(kkt, max_patterns=4)
    assert solution.status == "not_run"
    assert "max_patterns" in solution.notes
