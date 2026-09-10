import pytest

from src.core.bilevel import BILEVEL_DIFFICULTIES, bard_example, generate_bilevel_lp
from src.core.bilevel_oracle import (
    solve_bilevel_by_sampling,
    solve_bilevel_by_vertex_enumeration,
    solve_lower_level,
    verify_bilevel_solution,
)


def test_oracle_reproduces_the_textbook_optimum() -> None:
    """x = 4, y = 4, F = -12 -- found without KKT, complementarity or Big-M."""

    result = solve_bilevel_by_vertex_enumeration(bard_example())
    assert result.status == "optimal"
    assert result.upper_objective == pytest.approx(-12.0, abs=1e-6)
    assert result.x[0] == pytest.approx(4.0, abs=1e-6)
    assert result.y[0] == pytest.approx(4.0, abs=1e-6)


def test_sampling_agrees_with_the_vertex_oracle() -> None:
    """Sampling only ever returns bilevel-feasible points, so it cannot beat the optimum."""

    for difficulty in BILEVEL_DIFFICULTIES:
        for seed in range(3):
            blp = generate_bilevel_lp(2, seed=seed, difficulty=difficulty)
            oracle = solve_bilevel_by_vertex_enumeration(blp)
            sampled = solve_bilevel_by_sampling(blp, samples=200, seed=1)
            assert oracle.status == "optimal"
            if sampled.status == "optimal":
                assert oracle.upper_objective <= sampled.upper_objective + 1e-6


def test_the_oracle_optimum_survives_independent_verification() -> None:
    for difficulty in BILEVEL_DIFFICULTIES:
        blp = generate_bilevel_lp(2, seed=0, difficulty=difficulty)
        oracle = solve_bilevel_by_vertex_enumeration(blp)
        check = verify_bilevel_solution(blp, oracle.x, oracle.y)
        assert check.is_bilevel_feasible
        assert check.follower_suboptimality == pytest.approx(0.0, abs=1e-6)


def test_lower_level_is_solved_exactly() -> None:
    blp = bard_example()
    # At x = 4 the follower minimises y subject to y >= 3 - x = -1 and y >= 2x - 12 = -4,
    # y <= 12 - 2x = 4 and y >= (3x - 4)/2 = 4, so the follower's optimum is y = 4.
    solution = solve_lower_level(blp, [4.0])
    assert solution.status == "optimal"
    assert solution.value == pytest.approx(4.0, abs=1e-6)


def test_verification_rejects_a_point_the_follower_would_not_choose() -> None:
    blp = bard_example()
    check = verify_bilevel_solution(blp, [3.0], [6.0])
    assert check.jointly_feasible is True, "the point satisfies every constraint..."
    assert check.lower_level_optimal is False, "...but the follower would never pick it"
    assert check.follower_suboptimality > 1.0
    assert check.is_bilevel_feasible is False


def test_verification_rejects_a_jointly_infeasible_point() -> None:
    blp = bard_example()
    check = verify_bilevel_solution(blp, [0.0], [0.0])
    assert check.jointly_feasible is False
    assert check.broken_rows
    assert check.is_bilevel_feasible is False


def test_verification_rejects_wrong_dimensions() -> None:
    blp = generate_bilevel_lp(2, seed=0)
    assert verify_bilevel_solution(blp, [0.0], [0.0, 0.0, 0.0]).is_bilevel_feasible is False


def test_enumeration_refuses_to_run_when_the_subset_count_explodes() -> None:
    blp = generate_bilevel_lp(3, seed=0)
    result = solve_bilevel_by_vertex_enumeration(blp, max_combinations=5)
    assert result.status == "not_run"
    assert "max_combinations" in result.notes
