import numpy as np
import pytest

from src.core.bilevel import (
    BILEVEL_DIFFICULTIES,
    BilevelLinearProgram,
    bard_example,
    generate_bilevel_lp,
)


def test_bounds_are_folded_into_the_lower_system() -> None:
    """G = [B2; I; -I] and h = [b2; y_ub; 0] -- one multiplier family, not three."""

    blp = generate_bilevel_lp(2, seed=0)
    E, G, h = blp.lower_system()
    m2, n_y = len(blp.b2), blp.n_y
    assert G.shape == (m2 + 2 * n_y, n_y)
    assert np.allclose(G[:m2], np.asarray(blp.B2))
    assert np.allclose(G[m2 : m2 + n_y], np.eye(n_y))
    assert np.allclose(G[m2 + n_y :], -np.eye(n_y))
    assert np.allclose(h[:m2], blp.b2)
    assert np.allclose(h[m2 : m2 + n_y], blp.y_upper)
    assert np.allclose(h[m2 + n_y :], 0.0)
    # The folded rows do not involve x.
    assert np.allclose(E[m2:], 0.0)
    assert blp.n_lower_rows == m2 + 2 * n_y


def test_folded_rows_really_encode_the_y_box() -> None:
    """The appended rows must accept exactly the points inside `0 <= y <= y_upper`."""

    blp = generate_bilevel_lp(2, seed=1)
    E, G, h = blp.lower_system()
    m2 = len(blp.b2)
    box_rows, box_rhs = G[m2:], h[m2:]  # the folded bound rows only
    inside = np.asarray(blp.y_upper) / 2.0
    above = np.asarray(blp.y_upper) + 1.0
    below = -np.ones(blp.n_y)
    assert np.all(box_rows @ inside <= box_rhs + 1e-9)
    assert np.any(box_rows @ above > box_rhs + 1e-9)
    assert np.any(box_rows @ below > box_rhs + 1e-9)


def test_joint_system_covers_every_constraint() -> None:
    blp = generate_bilevel_lp(2, seed=2)
    P, q, names = blp.joint_system()
    assert P.shape == (blp.n_upper_rows + blp.n_lower_rows + 2 * blp.n_x, blp.n_vars)
    assert len(names) == len(q)
    assert sum(1 for n in names if n.startswith("upper_")) == blp.n_upper_rows
    assert sum(1 for n in names if n.startswith("x_")) == 2 * blp.n_x


def test_generated_instances_are_feasible_at_the_origin() -> None:
    for difficulty in BILEVEL_DIFFICULTIES:
        for seed in range(4):
            blp = generate_bilevel_lp(2, seed=seed, difficulty=difficulty)
            assert blp.is_jointly_feasible([0.0] * blp.n_x, [0.0] * blp.n_y)


def test_generator_is_deterministic_and_responds_to_its_inputs() -> None:
    a = generate_bilevel_lp(3, seed=4, difficulty="coupled")
    assert a.model_dump() == generate_bilevel_lp(3, seed=4, difficulty="coupled").model_dump()
    assert a.b2 != generate_bilevel_lp(3, seed=5, difficulty="coupled").b2
    assert a.A2 != generate_bilevel_lp(3, seed=4, difficulty="balanced").A2


def test_generated_instances_have_real_bilevel_tension() -> None:
    """The leader must want large y while the follower pushes y down."""

    for difficulty in BILEVEL_DIFFICULTIES:
        blp = generate_bilevel_lp(3, seed=0, difficulty=difficulty)
        assert all(v < 0 for v in blp.d1), "leader must reward large y"
        assert all(v > 0 for v in blp.d2), "follower must penalise large y"


def test_degenerate_difficulty_makes_the_follower_objective_parallel_to_a_row() -> None:
    blp = generate_bilevel_lp(3, seed=2, difficulty="degenerate")
    assert np.allclose(blp.d2, blp.B2[0])


def test_objective_arithmetic_and_violation_reporting() -> None:
    blp = bard_example()
    assert blp.upper_objective([4.0], [4.0]) == pytest.approx(-12.0)
    assert blp.lower_objective([4.0]) == pytest.approx(4.0)
    assert blp.is_jointly_feasible([4.0], [4.0])
    violation, broken = blp.joint_violation([0.0], [0.0])
    assert violation > 0 and "lower_0" in broken


def test_shape_validation_rejects_inconsistent_data() -> None:
    with pytest.raises(ValueError):
        BilevelLinearProgram(
            instance_id="bad", n_x=1, n_y=1, c1=[1.0, 2.0], d1=[1.0], d2=[1.0],
            x_upper=[1.0], y_upper=[1.0],
        )
    with pytest.raises(ValueError):
        BilevelLinearProgram(
            instance_id="bad", n_x=1, n_y=1, c1=[1.0], d1=[1.0], d2=[1.0],
            A2=[[1.0]], B2=[[1.0]], b2=[1.0, 2.0], x_upper=[1.0], y_upper=[1.0],
        )
