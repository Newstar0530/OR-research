"""The detector must detect. These tests break the transformation on purpose."""

import pytest

from src.core.bilevel import BILEVEL_DIFFICULTIES, bard_example, generate_bilevel_lp
from src.core.bilevel_oracle import solve_bilevel_by_vertex_enumeration
from src.core.equivalence import (
    _classify,
    sweep_big_m,
    verify_transformation_chain,
)
from src.core.mixed_integer_program import solve_mip
from src.core.transformations import derive_big_m_bounds, to_kkt_one_level


def test_the_chain_is_equivalent_on_the_textbook_instance() -> None:
    report = verify_transformation_chain(bard_example())
    assert report.chain_is_equivalent
    assert report.oracle_value == pytest.approx(-12.0, abs=1e-6)
    for step in report.steps:
        assert step.verdict == "equivalent"
        assert step.solution_verified
    assert report.big_m_rigorous
    assert "bilevel_to_kkt" in report.to_markdown()


def test_the_chain_is_equivalent_across_difficulties() -> None:
    for difficulty in BILEVEL_DIFFICULTIES:
        for seed in range(2):
            blp = generate_bilevel_lp(2, seed=seed, difficulty=difficulty)
            report = verify_transformation_chain(blp)
            assert report.chain_is_equivalent, (difficulty, seed, report.to_markdown())


def test_a_sign_error_in_the_kkt_derivation_is_caught() -> None:
    """Flip the stationarity right-hand side -- the classic derivation slip."""

    blp = bard_example()
    kkt = to_kkt_one_level(blp)
    for constraint in kkt.linear_constraints:
        if constraint.name.startswith("stationarity"):
            constraint.rhs = -constraint.rhs

    report = verify_transformation_chain(blp, kkt=kkt)
    step = report.step("bilevel_to_kkt")
    assert step is not None
    assert not step.passed, "a corrupted KKT system must not be reported as equivalent"
    assert report.chain_is_equivalent is False


def test_a_big_m_that_is_too_small_is_caught() -> None:
    """Too-small constants remove the optimum; that must never read as success."""

    blp = bard_example()
    bounds = derive_big_m_bounds(blp)
    report = verify_transformation_chain(
        blp,
        slack_big_m=[v * 0.1 for v in bounds.slack_bounds],
        dual_big_m=[v * 0.1 for v in bounds.dual_bounds],
    )
    step = report.step("kkt_to_mi01")
    assert step is not None
    assert step.verdict in ("model_infeasible", "model_worse_than_truth")
    assert step.passed is False
    assert report.big_m_source == "manual"


def test_the_folklore_large_flat_big_m_does_not_survive_verification() -> None:
    """`M = 1e6` lets a binary sit inside the integrality tolerance and still switch a multiplier."""

    report = verify_transformation_chain(bard_example(), slack_big_m=1e6, dual_big_m=1e6)
    step = report.step("kkt_to_mi01")
    assert step is not None
    assert step.verdict != "equivalent" or not step.solution_verified


def test_dropping_complementarity_is_classified_as_beating_the_truth() -> None:
    """A relaxation that admits points the bilevel program forbids must be named as such."""

    found = False
    for difficulty in BILEVEL_DIFFICULTIES:
        for seed in range(4):
            blp = generate_bilevel_lp(2, seed=seed, difficulty=difficulty)
            oracle = solve_bilevel_by_vertex_enumeration(blp)
            if oracle.status != "optimal":
                continue
            relaxed = solve_mip(to_kkt_one_level(blp).as_linear_program(), time_limit=30)
            if relaxed.status != "optimal" or relaxed.objective is None:
                continue
            if relaxed.objective < oracle.upper_objective - 1e-6:
                verdict, _, _ = _classify(
                    oracle.upper_objective, relaxed.objective, True, relaxed.status
                )
                assert verdict == "model_better_than_truth"
                found = True
                break
        if found:
            break
    assert found, "expected at least one instance where dropping complementarity helps the leader"


def test_classification_covers_every_outcome() -> None:
    assert _classify(-12.0, -12.0, True, "optimal")[0] == "equivalent"
    assert _classify(-12.0, -21.0, True, "optimal")[0] == "model_better_than_truth"
    assert _classify(-12.0, -5.0, True, "optimal")[0] == "model_worse_than_truth"
    assert _classify(-12.0, None, True, "infeasible")[0] == "model_infeasible"
    assert _classify(None, -12.0, True, "optimal")[0] == "reference_unavailable"
    # Direction matters: for a maximising leader the signs flip.
    assert _classify(10.0, 20.0, False, "optimal")[0] == "model_better_than_truth"
    assert _classify(10.0, 5.0, False, "optimal")[0] == "model_worse_than_truth"


def test_sweep_locates_a_threshold_and_reports_it_honestly() -> None:
    sweep = sweep_big_m(bard_example(), scales=(0.1, 0.5, 1.0, 10.0), uniform_values=(1000.0,))
    assert sweep.reference_value == pytest.approx(-12.0, abs=1e-6)
    by_scale = {p.scale: p for p in sweep.points if p.scale is not None}
    assert by_scale[0.1].verdict != "equivalent"
    assert by_scale[1.0].verdict == "equivalent" and by_scale[1.0].solution_verified
    assert by_scale[10.0].verdict == "equivalent"
    assert sweep.smallest_equivalent_scale is not None
    assert sweep.smallest_equivalent_scale <= 1.0
    assert sweep.monotone
    assert "Big-M Sensitivity" in sweep.to_markdown()


def test_derived_bounds_are_close_to_necessary() -> None:
    """Halving the derived constants should already break the reformulation."""

    broken = 0
    for seed in range(3):
        blp = generate_bilevel_lp(2, seed=seed)
        sweep = sweep_big_m(blp, scales=(0.5, 1.0))
        by_scale = {p.scale: p for p in sweep.points}
        assert by_scale[1.0].verdict == "equivalent"
        if by_scale[0.5].verdict != "equivalent":
            broken += 1
    assert broken >= 1, "if halving never breaks anything, the bounds are loose"
