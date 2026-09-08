import pytest

from src.core.binary_program import BinaryProgramInstance, LinearConstraint, generate_instance
from src.core.solve_result import SolveResult
from src.core.solver_backends import (
    BackendOutcome,
    BruteForceBackend,
    CpSatBackend,
    GreedyBackend,
    LocalSearchBackend,
    MethodSpec,
    PyomoBackend,
    SolverBackend,
    available_methods,
    build_method_specs,
    compute_ground_truth,
    describe_backends,
    solve_instance,
)


SMALL_FAMILIES = ["knapsack", "multi_knapsack", "set_cover"]


def test_availability_is_reported_for_every_method() -> None:
    for method, (available, reason) in available_methods().items():
        assert isinstance(available, bool)
        if not available:
            assert reason, f"{method} must explain why it is unavailable"


def test_describe_backends_is_inspectable() -> None:
    rows = describe_backends()
    assert {row["method"] for row in rows} >= {"brute_force", "greedy", "local_search"}
    for row in rows:
        assert row["role"] in {"reference", "exact", "baseline", "proposed"}
        assert row["kind"] in {"exact", "heuristic"}


def test_brute_force_refuses_instances_that_are_too_large() -> None:
    instance = generate_instance("knapsack", 12, seed=0)
    outcome = BruteForceBackend(max_vars=8).solve(instance)
    assert outcome.status == "not_run"
    assert "max_vars" in outcome.notes


def test_brute_force_detects_infeasibility() -> None:
    instance = BinaryProgramInstance(
        instance_id="impossible",
        family="test",
        objective_sense="minimize",
        objective_coefficients=[1.0, 1.0],
        constraints=[
            LinearConstraint(name="at_least_two", coefficients=[1.0, 1.0], sense=">=", rhs=2.0),
            LinearConstraint(name="at_most_one", coefficients=[1.0, 1.0], sense="<=", rhs=1.0),
        ],
    )
    assert BruteForceBackend().solve(instance).status == "infeasible"


@pytest.mark.parametrize("family", SMALL_FAMILIES)
def test_cp_sat_agrees_with_exhaustive_enumeration(family: str) -> None:
    pytest.importorskip("ortools")
    for seed in range(3):
        instance = generate_instance(family, 12, seed=seed)
        exhaustive = BruteForceBackend().solve(instance, time_limit=60)
        cp_sat = CpSatBackend().solve(instance, time_limit=30)
        assert exhaustive.status == "optimal"
        assert cp_sat.status == "optimal"
        assert exhaustive.objective == pytest.approx(cp_sat.objective)


@pytest.mark.parametrize("family", SMALL_FAMILIES)
def test_heuristics_are_feasible_and_never_beat_the_true_optimum(family: str) -> None:
    for seed in range(3):
        instance = generate_instance(family, 14, seed=seed)
        optimum = BruteForceBackend().solve(instance, time_limit=60).objective
        greedy = solve_instance(instance, "greedy", time_limit=5, known_optimum=optimum)
        local = solve_instance(instance, "local_search", time_limit=5, known_optimum=optimum)

        assert greedy.feasible and greedy.verification_ok
        assert local.feasible and local.verification_ok
        assert greedy.solver_status == "feasible", "a heuristic must never claim optimality"
        assert local.solver_status == "feasible"
        assert not instance.is_better(greedy.objective, optimum)
        assert not instance.is_better(local.objective, optimum)
        assert greedy.gap_to_known_optimum >= -1e-9
        assert local.gap_to_known_optimum >= -1e-9


@pytest.mark.parametrize("family", SMALL_FAMILIES)
def test_local_search_is_never_worse_than_its_greedy_start(family: str) -> None:
    for seed in range(4):
        instance = generate_instance(family, 16, seed=seed)
        greedy = GreedyBackend().solve(instance)
        local = LocalSearchBackend().solve(instance, time_limit=3, seed=seed)
        assert not instance.is_better(greedy.objective, local.objective)


def test_hard_instances_leave_a_real_gap() -> None:
    """The experiment must be able to show the heuristic losing, not just winning."""

    gaps = []
    for seed in range(5):
        instance = generate_instance("knapsack", 16, seed=seed, difficulty="strongly_correlated")
        optimum = BruteForceBackend().solve(instance, time_limit=60).objective
        result = solve_instance(instance, "greedy", time_limit=5, known_optimum=optimum)
        gaps.append(result.gap_to_known_optimum)
    assert max(gaps) > 0.0, "strongly correlated instances should defeat the greedy heuristic"


class _LyingBackend(SolverBackend):
    """Claims a proved optimum while returning a solution that breaks a constraint."""

    name = "liar"
    kind = "exact"

    def solve(self, instance, time_limit=10.0, seed=0):
        return BackendOutcome(
            status="optimal",
            objective=10_000.0,
            dual_bound=10_000.0,
            solution=[1] * instance.n_vars,
            notes="fabricated",
        )


class _CrashingBackend(SolverBackend):
    name = "crasher"
    kind = "heuristic"

    def solve(self, instance, time_limit=10.0, seed=0):
        raise RuntimeError("backend exploded")


def test_a_lying_backend_is_downgraded_to_error() -> None:
    instance = generate_instance("knapsack", 10, seed=1)
    specs = {"liar": MethodSpec("liar", _LyingBackend(), "exact", "test double")}
    result = solve_instance(instance, "liar", specs=specs)
    assert result.solver_status == "error"
    assert result.verification_ok is False
    assert result.is_trustworthy is False
    assert "VERIFICATION FAILURE" in result.notes
    # The objective written to the table is the recomputed one, not the claim.
    assert result.objective != 10_000.0


def test_a_crashing_backend_becomes_a_finding_not_a_silent_zero() -> None:
    instance = generate_instance("knapsack", 8, seed=1)
    specs = {"crasher": MethodSpec("crasher", _CrashingBackend(), "baseline", "test double")}
    result = solve_instance(instance, "crasher", specs=specs)
    assert result.solver_status == "error"
    assert result.verification_ok is False
    assert "backend exploded" in result.notes


def test_unknown_method_and_unavailable_backend_degrade_cleanly() -> None:
    instance = generate_instance("knapsack", 8, seed=1)
    unknown = solve_instance(instance, "no_such_method")
    assert unknown.solver_status == "error" and "Unknown method" in unknown.notes

    available, reason = PyomoBackend().availability()
    if not available:
        result = solve_instance(instance, "exact_pyomo")
        assert result.solver_status == "not_run"
        assert result.notes
        assert isinstance(result, SolveResult)


def test_ground_truth_is_only_recorded_when_proved() -> None:
    instance = compute_ground_truth(generate_instance("multi_knapsack", 14, seed=2))
    assert instance.known_optimum is not None
    assert instance.known_optimum_source.endswith("_proved")

    # With every exact method switched off, nothing may be recorded as optimal.
    specs = build_method_specs()
    specs.pop("brute_force")
    specs.pop("exact_cp_sat")
    specs.pop("exact_pyomo")
    unproved = compute_ground_truth(generate_instance("knapsack", 10, seed=2), specs=specs)
    assert unproved.known_optimum is None
    assert unproved.known_optimum_source == "unproved"
