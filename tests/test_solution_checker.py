from src.core.binary_program import BinaryProgramInstance, LinearConstraint, generate_instance
from src.core.solution_checker import check_binary_solution, verify_solver_claim


def _instance() -> BinaryProgramInstance:
    return BinaryProgramInstance(
        instance_id="t",
        family="test",
        objective_sense="maximize",
        objective_coefficients=[3.0, 5.0, 1.0],
        constraints=[LinearConstraint(name="cap", coefficients=[2.0, 4.0, 1.0], sense="<=", rhs=5.0)],
    )


def test_feasible_solution_is_accepted_and_objective_recomputed() -> None:
    report = check_binary_solution(_instance(), [0, 1, 1])
    assert report.feasible is True
    assert report.n_violated == 0
    assert report.recomputed_objective == 6.0


def test_violation_is_reported_with_magnitude() -> None:
    report = check_binary_solution(_instance(), [1, 1, 1])
    assert report.feasible is False
    assert report.n_violated == 1
    assert report.max_violation == 2.0
    assert report.violations[0].name == "cap"
    assert "cap" in report.to_markdown()


def test_non_binary_value_is_a_domain_error() -> None:
    report = check_binary_solution(_instance(), [2, 0, 0])
    assert report.feasible is False
    assert any("binary domain" in message for message in report.domain_errors)


def test_fractional_value_is_a_domain_error() -> None:
    report = check_binary_solution(_instance(), [0.5, 0, 0])
    assert report.feasible is False
    assert any("not integral" in message for message in report.domain_errors)


def test_wrong_length_and_missing_solution_are_domain_errors() -> None:
    assert check_binary_solution(_instance(), [1, 0]).feasible is False
    assert check_binary_solution(_instance(), None).feasible is False


def test_claim_of_infeasible_solution_is_caught() -> None:
    _, ok, problems = verify_solver_claim(_instance(), [1, 1, 1], "optimal", 9.0)
    assert ok is False
    assert any("violated constraint" in problem for problem in problems)


def test_misreported_objective_is_caught() -> None:
    _, ok, problems = verify_solver_claim(_instance(), [0, 1, 1], "optimal", 999.0)
    assert ok is False
    assert any("recomputing" in problem for problem in problems)


def test_honest_claim_passes_and_no_claim_is_not_contradicted() -> None:
    _, ok, problems = verify_solver_claim(_instance(), [0, 1, 1], "optimal", 6.0)
    assert ok is True and problems == []
    _, ok, problems = verify_solver_claim(_instance(), None, "no_solution", None)
    assert ok is True and problems == []


def test_generated_instances_accept_the_empty_and_reject_the_full_solution() -> None:
    instance = generate_instance("knapsack", 10, seed=1)
    assert check_binary_solution(instance, [0] * 10).feasible is True
    assert check_binary_solution(instance, [1] * 10).feasible is False
