import pytest

from src.core.binary_program import (
    BinaryProgramInstance,
    LinearConstraint,
    generate_instance,
    list_families,
)


def test_all_families_are_generatable() -> None:
    assert set(list_families()) == {"knapsack", "multi_knapsack", "set_cover"}
    for family in list_families():
        instance = generate_instance(family, 10, seed=0)
        assert instance.n_vars == 10
        assert instance.n_constraints >= 1
        assert instance.is_integral, "coefficients must stay integral for exact solvers"


def test_same_tuple_gives_identical_instance() -> None:
    a = generate_instance("multi_knapsack", 12, seed=3, difficulty="weakly_correlated")
    b = generate_instance("multi_knapsack", 12, seed=3, difficulty="weakly_correlated")
    assert a.model_dump() == b.model_dump()


def test_seed_and_difficulty_change_the_instance() -> None:
    base = generate_instance("knapsack", 12, seed=1, difficulty="uncorrelated")
    other_seed = generate_instance("knapsack", 12, seed=2, difficulty="uncorrelated")
    other_difficulty = generate_instance("knapsack", 12, seed=1, difficulty="strongly_correlated")
    assert base.objective_coefficients != other_seed.objective_coefficients
    assert base.objective_coefficients != other_difficulty.objective_coefficients


def test_strongly_correlated_knapsack_ties_profit_to_weight() -> None:
    instance = generate_instance("knapsack", 8, seed=4, difficulty="strongly_correlated")
    weights = instance.constraints[0].coefficients
    for profit, weight in zip(instance.objective_coefficients, weights):
        assert profit == weight + 10


def test_set_cover_is_always_coverable() -> None:
    instance = generate_instance("set_cover", 10, seed=7)
    all_ones = [1] * instance.n_vars
    assert all(c.is_satisfied(all_ones) for c in instance.constraints)
    assert instance.objective_sense == "minimize"


def test_objective_and_violation_arithmetic() -> None:
    instance = BinaryProgramInstance(
        instance_id="t",
        family="test",
        objective_sense="maximize",
        objective_coefficients=[3.0, 5.0],
        constraints=[
            LinearConstraint(name="cap", coefficients=[2.0, 4.0], sense="<=", rhs=4.0),
            LinearConstraint(name="cover", coefficients=[1.0, 1.0], sense=">=", rhs=1.0),
        ],
    )
    assert instance.objective_value([1, 1]) == 8.0
    assert instance.constraints[0].violation([1, 1]) == 2.0
    assert instance.constraints[0].violation([1, 0]) == 0.0
    assert instance.constraints[1].violation([0, 0]) == 1.0
    assert instance.is_better(9.0, 8.0) is True
    assert instance.is_better(7.0, 8.0) is False


def test_objective_value_rejects_wrong_length() -> None:
    instance = generate_instance("knapsack", 5, seed=0)
    with pytest.raises(ValueError):
        instance.objective_value([1, 0])


def test_instance_round_trips_through_json(tmp_path) -> None:
    instance = generate_instance("multi_knapsack", 9, seed=2)
    path = instance.to_json_file(tmp_path / "instance.json")
    assert BinaryProgramInstance.from_json_file(path).model_dump() == instance.model_dump()


def test_unknown_family_is_rejected() -> None:
    with pytest.raises(KeyError):
        generate_instance("travelling_salesman", 10, seed=0)
