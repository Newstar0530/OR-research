from src.schemas import ResearchIdea


def test_research_idea_score_bounds() -> None:
    idea = ResearchIdea(
        title="T",
        problem_context="C",
        core_hypothesis="H",
        expected_contribution="E",
        proposed_model_type="MILP",
        proposed_algorithm_type="Greedy",
        experimental_plan="Run tests",
        interestingness_score=5,
        novelty_score=5,
        feasibility_score=5,
        risk_score=5,
        assumptions=["A"],
        required_data=["D"],
        expected_outputs=["O"],
    )
    assert idea.feasibility_score == 5

