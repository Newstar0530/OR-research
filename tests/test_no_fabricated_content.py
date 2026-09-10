"""No front-half stage may answer a question it was never given the means to answer.

Every test here is a specific thing the pipeline used to assert about a study
it had never read: a research gap in supply-chain chaos, a knapsack algorithm,
a review score of 6. The general rule they enforce is that an artifact is
either a function of this run's inputs or an admitted gap -- never a constant
dressed as a finding.
"""

from pathlib import Path

import pandas as pd
import pytest

from src.agents.algorithm_agent import AlgorithmAgent
from src.agents.critic_agent import CriticAgent
from src.agents.domain_classifier_agent import DomainClassifierAgent
from src.agents.gap_synthesis_agent import GapSynthesisAgent
from src.agents.modeling_agent import ModelingAgent
from src.agents.novelty_agent import NoveltyAgent
from src.agents.reviewer_agent import ReviewerAgent
from src.core.artifact_provenance import (
    GAP_MARKER,
    ProvenanceLedger,
    declared_gap,
    write_provenance,
)
from src.llm_client import LLMClient
from src.schemas import ResearchIdea


MOCK = LLMClient(use_mock=True)

#: Vocabulary from the boilerplate that used to be emitted for every study.
SUPPLY_CHAIN_BOILERPLATE = ("bullwhip", "chaos precursor", "multi-echelon", "replenishment")
KNAPSACK_BOILERPLATE = ("value density", "dynamic programming for exact optimal values")


def _idea(**overrides) -> ResearchIdea:
    payload = dict(
        title="Bilevel to QUBO transformation rules",
        problem_context="bilevel programming",
        core_hypothesis="A derived Big-M preserves the optimum",
        expected_contribution="verified transformation chain",
        proposed_model_type="bilevel linear program",
        proposed_algorithm_type="KKT reformulation with binary expansion",
        experimental_plan="sweep Big-M and penalty",
        interestingness_score=7, novelty_score=5, feasibility_score=8, risk_score=4,
        assumptions=[], required_data=[], expected_outputs=[],
    )
    payload.update(overrides)
    return ResearchIdea(**payload)


def _profile(name: str = "optimization") -> dict:
    return {
        "name": name,
        "metrics": ["objective value", "optimality gap"],
        "algorithm_families": ["branch and bound", "Lagrangian relaxation"],
        "keywords": ["mixed integer programming"],
        "sensitivity_parameters": ["problem size"],
    }


# -- the gap analysis ------------------------------------------------------


def test_the_gap_analysis_no_longer_invents_a_supply_chain_research_gap(tmp_path: Path) -> None:
    """The exact defect: a bilevel/QUBO study was told its gap was about bullwhip."""

    text, source, _ = GapSynthesisAgent(MOCK).run(
        "Transform a bilevel program to a mixed 0-1 program and solve it with a quantum algorithm",
        _profile(),
    )
    lowered = text.lower()
    for phrase in SUPPLY_CHAIN_BOILERPLATE:
        assert phrase not in lowered, f"boilerplate survived: {phrase}"
    assert source == "not_generated"
    assert GAP_MARKER in text


def test_an_absent_gap_says_what_it_would_take_and_what_it_does_not_mean(tmp_path: Path) -> None:
    text, _, _ = GapSynthesisAgent(MOCK).run("any goal at all", _profile())
    assert "use_mock_llm" in text
    assert "nothing here says a gap does not exist" in text


def test_the_gap_analysis_reports_what_the_index_really_holds(tmp_path: Path) -> None:
    index = tmp_path / "literature_index.csv"
    pd.DataFrame(
        [
            {"filename": "a.pdf", "keyword_hits": "mixed integer programming", "snippet": "x"},
            {"filename": "b.pdf", "keyword_hits": "", "snippet": "y"},
        ]
    ).to_csv(index, index=False)

    text, _, inputs = GapSynthesisAgent(MOCK).run("mixed integer programming study", _profile(), index)
    assert "`a.pdf`" in text and "`b.pdf`" in text
    assert "literature_index" in inputs
    assert "first three pages" in text, "the index's real limits must be stated"


def test_a_literature_folder_about_something_else_is_called_out(tmp_path: Path) -> None:
    index = tmp_path / "literature_index.csv"
    pd.DataFrame([{"filename": "unrelated.pdf", "keyword_hits": "", "snippet": ""}]).to_csv(index, index=False)
    text, _, _ = GapSynthesisAgent(MOCK).run("bilevel programming", _profile(), index)
    assert "No indexed document shares a single term" in text


# -- the algorithm plan ----------------------------------------------------


def test_the_algorithm_plan_no_longer_proposes_knapsack_for_everything() -> None:
    plan, source, _ = AlgorithmAgent(MOCK).run(_idea(), "draft", _profile())
    blob = (plan.exact_solver_plan + plan.heuristic_plan + plan.pseudocode).lower()
    for phrase in KNAPSACK_BOILERPLATE:
        assert phrase not in blob, f"boilerplate survived: {phrase}"
    assert source == "derived"


def test_the_algorithm_plan_offers_the_domain_families_without_choosing_one() -> None:
    plan, _, inputs = AlgorithmAgent(MOCK).run(_idea(), "draft", _profile())
    assert plan.baseline_methods == ["branch and bound", "Lagrangian relaxation"]
    assert "Not yet selected" in plan.exact_solver_plan
    assert GAP_MARKER in plan.exact_solver_plan
    assert "domain_profile" in inputs


def test_the_plan_admits_that_the_template_is_what_actually_runs() -> None:
    """The plan never drove the code; saying so stops it being read as the method."""

    plan, _, _ = AlgorithmAgent(MOCK).run(_idea(), "draft", _profile())
    assert "experiment template" in plan.pseudocode
    assert "NOT ANALYSED" in plan.complexity_discussion


def test_a_domain_with_no_declared_families_says_so_rather_than_defaulting() -> None:
    plan, _, _ = AlgorithmAgent(MOCK).run(_idea(), "draft", {"name": "unknown"})
    assert plan.baseline_methods == ["NOT SELECTED - no algorithm family is declared for this domain"]


# -- the review ------------------------------------------------------------


def test_an_unperformed_review_has_no_scores_at_all() -> None:
    """A constant 6 used to reach cross-run memory as if it were a measurement."""

    review, source = ReviewerAgent(MOCK).run("# Report\n\nSome results and a baseline comparison.")
    assert source == "not_generated"
    assert review.review_performed is False
    assert review.overall_score is None
    assert all(
        value is None
        for value in (
            review.soundness_score, review.novelty_score, review.technical_quality_score,
            review.reproducibility_score, review.presentation_score, review.confidence_score,
        )
    )
    assert "NOT REVIEWED" in review.summary
    assert "does not mean the work scored badly" in review.summary


def test_the_unreviewed_path_still_checks_the_report_it_was_given() -> None:
    thorough = "# Report\n## Results\n## Methodology\n## Limitations\n## Baseline\n## Statistical evidence\n" + "x" * 500
    thin = "# Report"

    rich, _ = ReviewerAgent(MOCK).run(thorough)
    poor, _ = ReviewerAgent(MOCK).run(thin)

    assert "limitations" in rich.strengths[0]
    assert any("does not mention" in item for item in poor.weaknesses)
    assert any("too short" in item for item in poor.weaknesses)


# -- the critique ----------------------------------------------------------


def test_the_critic_reads_the_draft_and_names_the_placeholders() -> None:
    draft, source, _ = ModelingAgent(MOCK).run(_idea(), "goal")
    draft = draft.markdown
    assert source == "not_generated", "the scaffold is a template, not a formulation"
    critique, source = CriticAgent(MOCK).run(draft)

    assert source == "derived"
    joined = " ".join(critique.critical_issues)
    assert "6 of 6 checked elements" in joined
    assert "F(x, y, z; c, d, theta)" in joined
    assert "No summation, inequality or quantifier" in joined


def test_a_specific_model_is_not_criticised_as_a_scaffold() -> None:
    """The old critique said "still a scaffold" about anything, including this."""

    real = """# Model
## Sets and Indices
- J: jobs, indexed by j
## Parameters
- p_j: processing time of job j
## Decision Variables
- s_j: start time of job j
## Objective Function
minimize sum_j w_j * max(0, s_j + p_j - d_j)
## Constraints
s_j >= 0 for all j
s_j + p_j <= s_k + M * (1 - y_jk)
## Assumptions
- A single machine, no preemption.
"""
    critique, _ = CriticAgent(MOCK).run(real)
    joined = " ".join(critique.critical_issues)
    assert "No scaffold placeholder was detected" in joined
    assert "this check confirms the draft is specific, not that it is correct" in joined


def test_an_empty_draft_is_reported_as_empty() -> None:
    critique, _ = CriticAgent(MOCK).run("")
    assert any("no model draft" in item.lower() for item in critique.critical_issues)


# -- novelty and classification --------------------------------------------


def test_novelty_queries_come_from_the_idea_not_a_fixed_list() -> None:
    scheduling = _idea(
        title="Tardiness minimisation on one machine",
        proposed_model_type="time-indexed MILP",
        proposed_algorithm_type="dispatching heuristic",
        core_hypothesis="dispatching beats the exact solver under a time limit",
    )
    report, source = NoveltyAgent(MOCK).run(scheduling)
    blob = " ".join(report.search_queries).lower()

    assert source == "derived"
    assert "time-indexed milp" in blob and "dispatching" in blob
    # The old fixed queries named these for every study, whatever it was about.
    for stale in ("qubo", "qaoa", "mpec", "bilevel"):
        assert stale not in blob, f"fixed query survived: {stale}"


def test_novelty_risk_is_unknown_rather_than_medium_when_nothing_searched() -> None:
    report, _ = NoveltyAgent(MOCK).run(_idea())
    assert report.similarity_risk == "unknown"
    assert "NOT ASSESSED" in report.novelty_summary


@pytest.mark.parametrize(
    "goal",
    [
        "Bilevel knapsack interdiction with KKT reformulation and QUBO encoding",
        "建立雙層規劃轉混合01整數規劃的轉化規則，最後用量子演算法求解",
    ],
)
def test_the_classifier_can_see_bilevel_and_quantum_work_in_either_language(goal: str) -> None:
    result = DomainClassifierAgent(MOCK).run(goal)
    assert result["confidence"] > 0.0
    assert result["matched_terms"], "the terms it matched must be shown, not just a score"


def test_zero_matches_reports_zero_confidence_and_says_it_defaulted() -> None:
    """0.45 reads as a weak match. No match is a different claim."""

    result = DomainClassifierAgent(MOCK).run("Predict groove width from femtosecond laser power")
    assert result["confidence"] == 0.0
    assert result["profile"] == "optimization"
    assert "fallback, not a classification" in result["reason"]
    assert result["matched_terms"] == []


# -- the ledger ------------------------------------------------------------


def test_the_ledger_headline_says_whether_anything_reasoned_about_the_study(tmp_path: Path) -> None:
    ledger = ProvenanceLedger()
    ledger.record("domain_classification", "derived", inputs_used=["research_goal"])
    ledger.record("gap_synthesis", "not_generated", "no model was available")

    headline = ledger.headline()
    assert "1 of 2" in headline
    assert "no stage used a language model" in headline

    path = write_provenance(tmp_path, ledger)
    text = path.read_text(encoding="utf-8")
    assert "Declared Gaps" in text
    assert (tmp_path / "artifact_provenance.json").exists()


def test_a_declared_gap_is_loud_enough_not_to_be_skimmed_past() -> None:
    block = declared_gap("A research gap", "nothing read the literature", ["an LLM"])
    assert block.startswith(f"> **{GAP_MARKER}:")
    assert "an LLM" in block


# -- end to end ------------------------------------------------------------


BOILERPLATE_THAT_MUST_NEVER_APPEAR = (
    "bullwhip",
    "chaos precursor",
    "multi-echelon",
    "value density",
    "stockout rate",
    "safety stock",
)


def test_a_whole_run_contains_no_text_about_a_study_it_never_read(tmp_path: Path) -> None:
    """The real check: run the pipeline on one topic and grep every artifact.

    A unit test can only pin the stage it names. This one would have caught the
    supply-chain gap analysis, the knapsack algorithm plan and the constant
    review score together, from the outside, without knowing where they lived.
    """

    import shutil

    from src.config import load_config
    from src.orchestrator import ResearchOrchestrator

    root = Path.cwd()
    project = tmp_path / "project"
    project.mkdir()
    for name in ("configs", "templates", "domain_profiles"):
        shutil.copytree(root / name, project / name)

    config = load_config(project / "configs" / "default.yaml")
    config.project_name = "bilevel_qubo"
    config.output_dir = str(project / "runs")
    config.research_goal = (
        "Develop transformation rules from bilevel programming to a one-level nonlinear program, "
        "then to a mixed 0-1 integer program, solved with a quantum algorithm."
    )
    run_dir = ResearchOrchestrator(config, project_root=project).run()

    offenders: list[str] = []
    for path in sorted(run_dir.glob("*.md")):
        lowered = path.read_text(encoding="utf-8", errors="ignore").lower()
        for phrase in BOILERPLATE_THAT_MUST_NEVER_APPEAR:
            if phrase in lowered:
                offenders.append(f"{path.name}: {phrase}")
    assert not offenders, "boilerplate about another study appeared: " + "; ".join(offenders)

    provenance = (run_dir / "artifact_provenance.md").read_text(encoding="utf-8")
    assert "no stage used a language model" in provenance
    assert GAP_MARKER in provenance

    review = (run_dir / "automated_review.md").read_text(encoding="utf-8")
    assert "No scores: no review was performed" in review
    assert "/10" not in review, "an unperformed review must not print a score"
