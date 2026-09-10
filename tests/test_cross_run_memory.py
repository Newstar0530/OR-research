"""Memory is only useful if it is read back, and only safe if it stays modest.

These tests pin both halves: a record written by one run must come back intact
to the next, and what comes back must be phrased as evidence from that run
rather than as a settled fact about the world.
"""

import json
from pathlib import Path

from src.core.research_journal import ResearchJournal
from src.core.research_node import ResearchNode
from src.utils.research_memory import (
    RunRecord,
    append_memory,
    build_prior_findings,
    find_related_runs,
    goal_similarity,
    load_run_records,
    record_run,
    summarize_journal,
    write_prior_findings,
)


def _record(**overrides) -> RunRecord:
    payload = {
        "project_name": "run-a",
        "research_goal": "bilevel knapsack scheduling with big-M linearisation",
        "run_dir": "/runs/a",
        "primary_metric": "objective",
        "objective_direction": "minimize",
        "best_metric": 12.0,
        "evidence_strength": "inconclusive_not_significant",
        "supports_improvement_claim": False,
    }
    payload.update(overrides)
    return RunRecord(**payload)


# -- round trip ------------------------------------------------------------


def test_a_record_written_by_one_run_comes_back_to_the_next(tmp_path: Path) -> None:
    record_run(tmp_path, _record(refuted_hypotheses=["the heuristic did not beat the exact solver"]))
    loaded = load_run_records(tmp_path)

    assert len(loaded) == 1
    assert loaded[0].project_name == "run-a"
    assert loaded[0].best_metric == 12.0
    assert loaded[0].refuted_hypotheses == ["the heuristic did not beat the exact solver"]
    assert loaded[0].timestamp, "record_run must stamp the record with a time"


def test_records_come_back_in_the_order_they_were_written(tmp_path: Path) -> None:
    record_run(tmp_path, _record(project_name="first"))
    record_run(tmp_path, _record(project_name="second"))
    assert [record.project_name for record in load_run_records(tmp_path)] == ["first", "second"]
    assert [record.project_name for record in load_run_records(tmp_path, limit=1)] == ["second"]


def test_an_absent_memory_file_is_an_empty_history_not_an_error(tmp_path: Path) -> None:
    assert load_run_records(tmp_path) == []
    assert build_prior_findings(tmp_path, "any goal").is_empty


def test_one_unreadable_line_does_not_cost_the_run_its_whole_memory(tmp_path: Path) -> None:
    """Losing a line of memory must never be the reason a run fails to start."""

    record_run(tmp_path, _record(project_name="good"))
    path = tmp_path / "research_memory" / "runs.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        handle.write("{ this is not json\n")
        handle.write(json.dumps(["a list, not a record"]) + "\n")
        handle.write(json.dumps({"best_metric": "not a number"}) + "\n")

    loaded = load_run_records(tmp_path)
    assert [record.project_name for record in loaded] == ["good"]


def test_append_memory_still_writes_the_older_entry_kinds(tmp_path: Path) -> None:
    path = append_memory(tmp_path, "experiments", {"project_name": "x"})
    assert path.name == "experiments.jsonl"
    assert json.loads(path.read_text(encoding="utf-8").strip())["project_name"] == "x"


# -- matching --------------------------------------------------------------


def test_similarity_ignores_the_words_every_goal_contains() -> None:
    """`the`, `study`, `method` are in every goal, so they cannot distinguish any."""

    assert goal_similarity("a study of the method", "a study of the approach") == 0.0
    assert goal_similarity("bilevel knapsack", "bilevel knapsack") == 1.0
    assert goal_similarity("bilevel knapsack", "") == 0.0
    assert goal_similarity("bilevel knapsack pricing", "knapsack pricing bilevel") == 1.0


def test_an_unrelated_prior_run_is_not_offered_as_context(tmp_path: Path) -> None:
    record_run(tmp_path, _record(project_name="related"))
    record_run(
        tmp_path,
        _record(project_name="unrelated", run_dir="/runs/b", research_goal="femtosecond laser groove width prediction"),
    )

    matches = find_related_runs(tmp_path, "bilevel knapsack scheduling with big-M linearisation")
    assert [record.project_name for record, _ in matches] == ["related"]


def test_the_run_currently_executing_does_not_match_itself(tmp_path: Path) -> None:
    record_run(tmp_path, _record(run_dir="/runs/current"))
    findings = build_prior_findings(tmp_path, _record().research_goal, exclude_run_dirs=["/runs/current"])
    assert findings.is_empty
    assert findings.total_runs_in_memory == 1


def test_the_best_prior_metric_respects_the_objective_direction(tmp_path: Path) -> None:
    record_run(tmp_path, _record(best_metric=12.0))
    record_run(tmp_path, _record(run_dir="/runs/b", best_metric=7.0))
    assert build_prior_findings(tmp_path, _record().research_goal).best_prior_metric() == 7.0

    other = tmp_path / "maximising"
    record_run(other, _record(best_metric=12.0, objective_direction="maximize"))
    record_run(other, _record(run_dir="/runs/b", best_metric=7.0, objective_direction="maximize"))
    assert build_prior_findings(other, _record().research_goal).best_prior_metric() == 12.0


# -- how it is phrased -----------------------------------------------------


def test_an_empty_history_says_so_instead_of_reading_like_a_clean_result(tmp_path: Path) -> None:
    findings = build_prior_findings(tmp_path, "a brand new goal")
    assert findings.to_prompt() == ""
    text = findings.to_markdown()
    assert "No prior run" in text
    assert "starts without carried-over evidence" in text


def test_a_prior_negative_is_carried_as_evidence_not_as_a_settled_fact(tmp_path: Path) -> None:
    record_run(
        tmp_path,
        _record(refuted_hypotheses=["the heuristic did not beat the exact solver"]),
    )
    findings = build_prior_findings(tmp_path, _record().research_goal)

    prompt = findings.to_prompt()
    assert "not supported there" in prompt
    assert "worth retesting, not worth assuming" in prompt
    assert "Do not treat any of the above as settled" in prompt

    markdown = findings.to_markdown()
    assert "tested and not supported there" in markdown
    assert "not a reason to skip one" in markdown


def test_the_brief_is_written_to_the_run_directory(tmp_path: Path) -> None:
    record_run(tmp_path, _record())
    findings = build_prior_findings(tmp_path, _record().research_goal)
    path = write_prior_findings(tmp_path / "run", findings)
    assert path.exists()
    assert (tmp_path / "run" / "prior_findings.json").exists()
    assert "run-a" in path.read_text(encoding="utf-8")


# -- summarising a finished journal ----------------------------------------


def _journal_with_mutations() -> ResearchJournal:
    journal = ResearchJournal()
    parent = ResearchNode(
        iteration=0, branch_index=0, plan="p", status="success",
        metric_name="objective", metric_value=10.0, work_dir="/w",
    )
    journal.append(parent)
    journal.append(
        ResearchNode(
            parent_id=parent.id, iteration=1, branch_index=0, plan="p", status="success",
            metric_name="objective", metric_value=4.0, work_dir="/w",
            metadata={"mutation": {"kind": "tighten_bounds"}},
        )
    )
    journal.append(
        ResearchNode(
            parent_id=parent.id, iteration=1, branch_index=1, plan="p", status="success",
            metric_name="objective", metric_value=14.0, work_dir="/w",
            metadata={"mutation": {"kind": "loosen_tolerance"}},
        )
    )
    journal.append(
        ResearchNode(
            parent_id=parent.id, iteration=1, branch_index=2, plan="p", status="failed",
            metric_name="objective", work_dir="/w", failure_summary="KeyError: 'objective'",
            metadata={"mutation": {"kind": "swap_solver"}},
        )
    )
    return journal


def test_a_mutation_is_credited_against_its_own_parent_not_the_run_best() -> None:
    """Otherwise every mutation on a strong branch looks good and every mutation
    on a weak one looks bad, whatever the change itself did."""

    summary = summarize_journal(_journal_with_mutations(), "objective", "minimize")

    assert summary["helpful_mutations"] == ["tighten_bounds"]
    assert set(summary["unhelpful_mutations"]) == {"loosen_tolerance", "swap_solver"}


def test_the_summary_counts_what_ran_and_what_broke() -> None:
    journal = _journal_with_mutations()
    summary = summarize_journal(journal, "objective", "minimize")

    assert summary["node_count"] == 4
    assert summary["successful_nodes"] == 3
    assert summary["failed_nodes"] == 1
    assert summary["best_metric"] == 4.0
    assert summary["verification_failures"] == [f"{journal.nodes[3].id}: KeyError: 'objective'"]


def test_a_run_that_produced_nothing_records_zero_rather_than_silence() -> None:
    summary = summarize_journal(ResearchJournal(), "objective", "minimize")
    assert summary["node_count"] == 0
    assert summary["best_metric"] is None
    assert summary["helpful_mutations"] == []
