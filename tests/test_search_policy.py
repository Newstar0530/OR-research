"""The search policy has to make real decisions, not describe them afterwards.

Every test here asserts on a choice the policy made before the node existed:
which parent, which kind, and why. A test that only checked the trace file
would pass just as happily against the old report-after-the-fact behaviour.
"""

from pathlib import Path

import pytest

from src.core.research_journal import ResearchJournal
from src.core.research_node import ResearchNode
from src.core.search_policy import SearchPolicy, SearchPolicyConfig


def _node(
    journal: ResearchJournal,
    *,
    parent: ResearchNode | None = None,
    status: str = "success",
    metric: float | None = 10.0,
    kind: str = "improve",
    debug_depth: int = 0,
    exception_type: str | None = None,
    work_dir: str = "/tmp/node",
    maximize: bool = False,
    **metadata,
) -> ResearchNode:
    node = ResearchNode(
        parent_id=parent.id if parent else None,
        iteration=len(journal.nodes),
        branch_index=0,
        plan="p",
        status=status,  # type: ignore[arg-type]
        metric_name="objective",
        metric_value=metric,
        maximize=maximize,
        kind=kind,  # type: ignore[arg-type]
        debug_depth=debug_depth,
        exception_type=exception_type,
        work_dir=work_dir,
        metadata=dict(metadata),
    )
    journal.append(node)
    return node


# -- journal tree queries --------------------------------------------------


def test_the_journal_can_answer_structural_questions_about_its_forest() -> None:
    journal = ResearchJournal()
    root_a = _node(journal)
    child = _node(journal, parent=root_a)
    root_b = _node(journal)

    assert [n.id for n in journal.roots()] == [root_a.id, root_b.id]
    assert [n.id for n in journal.children_of(root_a.id)] == [child.id]
    assert {n.id for n in journal.leaves()} == {child.id, root_b.id}
    assert journal.depth_of(child.id) == 1
    assert journal.depth_of(root_a.id) == 0
    assert journal.node_by_id("nope") is None


def test_a_failure_that_has_already_been_expanded_is_no_longer_a_buggy_leaf() -> None:
    """Repairing it again would fork a second attempt from the same broken state."""

    journal = ResearchJournal()
    failed = _node(journal, status="failed", metric=None, exception_type="ValueError")
    assert [n.id for n in journal.buggy_leaves()] == [failed.id]

    _node(journal, parent=failed, kind="debug", status="failed", metric=None, debug_depth=1)
    assert failed.id not in {n.id for n in journal.buggy_leaves()}


# -- draft -----------------------------------------------------------------


def test_the_policy_drafts_until_the_configured_number_of_roots_exists() -> None:
    journal = ResearchJournal()
    policy = SearchPolicy(SearchPolicyConfig(num_drafts=3))

    first = policy.decide(journal, 0, 0, root_work_dir="/base")
    assert first.kind == "draft"
    assert first.parent_id is None
    assert first.parent_work_dir == "/base"

    _node(journal)
    _node(journal)
    assert policy.decide(journal, 0, 1).kind == "draft"

    _node(journal)
    assert policy.decide(journal, 0, 2).kind != "draft"


def test_a_child_does_not_count_towards_the_draft_quota() -> None:
    """Two roots means two independent starts, not one start with a child."""

    journal = ResearchJournal()
    root = _node(journal)
    _node(journal, parent=root)
    policy = SearchPolicy(SearchPolicyConfig(num_drafts=2))
    assert policy.decide(journal, 1, 0).kind == "draft"


# -- improve ---------------------------------------------------------------


def test_improve_picks_the_best_node_for_a_minimised_objective() -> None:
    journal = ResearchJournal()
    _node(journal, metric=10.0)
    good = _node(journal, metric=2.0)
    policy = SearchPolicy(SearchPolicyConfig(num_drafts=2, debug_probability=0.0), maximize=False)

    decision = policy.decide(journal, 1, 0)
    assert decision.kind == "improve"
    assert decision.parent_id == good.id
    assert decision.depth == good.depth + 1


def test_improve_follows_the_objective_direction() -> None:
    journal = ResearchJournal()
    low = _node(journal, metric=2.0, maximize=True)
    high = _node(journal, metric=10.0, maximize=True)
    maximising = SearchPolicy(SearchPolicyConfig(num_drafts=2, debug_probability=0.0), maximize=True)
    minimising = SearchPolicy(SearchPolicyConfig(num_drafts=2, debug_probability=0.0), maximize=False)

    assert maximising.decide(journal, 1, 0).parent_id == high.id
    assert minimising.decide(journal, 1, 0).parent_id == low.id


def test_an_already_expanded_node_loses_to_an_equally_good_untried_one() -> None:
    """The exploration bonus is what stops the search collapsing into a chain."""

    journal = ResearchJournal()
    expanded = _node(journal, metric=5.0)
    _node(journal, parent=expanded, metric=9.0)  # a child, so `expanded` has been tried
    untried = _node(journal, metric=5.0)

    policy = SearchPolicy(SearchPolicyConfig(num_drafts=2, debug_probability=0.0))
    assert policy.decide(journal, 1, 0).parent_id == untried.id


def test_sibling_slots_in_one_iteration_expand_different_parents() -> None:
    journal = ResearchJournal()
    best = _node(journal, metric=1.0)
    second = _node(journal, metric=2.0)
    policy = SearchPolicy(SearchPolicyConfig(num_drafts=2, debug_probability=0.0))

    reserved: set[str] = set()
    first = policy.decide(journal, 1, 0, reserved_parent_ids=reserved)
    reserved.add(first.parent_id)
    later = policy.decide(journal, 1, 1, reserved_parent_ids=reserved)

    assert first.parent_id == best.id
    assert later.parent_id == second.id


def test_a_slot_is_never_wasted_just_because_every_parent_is_reserved() -> None:
    """Expanding one parent twice beats spending the slot on nothing."""

    journal = ResearchJournal()
    only = _node(journal, metric=1.0)
    _node(journal, metric=2.0)
    policy = SearchPolicy(SearchPolicyConfig(num_drafts=2, debug_probability=0.0))

    decision = policy.decide(journal, 1, 2, reserved_parent_ids={n.id for n in journal.nodes})
    assert decision.kind == "draft"
    assert "fresh root" in decision.reason
    assert only.id in {item.node_id for item in decision.considered}


# -- debug -----------------------------------------------------------------


def test_a_failed_leaf_is_chosen_for_repair_when_the_coin_always_says_debug() -> None:
    journal = ResearchJournal()
    _node(journal, metric=1.0)
    broken = _node(journal, status="failed", metric=None, exception_type="KeyError")
    policy = SearchPolicy(SearchPolicyConfig(num_drafts=2, debug_probability=1.0))

    decision = policy.decide(journal, 1, 0)
    assert decision.kind == "debug"
    assert decision.parent_id == broken.id
    assert decision.debug_depth == 1
    assert "KeyError" in decision.reason


def test_repair_stops_at_the_debug_depth_limit() -> None:
    journal = ResearchJournal()
    good = _node(journal, metric=1.0)
    _node(journal, status="failed", metric=None, debug_depth=2, exception_type="KeyError")
    policy = SearchPolicy(SearchPolicyConfig(num_drafts=2, max_debug_depth=2, debug_probability=1.0))

    decision = policy.decide(journal, 1, 0)
    assert decision.kind == "improve"
    assert decision.parent_id == good.id


def test_a_branch_the_repair_backend_refused_is_never_offered_again() -> None:
    """Otherwise the same refusal is bought once per remaining debug slot."""

    journal = ResearchJournal()
    good = _node(journal, metric=1.0)
    _node(
        journal,
        status="contract_failed",
        metric=None,
        exception_type="IndexError",
        repair_exhausted=True,
    )
    policy = SearchPolicy(SearchPolicyConfig(num_drafts=2, debug_probability=1.0))

    decision = policy.decide(journal, 1, 0)
    assert decision.kind == "improve"
    assert decision.parent_id == good.id
    note = next(item.note for item in decision.considered if item.status == "contract_failed")
    assert "declined" in note


def test_a_failure_with_a_named_exception_is_repaired_before_an_unparsed_one() -> None:
    journal = ResearchJournal()
    _node(journal, metric=1.0)
    _node(journal, status="timeout", metric=None, exception_type=None)
    named = _node(journal, status="failed", metric=None, exception_type="ZeroDivisionError")
    policy = SearchPolicy(SearchPolicyConfig(num_drafts=2, debug_probability=1.0))

    assert policy.decide(journal, 1, 0).parent_id == named.id


def test_repair_happens_even_against_the_coin_when_nothing_can_be_improved() -> None:
    journal = ResearchJournal()
    _node(journal, status="failed", metric=None, exception_type="KeyError")
    _node(journal, status="failed", metric=None, exception_type="KeyError")
    policy = SearchPolicy(SearchPolicyConfig(num_drafts=2, debug_probability=0.0))

    decision = policy.decide(journal, 1, 0)
    assert decision.kind == "debug"
    assert "no successful node" in decision.reason


def test_a_node_with_no_workspace_cannot_be_repaired() -> None:
    """There is nothing on disk to copy, so the branch is not a repair candidate."""

    journal = ResearchJournal()
    good = _node(journal, metric=1.0)
    _node(journal, status="failed", metric=None, exception_type="KeyError", work_dir="")
    policy = SearchPolicy(SearchPolicyConfig(num_drafts=2, debug_probability=1.0))

    assert policy.decide(journal, 1, 0).parent_id == good.id


# -- determinism and configuration -----------------------------------------


def test_the_same_seed_replays_the_same_sequence_of_decisions() -> None:
    def sequence() -> list[str]:
        journal = ResearchJournal()
        _node(journal, metric=1.0)
        _node(journal, status="failed", metric=None, exception_type="KeyError")
        policy = SearchPolicy(SearchPolicyConfig(num_drafts=2, debug_probability=0.5, seed=1234))
        return [policy.decide(journal, 1, index).kind for index in range(12)]

    assert sequence() == sequence()


def test_different_seeds_can_diverge_so_the_coin_is_really_being_flipped() -> None:
    def sequence(seed: int) -> list[str]:
        journal = ResearchJournal()
        _node(journal, metric=1.0)
        _node(journal, status="failed", metric=None, exception_type="KeyError")
        policy = SearchPolicy(SearchPolicyConfig(num_drafts=2, debug_probability=0.5, seed=seed))
        return [policy.decide(journal, 1, index).kind for index in range(20)]

    assert len({tuple(sequence(seed)) for seed in range(6)}) > 1


@pytest.mark.parametrize(
    "config",
    [
        SearchPolicyConfig(num_drafts=0),
        SearchPolicyConfig(max_debug_depth=-1),
        SearchPolicyConfig(debug_probability=1.5),
    ],
)
def test_an_impossible_policy_configuration_is_rejected_rather_than_normalised(config) -> None:
    with pytest.raises(ValueError):
        config.validated()


def test_the_trace_reports_the_decisions_that_were_actually_made() -> None:
    journal = ResearchJournal()
    policy = SearchPolicy(SearchPolicyConfig(num_drafts=1, debug_probability=0.0))
    policy.decide(journal, 0, 0, root_work_dir=Path("/base"))
    _node(journal, metric=3.0)
    policy.decide(journal, 1, 0)

    trace = policy.trace_markdown()
    assert "- draft: 1" in trace
    assert "- improve: 1" in trace
    assert "best-first" in trace
