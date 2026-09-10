"""The controller must build a tree, not a chain with a tree-shaped report.

Before Layer 2 the controller kept one `parent_workspace` that moved to the
best node at the end of each iteration: every branch in an iteration mutated
the same parent, and a failed node was a dead end. These tests run the real
controller over a real subprocess and assert on the shape of what came out.
"""

import json
from pathlib import Path

from src.agent_system.state import ResearchState
from src.core.budget import ResearchBudget
from src.core.decision_engine import DecisionEngine
from src.core.experiment_search import ExperimentSearchController
from src.core.search_policy import SearchPolicyConfig
from src.execution.sandbox_runner import SandboxRunner
from src.llm_client import LLMClient
from src.schemas import AlgorithmPlan, ResearchIdea


WORKING_CODE = """
from pathlib import Path
import json
import pandas as pd
out = Path(__file__).resolve().parent
pd.DataFrame([
    {"instance_id": "i1", "method": "baseline", "objective": 10.0, "runtime_seconds": 0.1, "seed": 42},
    {"instance_id": "i1", "method": "proposed", "objective": 9.0, "runtime_seconds": 0.2, "seed": 42},
]).to_csv(out / "results.csv", index=False)
print("SUMMARY_JSON:" + json.dumps({"status": "success"}))
"""

BROKEN_CODE = """
from pathlib import Path
import json
import pandas as pd
rows = [1, 2, 3]
print("about to fail")
value = rows[99]
out = Path(__file__).resolve().parent
pd.DataFrame([
    {"instance_id": "i1", "method": "baseline", "objective": 10.0, "runtime_seconds": 0.1, "seed": 42},
    {"instance_id": "i1", "method": "proposed", "objective": float(value), "runtime_seconds": 0.2, "seed": 42},
]).to_csv(out / "results.csv", index=False)
print("SUMMARY_JSON:" + json.dumps({"status": "success"}))
"""

#: What a competent model would return: the smallest change that fixes the
#: IndexError, with every output the contract needs still in place.
REPAIRED_CODE = BROKEN_CODE.replace("rows[99]", "rows[-1]")


def _idea() -> ResearchIdea:
    return ResearchIdea(
        title="tree search test", problem_context="test", core_hypothesis="h",
        expected_contribution="c", proposed_model_type="generic",
        proposed_algorithm_type="generic", experimental_plan="run variants",
        interestingness_score=5, novelty_score=5, feasibility_score=8, risk_score=3,
        assumptions=[], required_data=[], expected_outputs=[],
    )


def _algorithm() -> AlgorithmPlan:
    return AlgorithmPlan(
        exact_solver_plan="baseline", heuristic_plan="proposed",
        baseline_methods=["baseline"], pseudocode="run", complexity_discussion="small",
        stopping_criteria="budget", required_packages=["pandas"],
        evaluation_metrics=["objective"],
    )


class RepairingLLM(LLMClient):
    """A mock client that answers repair requests with a real fix.

    Every other agent checks `use_mock` and never calls out, so this touches
    only the repair path -- which is the path under test.
    """

    def __init__(self) -> None:
        super().__init__(use_mock=True)
        object.__setattr__(self, "repair_calls", 0)

    def chat(self, system_prompt: str, user_prompt: str) -> str:
        if "repair generated Operations Research" in system_prompt:
            object.__setattr__(self, "repair_calls", self.repair_calls + 1)
            return f"```python\n{REPAIRED_CODE}```"
        return super().chat(system_prompt, user_prompt)


def _controller(
    policy: SearchPolicyConfig,
    iterations: int = 2,
    branches: int = 2,
    llm: LLMClient | None = None,
    code_editing_backend: str = "deterministic",
    max_repair_attempts: int = 1,
):
    return ExperimentSearchController(
        llm or LLMClient(use_mock=True),
        SandboxRunner(timeout_seconds=30),
        ResearchBudget(max_iterations=iterations, max_branches=branches, timeout_seconds=30),
        DecisionEngine("objective", "minimize"),
        search_policy=policy,
        code_editing_backend=code_editing_backend,
        max_repair_attempts=max_repair_attempts,
    )


def _state(tmp_path: Path) -> ResearchState:
    return ResearchState(
        project_name="x", research_goal="goal", domain_profile="optimization",
        run_dir=str(tmp_path),
    )


def test_the_search_opens_independent_roots_before_improving_any_of_them(tmp_path: Path) -> None:
    controller = _controller(SearchPolicyConfig(num_drafts=2, debug_probability=0.0, seed=0))
    journal = controller.run(tmp_path, _idea(), _algorithm(), WORKING_CODE, research_state=_state(tmp_path))

    roots = journal.roots()
    assert len(roots) == 2, "two draft slots must produce two independent roots"
    assert all(node.kind == "draft" for node in roots)
    assert all(node.parent_id is None for node in roots)


def test_sibling_branches_do_not_all_hang_off_the_same_parent(tmp_path: Path) -> None:
    """This is the exact behaviour the old single `parent_workspace` produced."""

    controller = _controller(SearchPolicyConfig(num_drafts=2, debug_probability=0.0, seed=0))
    journal = controller.run(tmp_path, _idea(), _algorithm(), WORKING_CODE, research_state=_state(tmp_path))

    improved = [node for node in journal.nodes if node.kind == "improve"]
    assert len(improved) == 2
    assert len({node.parent_id for node in improved}) == 2, "the two improve slots reused one parent"
    assert all(node.depth == 1 for node in improved)


def test_a_failed_node_is_parsed_so_the_policy_knows_what_broke(tmp_path: Path) -> None:
    controller = _controller(
        SearchPolicyConfig(num_drafts=2, debug_probability=1.0, seed=0), iterations=1
    )
    journal = controller.run(tmp_path, _idea(), _algorithm(), BROKEN_CODE, research_state=_state(tmp_path))

    failed = [node for node in journal.nodes if node.is_buggy]
    assert failed, "the broken script should have produced failed nodes"
    assert failed[0].exception_type == "IndexError"
    assert failed[0].failing_line is not None
    assert "IndexError" in failed[0].failure_summary


def test_a_failure_becomes_a_repair_branch_instead_of_a_dead_end(tmp_path: Path) -> None:
    controller = _controller(SearchPolicyConfig(num_drafts=2, debug_probability=1.0, seed=0))
    journal = controller.run(tmp_path, _idea(), _algorithm(), BROKEN_CODE, research_state=_state(tmp_path))

    debug_nodes = [node for node in journal.nodes if node.kind == "debug"]
    assert debug_nodes, "a failed leaf should have been picked up for repair"
    node = debug_nodes[0]
    assert node.debug_depth == 1
    assert journal.node_by_id(node.parent_id).is_buggy
    assert node.metadata["inherited_failure"]["exception_type"] == "IndexError"

    repairs = json.loads((tmp_path / "repair_trace.json").read_text(encoding="utf-8"))
    assert any(item.get("stage") == "inherited" for item in repairs)


def test_a_repair_the_backend_declined_closes_the_branch_without_re_running_it(tmp_path: Path) -> None:
    """Re-running byte-identical code buys the same failure for the price of a timeout."""

    controller = _controller(SearchPolicyConfig(num_drafts=2, debug_probability=1.0, seed=0))
    journal = controller.run(tmp_path, _idea(), _algorithm(), BROKEN_CODE, research_state=_state(tmp_path))

    declined = [node for node in journal.nodes if node.metadata.get("repair_exhausted")]
    assert declined, "the deterministic backend has no rule for IndexError, so it must decline"
    node = declined[0]
    assert node.execution is None, "the node must not have been executed after a declined repair"
    assert "No repair was applied" in node.analysis
    assert not (Path(node.work_dir) / "results.csv").exists()


def test_the_policy_trace_records_the_decisions_that_drove_the_run(tmp_path: Path) -> None:
    controller = _controller(SearchPolicyConfig(num_drafts=2, debug_probability=0.0, seed=0))
    controller.run(tmp_path, _idea(), _algorithm(), WORKING_CODE, research_state=_state(tmp_path))

    decisions = json.loads((tmp_path / "search_policy_trace.json").read_text(encoding="utf-8"))
    assert [item["kind"] for item in decisions] == ["draft", "draft", "improve", "improve"]
    assert all(item["reason"] for item in decisions)

    text = (tmp_path / "search_policy_trace.md").read_text(encoding="utf-8")
    assert "- draft: 2" in text
    assert "- improve: 2" in text


def test_the_expansion_queue_reports_what_was_expanded_not_what_might_be(tmp_path: Path) -> None:
    controller = _controller(SearchPolicyConfig(num_drafts=2, debug_probability=0.0, seed=0))
    journal = controller.run(tmp_path, _idea(), _algorithm(), WORKING_CODE, research_state=_state(tmp_path))

    queue = json.loads((tmp_path / "bfts_policy_queue.json").read_text(encoding="utf-8"))
    assert queue["selection_basis"] == "observed"
    expanded = {item["node_id"] for item in queue["queue"] if item["selected_for_expansion"]}
    assert expanded == {node.parent_id for node in journal.nodes if node.parent_id}


def test_the_node_repairs_itself_before_the_search_spends_a_branch_on_it(tmp_path: Path) -> None:
    """The cheapest repair is in place: same node, same slot, one more run.

    Only when a node's own attempts are exhausted is the failure worth a whole
    branch, so this path firing means the debug branch never has to."""

    llm = RepairingLLM()
    controller = _controller(
        SearchPolicyConfig(num_drafts=2, debug_probability=1.0, seed=0),
        llm=llm,
        code_editing_backend="llm",
        max_repair_attempts=1,
    )
    journal = controller.run(tmp_path, _idea(), _algorithm(), BROKEN_CODE, research_state=_state(tmp_path))

    assert llm.repair_calls > 0, "the repair backend was never consulted"
    assert all(node.status == "success" for node in journal.nodes)
    assert not [node for node in journal.nodes if node.kind == "debug"], (
        "a failure fixed in place should not also cost a debug branch"
    )
    repairs = json.loads((tmp_path / "repair_trace.json").read_text(encoding="utf-8"))
    assert all(item["stage"] == "in_node" and item["applied"] for item in repairs)


def test_a_debug_branch_repairs_a_failure_the_node_could_not_fix_itself(tmp_path: Path) -> None:
    """The whole chain: run fails, traceback is parsed, a debug branch is opened,
    the model rewrites the script, the guard accepts it, and the re-run succeeds.

    In-node repair is switched off so the failure survives its own node, which
    is the situation the debug branch exists for. Before this layer that failure
    was a dead end and the repair trace said no rule matched."""

    llm = RepairingLLM()
    controller = _controller(
        SearchPolicyConfig(num_drafts=2, debug_probability=1.0, seed=0),
        llm=llm,
        code_editing_backend="llm",
        max_repair_attempts=0,
    )
    journal = controller.run(tmp_path, _idea(), _algorithm(), BROKEN_CODE, research_state=_state(tmp_path))

    repaired = [node for node in journal.nodes if node.kind == "debug" and node.status == "success"]
    assert repaired, "a debug node should have run successfully after the rewrite"

    node = repaired[0]
    assert node.metric_value is not None
    assert node.debug_depth == 1
    assert journal.node_by_id(node.parent_id).is_buggy, "it must descend from the failure it fixed"
    assert (Path(node.work_dir) / "results.csv").exists()
    assert list(Path(node.work_dir).glob("experiment_before_repair_*.py")), "the original must be recoverable"
    assert "rows[-1]" in (Path(node.work_dir) / "experiment.py").read_text(encoding="utf-8")

    repairs = json.loads((tmp_path / "repair_trace.json").read_text(encoding="utf-8"))
    assert any(item["stage"] == "inherited" and item["applied"] for item in repairs)


def test_a_rewrite_that_fails_the_guard_never_reaches_the_workspace(tmp_path: Path) -> None:
    """A model that answers with prose leaves the node exactly as it was."""

    class UselessLLM(LLMClient):
        def chat(self, system_prompt: str, user_prompt: str) -> str:
            if "repair generated Operations Research" in system_prompt:
                return "You should check the index bounds before accessing the list."
            return super().chat(system_prompt, user_prompt)

    controller = _controller(
        SearchPolicyConfig(num_drafts=2, debug_probability=1.0, seed=0),
        llm=UselessLLM(use_mock=True),
        code_editing_backend="llm",
        max_repair_attempts=0,
    )
    journal = controller.run(tmp_path, _idea(), _algorithm(), BROKEN_CODE, research_state=_state(tmp_path))

    assert not [node for node in journal.nodes if node.kind == "debug" and node.status == "success"]
    declined = [node for node in journal.nodes if node.metadata.get("repair_exhausted")]
    assert declined, "the branch must be closed rather than re-run unchanged"
    assert "rows[99]" in (Path(declined[0].work_dir) / "experiment.py").read_text(encoding="utf-8")
