"""A mutation that changes nothing produced a node the search counted as explored.

`code.replace(source, target)` returns the same string when `source` is absent.
On any template that did not happen to contain `PROPOSED_EFFECT_LOW = 0.5`,
every replacement missed, the only difference was the docstring header the
refiner inserts, and the node became a byte-identical copy of its parent --
scored, ranked, and reported as a branch that had been explored. Nothing
noticed, because the refiner returned a bare string and no caller compared it
to what it was given.

These tests pin the two halves of the fix: the no-op is now visible, and a
model can apply the mutation's intent to a template the replacement keys know
nothing about.
"""

import json
from pathlib import Path

import pytest

from src.agents.experiment_refiner_agent import ExperimentRefinerAgent
from src.agents.mutation_agent import MutationAgent
from src.core.generated_code_guard import behavioural_lines, validate_mutation
from src.core.mutation import MutationSpec
from src.core.research_journal import ResearchJournal
from src.core.research_node import ResearchNode
from src.llm_client import LLMClient
from src.llm_errors import LLMCallError


#: A template with none of the constant names the replacements look for. This
#: is the normal case for every scaffold except `generic_or_template.py`.
FOREIGN_TEMPLATE = '''"""A scheduling experiment."""
from pathlib import Path
import json
import pandas as pd

N_JOBS = 12
SEED = 7

def main() -> None:
    out = Path(__file__).resolve().parent
    rows = [
        {"instance_id": "i1", "method": "baseline", "objective": 10.0, "seed": SEED},
        {"instance_id": "i1", "method": "proposed", "objective": 9.0, "seed": SEED},
    ]
    pd.DataFrame(rows).to_csv(out / "results.csv", index=False)
    print("SUMMARY_JSON:" + json.dumps({"status": "success"}))

if __name__ == "__main__":
    main()
'''

GENERIC_TEMPLATE = FOREIGN_TEMPLATE.replace(
    "N_JOBS = 12\nSEED = 7",
    "PROPOSED_EFFECT_LOW = 0.5\nPROPOSED_EFFECT_HIGH = 3.0\nSEED = 7",
)

STRESS = MutationSpec(
    kind="stress_test",
    description="Increase instance sizes and noise so the result has to survive harder cases.",
    replacements={"PROPOSED_EFFECT_LOW = 0.5": "PROPOSED_EFFECT_LOW = 0.1"},
)


class ScriptedLLM(LLMClient):
    def __init__(self, *responses: str) -> None:
        super().__init__(use_mock=False)
        object.__setattr__(self, "responses", list(responses))
        object.__setattr__(self, "prompts", [])

    def chat(self, system_prompt: str, user_prompt: str, stage: str = "") -> str:
        self.prompts.append(user_prompt)
        return self.responses[min(len(self.prompts) - 1, len(self.responses) - 1)]


class FailingLLM(LLMClient):
    def __init__(self, error: LLMCallError) -> None:
        super().__init__(use_mock=False)
        object.__setattr__(self, "error", error)

    def chat(self, system_prompt: str, user_prompt: str, stage: str = "") -> str:
        raise self.error


# -- the check that makes the no-op visible --------------------------------


def test_a_header_only_difference_is_not_a_variant() -> None:
    """The refiner always inserts a header, so this is the exact failure mode."""

    header_only = FOREIGN_TEMPLATE.replace(
        '"""A scheduling experiment."""', '"""Autonomous variant: iteration=1, branch=0."""'
    )
    verdict = validate_mutation(header_only, FOREIGN_TEMPLATE)

    assert not verdict.ok
    assert any("identical to its parent" in item for item in verdict.violations)
    assert any("re-measure the parent" in item for item in verdict.violations)


def test_comments_and_blank_lines_do_not_count_as_behaviour() -> None:
    cosmetic = FOREIGN_TEMPLATE.replace("N_JOBS = 12", "\n# tuned below\nN_JOBS = 12")
    assert behavioural_lines(cosmetic) == behavioural_lines(FOREIGN_TEMPLATE)
    assert not validate_mutation(cosmetic, FOREIGN_TEMPLATE).ok


def test_a_real_change_passes() -> None:
    changed = FOREIGN_TEMPLATE.replace("N_JOBS = 12", "N_JOBS = 40")
    assert validate_mutation(changed, FOREIGN_TEMPLATE).ok


def test_a_variant_still_may_not_break_the_contract_or_escape_the_sandbox() -> None:
    """It is a variant of a working experiment, not a licence to rewrite it."""

    no_results = FOREIGN_TEMPLATE.replace("results.csv", "elsewhere.csv")
    assert not validate_mutation(no_results, FOREIGN_TEMPLATE).ok

    networked = FOREIGN_TEMPLATE.replace("import json", "import socket\nimport json")
    assert not validate_mutation(networked, FOREIGN_TEMPLATE).ok


# -- the string path, and its honest failure -------------------------------


def test_string_replacement_on_a_foreign_template_reports_no_effect() -> None:
    """The defect, made visible: this node would have been a copy of its parent."""

    outcome = ExperimentRefinerAgent(LLMClient(use_mock=True)).run(
        FOREIGN_TEMPLATE, "Adversarial stress test; effect_range=0.1,2.0", 1, 0, STRESS
    )

    assert outcome.method == "none"
    assert outcome.applied is False
    assert outcome.is_effective is False
    assert "PROPOSED_EFFECT_LOW = 0.5" in outcome.unmatched_replacements
    assert any("matched nothing in this template" in note for note in outcome.notes)
    assert any("re-measure the parent" in note for note in outcome.notes)
    assert any("a language model can apply the mutation's intent" in note for note in outcome.notes)


def test_string_replacement_on_the_template_it_was_written_for_works() -> None:
    outcome = ExperimentRefinerAgent(LLMClient(use_mock=True)).run(
        GENERIC_TEMPLATE, "Adversarial stress test; effect_range=0.1,2.0", 1, 0, STRESS
    )

    assert outcome.method == "string_replacement"
    assert outcome.applied
    assert "PROPOSED_EFFECT_LOW = 0.1" in outcome.code
    assert outcome.unmatched_replacements == []


def test_the_seed_changes_so_a_variant_is_reproducible_and_distinct() -> None:
    code = 'import numpy as np\nrng = np.random.default_rng(42)\nresults.csv\nSUMMARY_JSON\n'
    outcome = ExperimentRefinerAgent(LLMClient(use_mock=True)).run(code, "plan", 2, 1, None)
    assert "default_rng(243)" in outcome.code  # 42 + 2*100 + 1
    assert outcome.applied


def test_the_summary_says_which_path_was_taken() -> None:
    effective = ExperimentRefinerAgent(LLMClient(use_mock=True)).run(
        GENERIC_TEMPLATE, "stress; effect_range=0.1,2.0", 1, 0, STRESS
    )
    inert = ExperimentRefinerAgent(LLMClient(use_mock=True)).run(
        FOREIGN_TEMPLATE, "stress", 1, 0, STRESS
    )
    assert "replacement(s) applied" in effective.summary()
    assert inert.summary().startswith("no effect")


# -- the model path ---------------------------------------------------------


def test_a_model_applies_the_intent_to_a_template_the_keys_do_not_fit() -> None:
    """This is what the string path cannot do at all."""

    rewritten = FOREIGN_TEMPLATE.replace("N_JOBS = 12", "N_JOBS = 40")
    llm = ScriptedLLM(f"```python\n{rewritten}```")

    outcome = ExperimentRefinerAgent(llm).run(FOREIGN_TEMPLATE, "stress test", 3, 1, STRESS)

    assert outcome.method == "llm"
    assert outcome.applied
    assert "N_JOBS = 40" in outcome.code
    # The header is inserted after the script's own module docstring, so the
    # original docstring survives rather than being displaced.
    assert '"""Autonomous variant: iteration=3, branch=1' in outcome.code
    assert outcome.code.lstrip().startswith('"""A scheduling experiment."""')


def test_the_prompt_carries_the_intent_and_the_reference_edits() -> None:
    rewritten = FOREIGN_TEMPLATE.replace("N_JOBS = 12", "N_JOBS = 40")
    llm = ScriptedLLM(f"```python\n{rewritten}```")
    ExperimentRefinerAgent(llm).run(FOREIGN_TEMPLATE, "stress test", 3, 1, STRESS)

    prompt = llm.prompts[0]
    assert STRESS.description in prompt
    assert "PROPOSED_EFFECT_LOW = 0.5" in prompt
    assert "even if these exact names do not appear" in prompt
    assert "Use 343 as the random seed" in prompt  # 42 + 3*100 + 1
    assert "weakening the baseline or the verification" in prompt, (
        "a mutation that improves the result by measuring less is a defect, and the model must be told"
    )


def test_a_rewrite_identical_to_the_parent_is_rejected_and_fed_back() -> None:
    llm = ScriptedLLM(
        f"```python\n{FOREIGN_TEMPLATE}```",
        f"```python\n{FOREIGN_TEMPLATE.replace('N_JOBS = 12', 'N_JOBS = 40')}```",
    )
    outcome = ExperimentRefinerAgent(llm).run(FOREIGN_TEMPLATE, "stress", 1, 0, STRESS)

    assert outcome.method == "llm"
    assert len(llm.prompts) == 2
    assert "rejected" in llm.prompts[1]
    assert "identical to its parent" in llm.prompts[1]


def test_a_model_that_only_gutted_the_script_falls_back_to_the_string_path() -> None:
    llm = ScriptedLLM('```python\nprint("SUMMARY_JSON:{}")\nresults.csv\n```')
    outcome = ExperimentRefinerAgent(llm).run(GENERIC_TEMPLATE, "stress; effect_range=0.1,2.0", 1, 0, STRESS)

    assert outcome.method == "string_replacement", "a rejected rewrite must not be shipped"
    assert "PROPOSED_EFFECT_LOW = 0.1" in outcome.code
    assert len(llm.prompts) == 2


def test_a_transient_model_failure_falls_back_rather_than_failing_the_node() -> None:
    outcome = ExperimentRefinerAgent(FailingLLM(LLMCallError("rate_limit", "slow down", 429))).run(
        GENERIC_TEMPLATE, "stress; effect_range=0.1,2.0", 1, 0, STRESS
    )
    assert outcome.method == "string_replacement"


def test_a_permanent_model_failure_is_not_swallowed() -> None:
    with pytest.raises(LLMCallError):
        ExperimentRefinerAgent(FailingLLM(LLMCallError("auth", "bad key", 401))).run(
            GENERIC_TEMPLATE, "stress", 1, 0, STRESS
        )


# -- the mutation agent reads its own history ------------------------------


def _node(kind: str, *, success: bool, method: str = "string_replacement") -> ResearchNode:
    return ResearchNode(
        iteration=0, branch_index=0, plan="p",
        status="success" if success else "failed",
        metric_name="objective", metric_value=1.0 if success else None,
        metadata={"mutation": {"kind": kind}, "mutation_outcome": {"method": method}},
    )


def test_a_mutation_kind_that_changed_nothing_is_spent() -> None:
    """Inert on this template. Choosing it again buys the same no-op."""

    journal = ResearchJournal()
    journal.append(_node("stress_test", success=True, method="none"))
    assert "stress_test" in MutationAgent.spent_kinds(journal)


def test_a_mutation_kind_whose_every_node_failed_is_spent() -> None:
    journal = ResearchJournal()
    journal.append(_node("component_ablation", success=False))
    journal.append(_node("component_ablation", success=False))
    assert "component_ablation" in MutationAgent.spent_kinds(journal)


def test_a_kind_that_worked_at_least_once_is_not_spent() -> None:
    journal = ResearchJournal()
    journal.append(_node("replication_check", success=False))
    journal.append(_node("replication_check", success=True))
    assert "replication_check" not in MutationAgent.spent_kinds(journal)


def test_an_empty_journal_spends_nothing() -> None:
    assert MutationAgent.spent_kinds(ResearchJournal()) == set()


def test_a_spent_kind_is_still_offered_when_it_is_the_only_option_but_says_so() -> None:
    """Skipping it entirely would leave the slot with no mutation at all."""

    journal = ResearchJournal()
    journal.append(_node("stress_test", success=True, method="none"))
    mutation = MutationAgent(LLMClient(use_mock=True)).run(
        journal, 1, 0, "Adversarial stress test", domain_profile="binary_program"
    )
    assert "already been spent" in str(mutation.metadata.get("note", ""))


# -- through the real controller -------------------------------------------


def test_the_trace_says_how_many_mutations_really_changed_anything(tmp_path: Path) -> None:
    """The number worth reading first, and the one that did not exist before."""

    from src.agent_system.state import ResearchState
    from src.core.budget import ResearchBudget
    from src.core.decision_engine import DecisionEngine
    from src.core.experiment_search import ExperimentSearchController
    from src.core.search_policy import SearchPolicyConfig
    from src.execution.sandbox_runner import SandboxRunner
    from src.schemas import AlgorithmPlan, ResearchIdea

    code = '''
from pathlib import Path
import json
import pandas as pd
N_JOBS = 12
out = Path(__file__).resolve().parent
pd.DataFrame([
    {"instance_id": "i1", "method": "baseline", "objective": 10.0, "runtime_seconds": 0.1, "seed": 42},
    {"instance_id": "i1", "method": "proposed", "objective": 9.0, "runtime_seconds": 0.2, "seed": 42},
]).to_csv(out / "results.csv", index=False)
print("SUMMARY_JSON:" + json.dumps({"status": "success"}))
'''
    idea = ResearchIdea(
        title="t", problem_context="p", core_hypothesis="h", expected_contribution="c",
        proposed_model_type="g", proposed_algorithm_type="g", experimental_plan="e",
        interestingness_score=5, novelty_score=5, feasibility_score=8, risk_score=3,
        assumptions=[], required_data=[], expected_outputs=[],
    )
    algorithm = AlgorithmPlan(
        exact_solver_plan="b", heuristic_plan="p", baseline_methods=["baseline"],
        pseudocode="r", complexity_discussion="s", stopping_criteria="b",
        required_packages=["pandas"], evaluation_metrics=["objective"],
    )
    controller = ExperimentSearchController(
        LLMClient(use_mock=True),
        SandboxRunner(timeout_seconds=30),
        ResearchBudget(max_iterations=1, max_branches=2, timeout_seconds=30),
        DecisionEngine("objective", "minimize"),
        search_policy=SearchPolicyConfig(num_drafts=2, debug_probability=0.0, seed=0),
    )
    journal = controller.run(
        tmp_path, idea, algorithm, code,
        research_state=ResearchState(
            project_name="x", research_goal="g", domain_profile="optimization", run_dir=str(tmp_path)
        ),
    )

    trace = (tmp_path / "mutation_trace.md").read_text(encoding="utf-8")
    assert "actually changed the experiment" in trace
    assert "re-measures the parent rather than exploring anything" in trace

    entries = json.loads((tmp_path / "mutation_trace.json").read_text(encoding="utf-8"))
    assert entries, "mutations must be recorded"
    assert all("outcome" in entry for entry in entries)

    # This template exposes no effect knobs, so on the string path the mutation
    # is inert -- and every node must say so rather than looking explored.
    inert = [node for node in journal.nodes if node.metadata.get("mutation_had_no_effect")]
    assert inert, "an inert mutation must be marked on the node"
    assert all(node.metadata["mutation_outcome"]["method"] == "none" for node in inert)
