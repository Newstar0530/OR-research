"""The repair backend that can rewrite anything, and therefore trusts nothing.

A fake LLM stands in for the provider so the loop itself is under test: what is
asked, what is accepted, what is rejected, and what is fed back when a rewrite
is refused. The point of these tests is that a bad rewrite never reaches disk.
"""

from pathlib import Path

import pytest

from src.core.code_editing_backend import (
    CompositeRepairBackend,
    LLMRepairBackend,
    WorkspaceRepairRequest,
)


BROKEN = '''"""An experiment with a bug."""
from pathlib import Path
import json
import pandas as pd

def main() -> None:
    out = Path(__file__).resolve().parent
    rows = [1, 2, 3]
    value = rows[99]
    pd.DataFrame([{"objective": value}]).to_csv(out / "results.csv", index=False)
    print("SUMMARY_JSON:" + json.dumps({"status": "success"}))

if __name__ == "__main__":
    main()
'''

FIXED = BROKEN.replace("rows[99]", "rows[-1]")

def _stderr(workspace: Path) -> str:
    """A traceback carrying this node's real absolute paths, as a run produces."""

    script = workspace / "experiment.py"
    return (
        "Traceback (most recent call last):\n"
        f'  File "{script}", line 12, in <module>\n'
        "    main()\n"
        f'  File "{script}", line 9, in main\n'
        "    value = rows[99]\n"
        "IndexError: list index out of range\n"
    )


class FakeLLM:
    """Returns canned responses in order and records what it was asked."""

    def __init__(self, *responses: str) -> None:
        self.responses = list(responses)
        self.prompts: list[str] = []

    def chat(self, system_prompt: str, user_prompt: str) -> str:
        self.prompts.append(user_prompt)
        return self.responses[min(len(self.prompts) - 1, len(self.responses) - 1)]


class ExplodingLLM:
    def chat(self, system_prompt: str, user_prompt: str) -> str:
        raise RuntimeError("provider is down")


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    (tmp_path / "experiment.py").write_text(BROKEN, encoding="utf-8")
    return tmp_path


def _request(workspace: Path, stderr: str | None = None) -> WorkspaceRepairRequest:
    return WorkspaceRepairRequest(
        work_dir=workspace,
        stderr=_stderr(workspace) if stderr is None else stderr,
        stdout="about to fail\n",
    )


# -- the happy path --------------------------------------------------------


def test_a_valid_rewrite_is_written_and_the_original_kept(workspace: Path) -> None:
    llm = FakeLLM(f"Here you go:\n```python\n{FIXED}```")
    result = LLMRepairBackend(llm).repair(_request(workspace))

    assert result.applied
    assert (workspace / "experiment.py").read_text(encoding="utf-8").strip() == FIXED.strip()
    assert (workspace / "experiment_before_repair_0.py").read_text(encoding="utf-8") == BROKEN
    assert any("not assumed correct, only bounded" in note for note in result.notes)


def test_the_model_is_shown_the_parsed_failure_and_the_whole_file(workspace: Path) -> None:
    llm = FakeLLM(f"```python\n{FIXED}```")
    LLMRepairBackend(llm).repair(_request(workspace))

    prompt = llm.prompts[0]
    assert "IndexError" in prompt
    assert "list index out of range" in prompt
    assert "value = rows[99]" in prompt, "the failing line must be quoted"
    assert "Current `experiment.py`" in prompt
    assert str(workspace) not in prompt, "absolute node paths must be shortened away"
    assert "Location: experiment.py, line 9" in prompt


def test_every_attempt_is_saved_for_a_human_to_read(workspace: Path) -> None:
    LLMRepairBackend(FakeLLM(f"```python\n{FIXED}```")).repair(_request(workspace))
    saved = list(workspace.glob("llm_repair_attempt_*.md"))
    assert len(saved) == 1
    assert "## Prompt" in saved[0].read_text(encoding="utf-8")


# -- a bad rewrite must never reach disk -----------------------------------


@pytest.mark.parametrize(
    "bad",
    [
        "def main(:\n    pass\n",                                   # does not parse
        'print("SUMMARY_JSON:{}")\n',                               # gutted
        BROKEN.replace("import json", "import socket\nimport json"),  # escapes the sandbox
        "I would suggest checking the index bounds.",               # not code at all
    ],
)
def test_a_rewrite_that_fails_the_checks_leaves_the_file_untouched(workspace: Path, bad: str) -> None:
    result = LLMRepairBackend(FakeLLM(bad), max_attempts=1).repair(_request(workspace))

    assert result.attempted
    assert not result.applied
    assert (workspace / "experiment.py").read_text(encoding="utf-8") == BROKEN
    assert not (workspace / "experiment_before_repair_0.py").exists()
    assert result.notes and "rejected" in result.notes[0]


def test_a_rejection_is_fed_back_as_the_next_instruction(workspace: Path) -> None:
    """Throwing the rejection away would buy the same bad rewrite twice."""

    llm = FakeLLM(BROKEN.replace("import json", "import socket\nimport json"), f"```python\n{FIXED}```")
    result = LLMRepairBackend(llm, max_attempts=2).repair(_request(workspace))

    assert result.applied
    assert len(llm.prompts) == 2
    assert "rejected" in llm.prompts[1]
    assert "socket" in llm.prompts[1]
    assert "attempt 2 of 2" in result.reason


def test_giving_up_says_how_many_attempts_were_spent_and_why(workspace: Path) -> None:
    llm = FakeLLM("not code")
    result = LLMRepairBackend(llm, max_attempts=3).repair(_request(workspace))

    assert len(llm.prompts) == 3
    assert not result.applied
    assert "3 attempt(s)" in result.reason
    assert "left unchanged" in result.reason
    assert len(result.notes) == 3


# -- refusing to ask -------------------------------------------------------


def test_a_failure_with_nothing_observable_is_not_handed_to_a_model(workspace: Path) -> None:
    """Asking a model to fix an unnamed failure invites an arbitrary rewrite."""

    llm = FakeLLM(f"```python\n{FIXED}```")
    request = WorkspaceRepairRequest(work_dir=workspace, stderr="", stdout="")
    result = LLMRepairBackend(llm).repair(request)

    assert not result.attempted
    assert llm.prompts == []
    assert "nothing to hand a model" in result.reason


def test_a_missing_script_is_reported_not_created(workspace: Path) -> None:
    (workspace / "experiment.py").unlink()
    result = LLMRepairBackend(FakeLLM("x")).repair(_request(workspace))
    assert not result.attempted
    assert "not found" in result.reason


def test_no_llm_client_is_a_reason_not_a_crash(workspace: Path) -> None:
    result = LLMRepairBackend(None).repair(_request(workspace))
    assert not result.attempted
    assert "No LLM client" in result.reason


def test_a_provider_outage_does_not_take_the_run_down(workspace: Path) -> None:
    result = LLMRepairBackend(ExplodingLLM()).repair(_request(workspace))
    assert result.attempted
    assert not result.applied
    assert "provider is down" in result.reason
    assert (workspace / "experiment.py").read_text(encoding="utf-8") == BROKEN


# -- dispatch --------------------------------------------------------------


def test_auto_falls_through_to_the_llm_when_no_rule_matches(workspace: Path) -> None:
    """The deterministic backend has no rule for IndexError; that is the whole point."""

    llm = FakeLLM(f"```python\n{FIXED}```")
    result = CompositeRepairBackend(mode="auto", llm=llm).repair(_request(workspace))

    assert result.applied
    assert result.backend == "llm"
    assert any("deterministic backend skipped" in note.lower() for note in result.notes)


def test_the_llm_is_not_consulted_in_deterministic_mode(workspace: Path) -> None:
    llm = FakeLLM(f"```python\n{FIXED}```")
    result = CompositeRepairBackend(mode="deterministic", llm=llm).repair(_request(workspace))

    assert llm.prompts == []
    assert result.backend == "deterministic"
    assert not result.applied


def test_disabled_stays_disabled_even_with_a_client(workspace: Path) -> None:
    llm = FakeLLM(f"```python\n{FIXED}```")
    result = CompositeRepairBackend(mode="disabled", llm=llm).repair(_request(workspace))
    assert llm.prompts == []
    assert result.backend == "disabled"


def test_auto_reports_every_backend_that_declined(workspace: Path) -> None:
    result = CompositeRepairBackend(mode="auto", llm=None).repair(_request(workspace))

    assert not result.attempted
    assert result.backend == "auto"
    joined = " ".join(result.notes).lower()
    assert "deterministic" in joined and "llm" in joined and "aider" in joined
