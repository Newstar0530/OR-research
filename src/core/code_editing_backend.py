from __future__ import annotations

import ast
import shutil
import subprocess
from pathlib import Path

from pydantic import BaseModel, Field

from src.core.generated_code_guard import (
    GuardVerdict,
    extract_code,
    validate_generated_experiment,
)
from src.execution.traceback_parser import RepairContext, build_repair_context


class WorkspaceRepairRequest(BaseModel):
    work_dir: Path
    stderr: str = ""
    stdout: str = ""
    attempt: int = 0
    script_name: str = "experiment.py"
    #: The parsed failure. Built from stderr when absent, so callers that do not
    #: supply one still get structured matching instead of substring guessing.
    context: RepairContext | None = None

    def resolved_context(self) -> RepairContext:
        if self.context is not None:
            return self.context
        return build_repair_context(
            self.stderr,
            self.work_dir,
            script_name=self.script_name,
            stdout=self.stdout,
            attempt=self.attempt,
        )


class WorkspaceRepairResult(BaseModel):
    attempted: bool = False
    applied: bool = False
    backend: str = "none"
    reason: str = ""
    notes: list[str] = Field(default_factory=list)
    command: list[str] = Field(default_factory=list)
    stdout_preview: str = ""
    stderr_preview: str = ""


class DeterministicRepairBackend:
    name = "deterministic"

    def repair(self, request: WorkspaceRepairRequest) -> WorkspaceRepairResult:
        script = request.work_dir / request.script_name
        if not script.exists():
            return WorkspaceRepairResult(attempted=False, backend=self.name, reason=f"{request.script_name} not found.")
        context = request.resolved_context()
        message = f"{context.exception_type or ''}: {context.exception_message or ''}"
        if (
            context.exception_type == "SyntaxError"
            and "from __future__ imports must occur at the beginning" in message
        ) or "from __future__ imports must occur at the beginning" in request.stderr:
            text = script.read_text(encoding="utf-8")
            repaired = _move_future_imports_after_module_docstring(text)
            try:
                ast.parse(repaired)
            except SyntaxError as exc:
                return WorkspaceRepairResult(
                    attempted=True,
                    applied=False,
                    backend=self.name,
                    reason=f"Repair produced invalid Python: {exc}",
                )
            script.write_text(repaired, encoding="utf-8")
            return WorkspaceRepairResult(
                attempted=True,
                applied=True,
                backend=self.name,
                reason="Moved future imports before autonomous headers.",
                notes=["This is a bounded deterministic repair, not an unrestricted code rewrite."],
            )
        return WorkspaceRepairResult(
            attempted=False,
            backend=self.name,
            reason=(
                f"No deterministic repair rule matches `{context.exception_type or 'an unparsed failure'}`."
                " A bounded string rewrite cannot fix this class of error; it needs a code-editing"
                " backend that can reason about the program."
            ),
            notes=[context.filtered_traceback] if context.filtered_traceback else [],
        )


class AiderRepairBackend:
    name = "aider"

    def __init__(
        self,
        command: str = "aider",
        model: str | None = None,
        timeout_seconds: int = 120,
    ) -> None:
        self.command = command
        self.model = model
        self.timeout_seconds = timeout_seconds

    def repair(self, request: WorkspaceRepairRequest) -> WorkspaceRepairResult:
        script = request.work_dir / request.script_name
        if not script.exists():
            return WorkspaceRepairResult(attempted=False, backend=self.name, reason=f"{request.script_name} not found.")
        executable = shutil.which(self.command)
        if not executable:
            return WorkspaceRepairResult(attempted=False, backend=self.name, reason=f"Aider command `{self.command}` was not found on PATH.")

        prompt_path = request.work_dir / f"aider_repair_prompt_attempt_{request.attempt}.md"
        prompt_path.write_text(
            _aider_repair_prompt(script, request.resolved_context()), encoding="utf-8"
        )
        cmd = [
            executable,
            "--yes-always",
            "--no-auto-commits",
            "--message-file",
            prompt_path.name,
            script.name,
        ]
        if self.model:
            cmd[1:1] = ["--model", self.model]
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(request.work_dir),
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                shell=False,
            )
        except subprocess.TimeoutExpired as exc:
            return WorkspaceRepairResult(
                attempted=True,
                applied=False,
                backend=self.name,
                reason=f"Aider timed out after {self.timeout_seconds} seconds.",
                command=cmd,
                stdout_preview=_preview(exc.stdout or ""),
                stderr_preview=_preview(exc.stderr or ""),
            )

        if proc.returncode != 0:
            return WorkspaceRepairResult(
                attempted=True,
                applied=False,
                backend=self.name,
                reason=f"Aider exited with return code {proc.returncode}.",
                command=cmd,
                stdout_preview=_preview(proc.stdout),
                stderr_preview=_preview(proc.stderr),
            )
        try:
            ast.parse(script.read_text(encoding="utf-8"))
        except SyntaxError as exc:
            return WorkspaceRepairResult(
                attempted=True,
                applied=False,
                backend=self.name,
                reason=f"Aider modified the script but Python syntax is still invalid: {exc}",
                command=cmd,
                stdout_preview=_preview(proc.stdout),
                stderr_preview=_preview(proc.stderr),
            )
        return WorkspaceRepairResult(
            attempted=True,
            applied=True,
            backend=self.name,
            reason="Aider completed and the edited experiment.py parses as Python.",
            command=cmd,
            stdout_preview=_preview(proc.stdout),
            stderr_preview=_preview(proc.stderr),
            notes=[
                "Aider was constrained to the node workspace and experiment.py.",
                f"Repair prompt saved to {prompt_path.name}.",
            ],
        )



REPAIR_SYSTEM_PROMPT = (
    "You repair generated Operations Research experiment scripts. You return one complete "
    "Python file and nothing else: no explanation, no diff, no partial snippet. You change the "
    "smallest thing that fixes the reported error and you preserve the script's outputs."
)


class LLMRepairBackend:
    """Ask a model to rewrite the script, then refuse to trust it.

    This is the backend that can fix a failure nobody wrote a rule for, which
    also makes it the one that can quietly destroy the experiment. So the model
    never writes to disk directly: its output goes through
    `validate_generated_experiment` first, and a rejection is fed back as the
    next instruction rather than thrown away. A model that cannot satisfy the
    checks in `max_attempts` tries leaves the workspace untouched.

    The original script and every attempt are saved beside the workspace, so a
    repair that made things worse is recoverable and readable afterwards.
    """

    name = "llm"

    def __init__(self, llm, max_attempts: int = 2, min_retained_fraction: float = 0.5) -> None:
        self.llm = llm
        self.max_attempts = max(1, max_attempts)
        self.min_retained_fraction = min_retained_fraction

    def repair(self, request: WorkspaceRepairRequest) -> WorkspaceRepairResult:
        script = request.work_dir / request.script_name
        if not script.exists():
            return WorkspaceRepairResult(attempted=False, backend=self.name, reason=f"{request.script_name} not found.")
        if self.llm is None:
            return WorkspaceRepairResult(
                attempted=False,
                backend=self.name,
                reason="No LLM client was supplied to the repair backend.",
            )
        context = request.resolved_context()
        if not (context.exception_type or context.filtered_traceback or context.stdout_tail):
            # Nothing observable went wrong. Asking a model to fix an unnamed
            # failure invites it to rewrite whatever it feels like.
            return WorkspaceRepairResult(
                attempted=False,
                backend=self.name,
                reason=(
                    "The failure produced no traceback, no exception and no output, so there is "
                    "nothing to hand a model. Check the runner's timeout and stderr capture."
                ),
            )

        original = script.read_text(encoding="utf-8")
        notes: list[str] = []
        feedback = ""
        for attempt in range(self.max_attempts):
            prompt = _llm_repair_prompt(request.script_name, original, context, feedback)
            try:
                response = self.llm.chat(REPAIR_SYSTEM_PROMPT, prompt)
            except Exception as exc:  # a provider outage must not kill the run
                return WorkspaceRepairResult(
                    attempted=True,
                    applied=False,
                    backend=self.name,
                    reason=f"The LLM call failed: {exc}",
                    notes=notes,
                )
            (request.work_dir / f"llm_repair_attempt_{request.attempt}_{attempt}.md").write_text(
                f"# Repair attempt {attempt}\n\n## Prompt\n\n{prompt}\n\n## Response\n\n{response}\n",
                encoding="utf-8",
            )
            candidate = extract_code(response)
            verdict = validate_generated_experiment(
                candidate, original, min_retained_fraction=self.min_retained_fraction
            )
            if verdict.ok:
                backup = request.work_dir / f"experiment_before_repair_{request.attempt}.py"
                backup.write_text(original, encoding="utf-8")
                script.write_text(candidate, encoding="utf-8")
                notes.extend(verdict.notes)
                notes.append(f"Original saved to {backup.name}; the rewrite is not assumed correct, only bounded.")
                return WorkspaceRepairResult(
                    attempted=True,
                    applied=True,
                    backend=self.name,
                    reason=(
                        f"A rewritten `{request.script_name}` passed the generated-code checks on "
                        f"attempt {attempt + 1} of {self.max_attempts}."
                    ),
                    notes=notes,
                )
            notes.append(f"Attempt {attempt + 1} rejected: " + "; ".join(verdict.violations))
            feedback = verdict.to_feedback()

        return WorkspaceRepairResult(
            attempted=True,
            applied=False,
            backend=self.name,
            reason=(
                f"No rewrite passed the generated-code checks in {self.max_attempts} attempt(s), "
                "so the workspace was left unchanged."
            ),
            notes=notes,
        )


def _llm_repair_prompt(
    script_name: str, original: str, context: RepairContext, feedback: str = ""
) -> str:
    """The whole file, the parsed failure, and the rules the rewrite must satisfy."""

    sections = [
        context.to_prompt(),
        "",
        "Rules for your rewrite:",
        f"- Return the complete contents of `{script_name}`, and nothing else.",
        "- Keep writing `results.csv` and keep printing the final `SUMMARY_JSON:` line.",
        "- Do not import socket, urllib, requests, subprocess, multiprocessing or shutil.",
        "- Do not call eval, exec or any shell.",
        "- Write only inside the script's own directory, using paths relative to __file__.",
        "- Do not delete the experiment to make the error go away.",
        "",
        f"Current `{script_name}`:",
        "```python",
        original,
        "```",
    ]
    if feedback:
        sections.extend(["", feedback])
    return "\n".join(sections)


class CompositeRepairBackend:
    """Dispatches to one backend, or tries them cheapest-first under `auto`.

    The order in `auto` is deliberate: the deterministic rule costs nothing and
    is certain when it applies, the LLM rewrite costs a call and is bounded but
    not certain, and Aider costs a subprocess and an external install. Each is
    only reached because the one before it had nothing to offer, and the reason
    it had nothing is carried forward into the result rather than discarded.
    """

    def __init__(
        self,
        mode: str = "deterministic",
        aider_command: str = "aider",
        aider_model: str | None = None,
        timeout_seconds: int = 120,
        llm=None,
        llm_attempts: int = 2,
    ) -> None:
        self.mode = mode
        self.deterministic = DeterministicRepairBackend()
        self.llm_backend = LLMRepairBackend(llm, max_attempts=llm_attempts)
        self.aider = AiderRepairBackend(aider_command, aider_model, timeout_seconds)

    def repair(self, request: WorkspaceRepairRequest) -> WorkspaceRepairResult:
        if self.mode == "disabled":
            return WorkspaceRepairResult(attempted=False, backend="disabled", reason="Code editing backend is disabled.")
        if self.mode == "deterministic":
            return self.deterministic.repair(request)
        if self.mode == "llm":
            return self.llm_backend.repair(request)
        if self.mode == "aider":
            return self.aider.repair(request)
        if self.mode == "auto":
            skipped: list[str] = []
            for backend in (self.deterministic, self.llm_backend, self.aider):
                result = backend.repair(request)
                if result.applied or result.attempted:
                    result.notes = skipped + list(result.notes)
                    return result
                skipped.append(f"{backend.name} backend skipped: {result.reason}")
            return WorkspaceRepairResult(
                attempted=False,
                backend="auto",
                reason="No backend attempted repair.",
                notes=skipped,
            )
        return WorkspaceRepairResult(
            attempted=False,
            backend=self.mode,
            reason=f"Unknown code editing backend `{self.mode}`.",
        )


def _aider_repair_prompt(script: Path, context: RepairContext) -> str:
    """Hand over the parsed failure, not the raw wall of stderr.

    The framework's own frames are already gone, so the editor's attention goes
    to the generated experiment rather than to the machinery that ran it.
    """

    return f"""You are repairing a generated Operations Research experiment script.

Edit only `{script.name}`.
Keep the script self-contained.
Preserve the required outputs:
- results.csv
- figures/ when plotting is available
- a final line beginning with SUMMARY_JSON:

Do not add network calls.
Do not call external solvers unless the current script already does so.
Do not change files outside this workspace.

{context.to_prompt()}
"""


def _move_future_imports_after_module_docstring(text: str) -> str:
    lines = text.splitlines()
    future_lines = [line for line in lines if line.startswith("from __future__ import")]
    other_lines = [line for line in lines if not line.startswith("from __future__ import")]
    if not future_lines:
        return text
    insert_at = 0
    if other_lines and other_lines[0].startswith('"""'):
        insert_at = 1
        if other_lines[0].count('"""') < 2:
            while insert_at < len(other_lines) and '"""' not in other_lines[insert_at]:
                insert_at += 1
            if insert_at < len(other_lines):
                insert_at += 1
    updated = other_lines[:insert_at] + future_lines + other_lines[insert_at:]
    return "\n".join(updated) + ("\n" if text.endswith("\n") else "")


def _preview(text: str, limit: int = 1000) -> str:
    text = text or ""
    if len(text) <= limit:
        return text
    return text[:limit] + "\n...[truncated]"
