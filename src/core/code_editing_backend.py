from __future__ import annotations

import ast
import shutil
import subprocess
from pathlib import Path

from pydantic import BaseModel, Field


class WorkspaceRepairRequest(BaseModel):
    work_dir: Path
    stderr: str = ""
    stdout: str = ""
    attempt: int = 0
    script_name: str = "experiment.py"


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
        if "from __future__ imports must occur at the beginning" in request.stderr:
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
        return WorkspaceRepairResult(attempted=False, backend=self.name, reason="No deterministic repair rule matched stderr.")


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
        prompt_path.write_text(_aider_repair_prompt(script, request.stderr, request.stdout), encoding="utf-8")
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


class CompositeRepairBackend:
    def __init__(
        self,
        mode: str = "deterministic",
        aider_command: str = "aider",
        aider_model: str | None = None,
        timeout_seconds: int = 120,
    ) -> None:
        self.mode = mode
        self.deterministic = DeterministicRepairBackend()
        self.aider = AiderRepairBackend(aider_command, aider_model, timeout_seconds)

    def repair(self, request: WorkspaceRepairRequest) -> WorkspaceRepairResult:
        if self.mode == "disabled":
            return WorkspaceRepairResult(attempted=False, backend="disabled", reason="Code editing backend is disabled.")
        if self.mode == "deterministic":
            return self.deterministic.repair(request)
        if self.mode == "aider":
            return self.aider.repair(request)
        if self.mode == "auto":
            first = self.deterministic.repair(request)
            if first.applied or first.attempted:
                return first
            second = self.aider.repair(request)
            if second.attempted or second.applied:
                second.notes.insert(0, f"Deterministic backend skipped: {first.reason}")
                return second
            return WorkspaceRepairResult(
                attempted=False,
                backend="auto",
                reason=f"No backend attempted repair. deterministic={first.reason}; aider={second.reason}",
            )
        return WorkspaceRepairResult(
            attempted=False,
            backend=self.mode,
            reason=f"Unknown code editing backend `{self.mode}`.",
        )


def _aider_repair_prompt(script: Path, stderr: str, stdout: str) -> str:
    return f"""You are repairing a generated Operations Research experiment script.

Edit only `{script.name}`.
Make the smallest change needed to fix the failure.
Keep the script self-contained.
Preserve the required outputs:
- results.csv
- figures/ when plotting is available
- a final line beginning with SUMMARY_JSON:

Do not add network calls.
Do not call external solvers unless the current script already does so.
Do not change files outside this workspace.

STDOUT:
```text
{_preview(stdout, 2000)}
```

STDERR:
```text
{_preview(stderr, 4000)}
```
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
