from __future__ import annotations

from pathlib import Path

from src.core.code_editing_backend import (
    CompositeRepairBackend,
    WorkspaceRepairRequest,
    WorkspaceRepairResult,
)
from src.execution.traceback_parser import RepairContext, build_repair_context


def repair_workspace_after_failure(
    work_dir: str | Path,
    stderr: str,
    *,
    stdout: str = "",
    attempt: int = 0,
    backend: str = "deterministic",
    aider_command: str = "aider",
    aider_model: str | None = None,
    timeout_seconds: int = 120,
    script_name: str = "experiment.py",
    previous_attempts: list[str] | None = None,
    context: RepairContext | None = None,
    llm=None,
    llm_attempts: int = 2,
) -> WorkspaceRepairResult:
    request = WorkspaceRepairRequest(
        work_dir=Path(work_dir),
        stderr=stderr,
        stdout=stdout,
        attempt=attempt,
        script_name=script_name,
        context=context
        or build_repair_context(
            stderr,
            Path(work_dir),
            script_name=script_name,
            stdout=stdout,
            attempt=attempt,
            previous_attempts=previous_attempts or (),
        ),
    )
    return CompositeRepairBackend(
        mode=backend,
        aider_command=aider_command,
        aider_model=aider_model,
        timeout_seconds=timeout_seconds,
        llm=llm,
        llm_attempts=llm_attempts,
    ).repair(request)
