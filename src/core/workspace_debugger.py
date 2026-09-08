from __future__ import annotations

from pathlib import Path

from src.core.code_editing_backend import (
    CompositeRepairBackend,
    WorkspaceRepairRequest,
    WorkspaceRepairResult,
)


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
) -> WorkspaceRepairResult:
    request = WorkspaceRepairRequest(
        work_dir=Path(work_dir),
        stderr=stderr,
        stdout=stdout,
        attempt=attempt,
    )
    return CompositeRepairBackend(
        mode=backend,
        aider_command=aider_command,
        aider_model=aider_model,
        timeout_seconds=timeout_seconds,
    ).repair(request)
