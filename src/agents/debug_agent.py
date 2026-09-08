from __future__ import annotations

from pathlib import Path

from src.execution.sandbox_runner import SandboxRunner
from src.llm_client import LLMClient
from src.schemas import DebugReport, ExecutionResult


class DebugAgent:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def run(self, code_path: Path, run_dir: Path, runner: SandboxRunner, max_attempts: int) -> tuple[ExecutionResult, DebugReport]:
        attempts = 0
        patches: list[str] = []
        result = runner.run_python(code_path, run_dir)
        while result.status != "success" and attempts < max_attempts:
            attempts += 1
            failed_copy = run_dir / f"generated_experiment_failed_attempt_{attempts}.py"
            failed_copy.write_text(code_path.read_text(encoding="utf-8"), encoding="utf-8")
            patches.append("No automatic patch applied in MVP mock mode; failure artifact saved for human inspection.")
            result = runner.run_python(code_path, run_dir)
        report = DebugReport(
            debug_attempts=attempts,
            final_status=result.status,
            remaining_errors=result.stderr,
            patch_summary=patches,
        )
        return result, report

