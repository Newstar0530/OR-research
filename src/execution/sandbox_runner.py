from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

from src.execution.result_parser import parse_summary_json
from src.schemas import ExecutionResult


class SandboxRunner:
    def __init__(self, timeout_seconds: int = 30) -> None:
        self.timeout_seconds = timeout_seconds

    def run_python(
        self,
        script_path: Path,
        run_dir: Path,
        log_name: str = "subprocess_last.log",
    ) -> ExecutionResult:
        script_path = script_path.resolve()
        run_dir = run_dir.resolve()
        if run_dir not in script_path.parents and script_path != run_dir:
            raise ValueError("Script must be inside the run directory.")
        env = os.environ.copy()
        env["RESEARCH_RUN_DIR"] = str(run_dir)
        # Generated experiments may need to import the engine (solvers, checkers)
        # from a workspace nested several directories below the project root.
        project_root = Path(__file__).resolve().parents[2]
        env["RESEARCH_PROJECT_ROOT"] = str(project_root)
        env["PYTHONPATH"] = os.pathsep.join(
            [str(project_root), *([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])]
        )
        start = time.perf_counter()
        timed_out = False
        try:
            proc = subprocess.run(
                [sys.executable, str(script_path)],
                cwd=str(run_dir),
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                env=env,
                shell=False,
            )
            stdout, stderr, returncode = proc.stdout, proc.stderr, proc.returncode
            status = "success" if returncode == 0 else "failed"
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            stdout = exc.stdout or ""
            stderr = exc.stderr or f"Timed out after {self.timeout_seconds} seconds."
            returncode = None
            status = "timeout"
        duration = time.perf_counter() - start
        metrics = parse_summary_json(stdout) if status == "success" else {}
        log_path = run_dir / log_name
        log_path.write_text(f"STDOUT\n{stdout}\n\nSTDERR\n{stderr}\n", encoding="utf-8")
        return ExecutionResult(
            status=status,
            returncode=returncode,
            stdout=stdout,
            stderr=stderr,
            duration_seconds=duration,
            timed_out=timed_out,
            metrics=metrics,
            log_path=log_path,
        )

