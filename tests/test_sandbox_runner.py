from pathlib import Path

from src.execution.sandbox_runner import SandboxRunner


def test_sandbox_runner_success(tmp_path: Path) -> None:
    script = tmp_path / "ok.py"
    script.write_text('print("SUMMARY_JSON:" + "{\\"metric\\": 1}")\n', encoding="utf-8")
    result = SandboxRunner(timeout_seconds=5).run_python(script, tmp_path)
    assert result.status == "success"
    assert result.metrics["metric"] == 1


def test_sandbox_runner_failed_code(tmp_path: Path) -> None:
    script = tmp_path / "bad.py"
    script.write_text('raise RuntimeError("boom")\n', encoding="utf-8")
    result = SandboxRunner(timeout_seconds=5).run_python(script, tmp_path)
    assert result.status == "failed"
    assert "boom" in result.stderr


def test_sandbox_runner_timeout(tmp_path: Path) -> None:
    script = tmp_path / "slow.py"
    script.write_text("import time\ntime.sleep(2)\n", encoding="utf-8")
    result = SandboxRunner(timeout_seconds=1).run_python(script, tmp_path)
    assert result.status == "timeout"

