from pathlib import Path

from src.main import main
from src.utils.run_tracker import RunTracker, load_run_status, write_run_status


def test_run_tracker_writes_status(tmp_path: Path) -> None:
    tracker = RunTracker.started()
    tracker.finish("completed", tmp_path, ["stage"])
    write_run_status(tmp_path, tracker)

    loaded = load_run_status(tmp_path)

    assert loaded["status"] == "completed"
    assert (tmp_path / "run_status.md").exists()


def test_resume_cli_skips_completed_run(tmp_path: Path) -> None:
    tracker = RunTracker.started()
    tracker.finish("completed", tmp_path, ["stage"])
    write_run_status(tmp_path, tracker)

    assert main(["resume", "--run-dir", str(tmp_path)]) == 0
