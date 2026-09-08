from pathlib import Path

from src.execution.experiment_logger import ExperimentLogger


def test_experiment_logger_writes_journal_and_json(tmp_path: Path) -> None:
    logger = ExperimentLogger(tmp_path)
    logger.append("stage", "success", metrics={"x": 1}, notes="ok")
    assert "stage" in (tmp_path / "experiment_journal.md").read_text(encoding="utf-8")
    assert (tmp_path / "execution_log.json").exists()

