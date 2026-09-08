from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any

from src.schemas import ExperimentLogEntry
from src.utils.file_utils import write_json


class ExperimentLogger:
    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.journal_path = run_dir / "experiment_journal.md"
        self.log_path = run_dir / "execution_log.json"
        self.entries: list[ExperimentLogEntry] = []
        if not self.journal_path.exists():
            self.journal_path.write_text("# Experiment Journal\n\n", encoding="utf-8")

    def append(self, stage: str, status: str, code_path: Path | None = None, metrics: dict[str, Any] | None = None, plots: list[str] | None = None, notes: str = "") -> None:
        entry = ExperimentLogEntry(
            timestamp=datetime.now(),
            stage=stage,
            status=status,
            code_path=str(code_path) if code_path else None,
            metrics=metrics or {},
            plots=plots or [],
            notes=notes,
        )
        self.entries.append(entry)
        version = ""
        if code_path and code_path.exists():
            digest = hashlib.sha256(code_path.read_bytes()).hexdigest()[:12]
            version = f"\n- code_sha256_12: `{digest}`"
        with self.journal_path.open("a", encoding="utf-8") as f:
            f.write(f"## {entry.timestamp.isoformat(timespec='seconds')} - {stage}\n")
            f.write(f"- status: {status}{version}\n")
            if entry.metrics:
                f.write(f"- metrics: `{entry.metrics}`\n")
            if entry.plots:
                f.write(f"- plots: {', '.join(entry.plots)}\n")
            if notes:
                f.write(f"- notes: {notes}\n")
            f.write("\n")
        write_json(self.log_path, [e.model_dump() for e in self.entries])

