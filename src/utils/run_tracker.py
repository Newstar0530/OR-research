from __future__ import annotations

import json
import time
from pathlib import Path

from pydantic import BaseModel, Field


class RunTracker(BaseModel):
    status: str = "started"
    started_at_epoch: float
    ended_at_epoch: float | None = None
    duration_seconds: float | None = None
    completed_stages: list[str] = Field(default_factory=list)
    approximate_artifact_tokens: int = 0
    notes: list[str] = Field(default_factory=list)

    @classmethod
    def started(cls) -> "RunTracker":
        return cls(started_at_epoch=time.time())

    def finish(self, status: str, run_dir: str | Path, completed_stages: list[str]) -> None:
        self.status = status
        self.ended_at_epoch = time.time()
        self.duration_seconds = self.ended_at_epoch - self.started_at_epoch
        self.completed_stages = list(completed_stages)
        self.approximate_artifact_tokens = approximate_artifact_tokens(run_dir)

    def to_markdown(self) -> str:
        lines = ["# Run Tracker", "", f"- status: {self.status}", f"- duration_seconds: {self.duration_seconds if self.duration_seconds is not None else 'running'}", f"- approximate_artifact_tokens: {self.approximate_artifact_tokens}", ""]
        lines.append("## Completed Stages")
        lines.extend(f"- {stage}" for stage in self.completed_stages or ["None recorded."])
        lines.append("")
        lines.append("## Notes")
        lines.extend(f"- {note}" for note in self.notes or ["Token counts are approximate local artifact word counts, not provider billing records."])
        return "\n".join(lines) + "\n"


def write_run_status(run_dir: str | Path, tracker: RunTracker) -> None:
    root = Path(run_dir)
    (root / "run_status.json").write_text(tracker.model_dump_json(indent=2), encoding="utf-8")
    (root / "run_status.md").write_text(tracker.to_markdown(), encoding="utf-8")


def approximate_artifact_tokens(run_dir: str | Path) -> int:
    root = Path(run_dir)
    total_words = 0
    for path in root.glob("*.md"):
        try:
            total_words += len(path.read_text(encoding="utf-8", errors="ignore").split())
        except Exception:
            continue
    return int(total_words * 1.3)


def load_run_status(run_dir: str | Path) -> dict:
    path = Path(run_dir) / "run_status.json"
    if not path.exists():
        return {"status": "unknown", "notes": ["run_status.json not found."]}
    return json.loads(path.read_text(encoding="utf-8"))
