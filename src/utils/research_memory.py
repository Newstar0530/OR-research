from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


def append_memory(project_root: Path, entry_type: str, payload: dict[str, Any]) -> Path:
    memory_dir = project_root / "research_memory"
    memory_dir.mkdir(parents=True, exist_ok=True)
    path = memory_dir / f"{entry_type}.jsonl"
    record = {"timestamp": datetime.now().isoformat(timespec="seconds"), **payload}
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    return path

