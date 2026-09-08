from __future__ import annotations

import ast
from pathlib import Path

from pydantic import BaseModel, Field


class UnifiedDiffProposal(BaseModel):
    target_file: str = "experiment.py"
    diff: str
    rationale: str = ""
    required_markers: list[str] = Field(default_factory=lambda: ["results.csv", "SUMMARY_JSON"])


class UnifiedDiffResult(BaseModel):
    applied: bool
    reason: str


class SafeUnifiedDiffApplier:
    def __init__(self, work_dir: str | Path) -> None:
        self.work_dir = Path(work_dir).resolve()

    def apply(self, proposal: UnifiedDiffProposal) -> UnifiedDiffResult:
        target = (self.work_dir / proposal.target_file).resolve()
        try:
            target.relative_to(self.work_dir)
        except ValueError:
            return UnifiedDiffResult(applied=False, reason="Patch target is outside work directory.")
        if target.name != "experiment.py":
            return UnifiedDiffResult(applied=False, reason="Only experiment.py is supported.")
        if not target.exists():
            return UnifiedDiffResult(applied=False, reason="Target file does not exist.")
        original = target.read_text(encoding="utf-8")
        try:
            updated = apply_simple_unified_diff(original, proposal.diff)
            ast.parse(updated)
        except Exception as exc:
            return UnifiedDiffResult(applied=False, reason=f"Patch failed validation: {exc}")
        missing = [marker for marker in proposal.required_markers if marker not in updated]
        if missing:
            return UnifiedDiffResult(applied=False, reason=f"Patch would remove required markers: {missing}")
        target.write_text(updated, encoding="utf-8")
        return UnifiedDiffResult(applied=True, reason="Unified diff patch applied.")


def apply_simple_unified_diff(original: str, diff: str) -> str:
    lines = original.splitlines()
    output: list[str] = []
    cursor = 0
    diff_lines = [line for line in diff.splitlines() if not line.startswith(("---", "+++"))]
    index = 0
    while index < len(diff_lines):
        line = diff_lines[index]
        if not line.startswith("@@"):
            index += 1
            continue
        old_start = _parse_hunk_start(line)
        output.extend(lines[cursor:old_start])
        cursor = old_start
        index += 1
        while index < len(diff_lines) and not diff_lines[index].startswith("@@"):
            item = diff_lines[index]
            if item.startswith(" "):
                expected = item[1:]
                if cursor >= len(lines) or lines[cursor] != expected:
                    raise ValueError(f"Context mismatch: {expected[:80]}")
                output.append(lines[cursor])
                cursor += 1
            elif item.startswith("-"):
                expected = item[1:]
                if cursor >= len(lines) or lines[cursor] != expected:
                    raise ValueError(f"Deletion mismatch: {expected[:80]}")
                cursor += 1
            elif item.startswith("+"):
                output.append(item[1:])
            index += 1
    output.extend(lines[cursor:])
    return "\n".join(output) + ("\n" if original.endswith("\n") else "")


def _parse_hunk_start(header: str) -> int:
    # Example: @@ -3,7 +3,8 @@
    old = header.split(" ")[1]
    start = int(old.split(",")[0].lstrip("-"))
    return max(0, start - 1)
