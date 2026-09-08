from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field


class PatchOperation(BaseModel):
    find: str
    replace: str
    occurrence: int = 1
    rationale: str = ""


class PatchProposal(BaseModel):
    target_file: str = "experiment.py"
    rationale: str
    operations: list[PatchOperation] = Field(default_factory=list)
    expected_effect: str
    safety_checks: list[str] = Field(default_factory=list)


class PatchResult(BaseModel):
    applied: bool
    reason: str
    operations_applied: int = 0


class SafePatchApplier:
    required_markers = ["results.csv", "SUMMARY_JSON"]

    def __init__(self, work_dir: str | Path) -> None:
        self.work_dir = Path(work_dir).resolve()

    def apply(self, proposal: PatchProposal) -> PatchResult:
        target = (self.work_dir / proposal.target_file).resolve()
        try:
            target.relative_to(self.work_dir)
        except ValueError:
            return PatchResult(applied=False, reason="Patch target is outside work directory.")
        if target.name != "experiment.py":
            return PatchResult(applied=False, reason="Only experiment.py can be patched by the safe patcher.")
        if not target.exists():
            return PatchResult(applied=False, reason="Patch target does not exist.")
        text = target.read_text(encoding="utf-8")
        if not proposal.operations:
            return PatchResult(applied=False, reason="Patch proposal contains no operations.")
        updated = text
        applied = 0
        for operation in proposal.operations:
            if not operation.find:
                return PatchResult(applied=False, reason="Patch find string cannot be empty.")
            if operation.find not in updated:
                return PatchResult(applied=False, reason=f"Find string not found: {operation.find[:80]}")
            updated = _replace_nth(updated, operation.find, operation.replace, operation.occurrence)
            applied += 1
        missing_markers = [marker for marker in self.required_markers if marker not in updated]
        if missing_markers:
            return PatchResult(applied=False, reason=f"Patch would remove required markers: {missing_markers}")
        target.write_text(updated, encoding="utf-8")
        return PatchResult(applied=True, reason="Patch applied.", operations_applied=applied)


def _replace_nth(text: str, find: str, replace: str, occurrence: int) -> str:
    if occurrence <= 0:
        occurrence = 1
    start = -1
    search_from = 0
    for _ in range(occurrence):
        start = text.find(find, search_from)
        if start == -1:
            raise ValueError("Occurrence not found.")
        search_from = start + len(find)
    return text[:start] + replace + text[start + len(find):]

