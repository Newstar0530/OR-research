from pathlib import Path

from src.agent_system.diff_patch import SafeUnifiedDiffApplier, UnifiedDiffProposal


def test_safe_unified_diff_applies_and_validates_python(tmp_path: Path) -> None:
    target = tmp_path / "experiment.py"
    target.write_text(
        'print("results.csv")\nprint("SUMMARY_JSON: {}")\nVALUE = 1\n',
        encoding="utf-8",
    )
    diff = """--- experiment.py
+++ experiment.py
@@ -1,3 +1,3 @@
 print("results.csv")
 print("SUMMARY_JSON: {}")
-VALUE = 1
+VALUE = 2
"""

    result = SafeUnifiedDiffApplier(tmp_path).apply(UnifiedDiffProposal(diff=diff))

    assert result.applied
    assert "VALUE = 2" in target.read_text(encoding="utf-8")


def test_safe_unified_diff_rejects_marker_removal(tmp_path: Path) -> None:
    target = tmp_path / "experiment.py"
    target.write_text('print("results.csv")\nprint("SUMMARY_JSON: {}")\n', encoding="utf-8")
    diff = """--- experiment.py
+++ experiment.py
@@ -1,2 +1,1 @@
 print("results.csv")
-print("SUMMARY_JSON: {}")
"""

    result = SafeUnifiedDiffApplier(tmp_path).apply(UnifiedDiffProposal(diff=diff))

    assert not result.applied
