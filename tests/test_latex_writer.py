from pathlib import Path

from src.utils.latex_writer import write_latex_paper


def test_latex_writer_creates_tex(tmp_path: Path) -> None:
    (tmp_path / "final_report.md").write_text("# Title\n\n- A cautious claim", encoding="utf-8")
    (tmp_path / "model_draft.md").write_text("# Model\n\nNeeds human verification.", encoding="utf-8")
    (tmp_path / "statistical_evidence.md").write_text("# Stats\n\nPreliminary.", encoding="utf-8")
    (tmp_path / "literature_grounding.md").write_text("# Grounding\n\nWeak.", encoding="utf-8")
    (tmp_path / "automated_review.md").write_text("# Review\n\nBorderline.", encoding="utf-8")

    path = write_latex_paper(tmp_path)

    text = path.read_text(encoding="utf-8")
    assert "\\documentclass" in text
    assert "AI-Generated Draft Warning" in text
