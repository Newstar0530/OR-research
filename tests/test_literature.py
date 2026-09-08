from pathlib import Path

from src.utils.literature import build_literature_index


def test_literature_index_empty_folder(tmp_path: Path) -> None:
    index_path, review_path = build_literature_index(tmp_path, tmp_path / "out")
    assert index_path.exists()
    assert review_path.exists()
    assert "No PDF or DOCX files found" in review_path.read_text(encoding="utf-8")
