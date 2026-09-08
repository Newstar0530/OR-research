from src.utils.research_memory import append_memory


def test_append_memory(tmp_path) -> None:
    path = append_memory(tmp_path, "experiments", {"project_name": "x"})
    assert path.exists()
    assert "project_name" in path.read_text(encoding="utf-8")

