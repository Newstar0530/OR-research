from pathlib import Path

from src.utils.bfts_tree import build_research_tree


def test_bfts_tree_selects_best_candidates(tmp_path: Path) -> None:
    journal = tmp_path / "autonomous_journal.json"
    journal.write_text(
        '{"nodes": ['
        '{"id": "a", "parent_id": null, "iteration": 0, "branch_index": 0, "status": "success", "metric_value": 10},'
        '{"id": "b", "parent_id": "a", "iteration": 1, "branch_index": 0, "status": "success", "metric_value": 8}'
        "]}",
        encoding="utf-8",
    )

    report = build_research_tree(journal, tmp_path, maximize=False)

    assert report.best_node_id == "b"
    assert "b" in report.next_expansion_candidates
    assert (tmp_path / "research_tree.md").exists()
