from pathlib import Path

import pandas as pd

from src.utils.citation_manager import build_citation_artifacts


def test_citation_manager_writes_bibtex(tmp_path: Path) -> None:
    index = tmp_path / "literature_index.csv"
    pd.DataFrame(
        [{"title_guess": "Robust Network Reliability", "filename": "paper.pdf", "path": "C:/paper.pdf"}]
    ).to_csv(index, index=False)

    report = build_citation_artifacts(index, tmp_path)

    assert report.records[0].key == "robust_network_reliability"
    assert "@misc{robust_network_reliability" in (tmp_path / "references.bib").read_text(encoding="utf-8")
