from pathlib import Path

import pandas as pd

from src.utils.literature_grounding import build_literature_grounding


def test_literature_grounding_matches_local_snippet(tmp_path: Path) -> None:
    report = tmp_path / "final_report.md"
    report.write_text(
        "The literature suggests robust network reliability models improve supply chain disruption analysis.",
        encoding="utf-8",
    )
    index = tmp_path / "literature_index.csv"
    pd.DataFrame(
        [
            {
                "filename": "network_reliability.pdf",
                "title_guess": "Network reliability",
                "keyword_hits": "network_analysis",
                "category": "network_analysis",
                "snippet": "Robust network reliability for supply chain disruption analysis.",
            }
        ]
    ).to_csv(index, index=False)

    grounding = build_literature_grounding(report, index, tmp_path)

    assert grounding.claims_checked == 1
    assert grounding.grounded_claims == 1
    assert grounding.records[0].embedding_score > 0
    assert (tmp_path / "literature_grounding.md").exists()
