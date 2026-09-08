from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.llm_client import LLMClient


class GapSynthesisAgent:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def run(self, research_goal: str, domain_profile: dict, literature_index: Path | None = None) -> str:
        profile_name = domain_profile.get("name", "unknown")
        papers = []
        if literature_index and literature_index.exists():
            df = pd.read_csv(literature_index)
            if "keyword_hits" in df.columns:
                df["score"] = df["keyword_hits"].fillna("").apply(
                    lambda s: sum(term in s for term in ["supply_chain", "chaos", "reliability", "adaptive_control", "network"])
                )
                papers = df.sort_values(["score", "filename"], ascending=[False, True]).head(12).to_dict("records")

        lines = [
            "# Research Gap Analysis",
            "",
            f"## Research Goal",
            research_goal,
            "",
            f"## Selected Domain",
            profile_name,
            "",
            "## Local Literature Signals",
        ]
        if papers:
            for paper in papers:
                lines.append(f"- `{paper.get('filename')}`: {paper.get('keyword_hits') or 'no keyword hits'}")
        else:
            lines.append("- No local literature index was available.")
        lines += [
            "",
            "## Candidate Research Gaps",
            "- Existing supply-chain chaos studies often analyze instability, but do not always connect early-warning indicators to reliability-oriented control metrics.",
            "- Reliability studies often evaluate failure probability or multi-state performance, but may not use chaos precursor indicators as online triggers.",
            "- Adaptive control papers may reduce bullwhip or synchronize demand-supply response, but need explicit false-alarm, missed-detection, and lead-time evaluation.",
            "",
            "## Candidate Innovation Points",
            "- Define a chaos-precursor indicator set combining rolling variance, lag-1 autocorrelation, bullwhip ratio, and recovery-time signals.",
            "- Use the indicator as a trigger for adaptive replenishment control in a multi-echelon supply-chain network.",
            "- Evaluate both control performance and reliability outcomes: service level, stockout rate, instability lead time, false alarms, and recovery time.",
            "",
            "## Testable Hypotheses",
            "- H1: Chaos precursor indicators rise before service-level degradation under transport delay or demand regime shifts.",
            "- H2: Indicator-triggered adaptive control reduces stockout rate and recovery time compared with a fixed order-up-to policy.",
            "- H3: Overly sensitive warning thresholds reduce instability impact but increase false alarms and operating cost.",
            "",
            "## Required Human Verification",
            "- Confirm whether extracted local papers truly support each gap and citation claim.",
            "- Verify that simulated instability is not merely random noise mislabeled as chaos.",
            "- Validate the mathematical definition of each precursor indicator before thesis use.",
        ]
        return "\n".join(lines) + "\n"

