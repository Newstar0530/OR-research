from __future__ import annotations

import re
from pathlib import Path


RISK_PATTERNS = {
    "novelty_claim": [r"\bnovel\b", r"\bnew\b", r"first\s+to", r"previously\s+unexplored"],
    "quantum_advantage": [r"quantum advantage", r"quantum speedup", r"outperform[s]?", r"superior"],
    "proof_claim": [r"\bprove[sdn]?\b", r"guarantee[s]?", r"optimality\s+proof", r"convergence\s+proof"],
    "strong_generalization": [r"\balways\b", r"\buniversally\b", r"for\s+all\s+cases", r"without\s+exception"],
}


def check_report_claims(report_text: str, literature_index: Path | None = None) -> str:
    lines = ["# Claim Check", ""]
    lines.append("This automated check flags high-risk research claims that need human verification.")
    lines.append("")
    found_any = False
    for label, patterns in RISK_PATTERNS.items():
        hits = []
        for pattern in patterns:
            for match in re.finditer(pattern, report_text, flags=re.IGNORECASE):
                start = max(0, match.start() - 90)
                end = min(len(report_text), match.end() + 120)
                snippet = re.sub(r"\s+", " ", report_text[start:end]).strip()
                hits.append(snippet)
        if hits:
            found_any = True
            lines.append(f"## {label}")
            for hit in hits[:8]:
                lines.append(f"- Needs support: {hit}")
            lines.append("")
    if not found_any:
        lines.append("No high-risk claim patterns were detected.")
        lines.append("")
    if literature_index and literature_index.exists():
        lines.append("## Local Literature Evidence")
        lines.append(f"- Literature index available: `{literature_index.name}`")
        lines.append("- This check does not verify citations automatically; use the index to confirm author/year/source details.")
    else:
        lines.append("## Local Literature Evidence")
        lines.append("- No literature index was available for this run.")
    lines.append("")
    lines.append("## Required Human Checks")
    lines.append("- Verify every novelty statement against the local and external literature.")
    lines.append("- Do not claim quantum advantage without controlled classical baselines and statistical evidence.")
    lines.append("- Verify KKT assumptions, Big-M values, QUBO penalties, and discretization error before drawing conclusions.")
    return "\n".join(lines) + "\n"
