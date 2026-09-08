from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
from pydantic import BaseModel, Field


class AutonomyReadinessReport(BaseModel):
    score: int = Field(ge=0, le=100)
    level: str
    passed_checks: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    blockers: list[str] = Field(default_factory=list)
    required_human_actions: list[str] = Field(default_factory=list)

    def to_markdown(self) -> str:
        lines = [
            "# Autonomy Readiness",
            "",
            f"- score: {self.score}/100",
            f"- level: {self.level}",
            "",
            "## Passed Checks",
        ]
        lines.extend(f"- {item}" for item in self.passed_checks or ["None recorded."])
        lines.append("")
        lines.append("## Warnings")
        lines.extend(f"- {item}" for item in self.warnings or ["None recorded."])
        lines.append("")
        lines.append("## Blockers")
        lines.extend(f"- {item}" for item in self.blockers or ["None recorded."])
        lines.append("")
        lines.append("## Required Human Actions")
        lines.extend(f"- {item}" for item in self.required_human_actions or ["Review final report before using results."])
        lines.append("")
        lines.append("## Interpretation")
        lines.append(
            "This score estimates whether the run produced enough inspectable evidence for a human researcher to review. "
            "It is not permission to submit, publish, or treat conclusions as verified."
        )
        return "\n".join(lines) + "\n"


def evaluate_autonomy_readiness(run_dir: str | Path) -> AutonomyReadinessReport:
    root = Path(run_dir)
    score = 0
    passed: list[str] = []
    warnings: list[str] = []
    blockers: list[str] = []
    human_actions: list[str] = []

    required_artifacts = [
        "selected_idea.json",
        "novelty_report.md",
        "model_draft.md",
        "model_critique.md",
        "algorithm_plan.md",
        "results.csv",
        "sensitivity_report.md",
        "final_report.md",
        "automated_review.md",
        "claim_check.md",
    ]
    present = [name for name in required_artifacts if (root / name).exists()]
    missing = [name for name in required_artifacts if name not in present]
    score += int(25 * len(present) / len(required_artifacts))
    if missing:
        blockers.append("Missing required artifacts: " + ", ".join(missing))
    else:
        passed.append("All core inspectable artifacts are present.")

    contract_text = _read(root / "experiment_contract.md")
    if "Status: passed" in contract_text:
        score += 20
        passed.append("Experiment output contract passed.")
    else:
        blockers.append("Experiment output contract did not pass or is missing.")

    results_path = root / "results.csv"
    if results_path.exists():
        try:
            df = pd.read_csv(results_path)
            if df.empty:
                blockers.append("results.csv is empty.")
            else:
                score += 10
                passed.append(f"results.csv contains {len(df)} rows.")
                if "method" in df.columns and df["method"].nunique() >= 2:
                    score += 10
                    passed.append("At least two methods are compared.")
                else:
                    warnings.append("Results do not clearly compare multiple methods.")
                if "seed" in df.columns and df["seed"].nunique() >= 2:
                    score += 5
                    passed.append("Multiple seeds or replications are present.")
                else:
                    warnings.append("Few or no repeated seeds were detected.")
                if any(col in df.columns for col in ["gap", "constraint_violation", "feasible", "robustness_metric"]):
                    score += 5
                    passed.append("Results include at least one quality or feasibility diagnostic.")
                else:
                    warnings.append("No explicit gap, feasibility, violation, or robustness diagnostic column was found.")
        except Exception as exc:
            blockers.append(f"Could not parse results.csv: {exc}")

    claim_text = _read(root / "claim_check.md")
    claim_hits = len(re.findall(r"Needs support:", claim_text))
    if claim_hits == 0 and claim_text:
        score += 10
        passed.append("Claim checker found no high-risk claim pattern.")
    elif claim_hits > 0:
        warnings.append(f"Claim checker flagged {claim_hits} unsupported high-risk claim snippets.")
        human_actions.append("Verify every flagged claim against literature and experiment evidence.")

    novelty_text = _read(root / "novelty_report.md").lower()
    if "no verified sources found" in novelty_text or "mock" in novelty_text:
        warnings.append("Novelty evidence is weak or mock-generated.")
        human_actions.append("Run a real literature search and attach verified citations before claiming novelty.")
    elif novelty_text:
        score += 5
        passed.append("Novelty report contains non-empty evidence text.")

    review_text = _read(root / "automated_review.md")
    overall = _extract_score(review_text, "Overall")
    if overall is not None and overall >= 6:
        score += 10
        passed.append(f"Automated reviewer overall score is {overall}/10.")
    elif overall is not None:
        warnings.append(f"Automated reviewer overall score is low: {overall}/10.")
    else:
        warnings.append("Could not extract automated reviewer overall score.")

    if (root / "human_gate_requests.md").exists():
        warnings.append("LLM agents requested explicit human review gates.")
        human_actions.append("Review human_gate_requests.md before extending experiments.")

    if not human_actions:
        human_actions.append("Review model assumptions, experiment fairness, and report claims before using conclusions.")

    score = max(0, min(100, score))
    level = _level(score, blockers)
    return AutonomyReadinessReport(
        score=score,
        level=level,
        passed_checks=passed,
        warnings=warnings,
        blockers=blockers,
        required_human_actions=human_actions,
    )


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore") if path.exists() else ""


def _extract_score(text: str, label: str) -> int | None:
    match = re.search(rf"{re.escape(label)}:\s*(\d+)\s*/\s*10", text, flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def _level(score: int, blockers: list[str]) -> str:
    if blockers:
        return "blocked"
    if score >= 80:
        return "review-ready"
    if score >= 60:
        return "needs-human-review"
    return "exploratory-only"
