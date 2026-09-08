from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.llm_client import LLMClient
from src.schemas import SensitivityReport
from src.utils.prompt_utils import human_verification_footer


class SensitivityAgent:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def run(self, results_csv: Path) -> SensitivityReport:
        tested = ["problem_size", "method", "seed"]
        effects = ["No results.csv was available for automated sensitivity analysis."]
        summary = "# Sensitivity Analysis\n\nNo executable results were found."
        if results_csv.exists():
            df = pd.read_csv(results_csv)
            numeric_cols = list(df.select_dtypes(include="number").columns)
            candidate_parameters = [
                col
                for col in ["problem_size", "capacity_ratio", "tightness", "density", "arrival_rate", "demand_sigma", "shortage_cost"]
                if col in df.columns
            ]
            tested = candidate_parameters or [col for col in ["method", "seed"] if col in df.columns]
            if {"method", "objective"}.issubset(df.columns):
                grouped = df.groupby("method", as_index=False)["objective"].mean()
                table = "\n".join(
                    f"- method={row.method}, mean_objective={row.objective:.4g}"
                    for row in grouped.itertuples(index=False)
                )
                effects = [
                    f"Objective was summarized for {len(grouped)} methods.",
                    "Method-level objective summary:\n" + table,
                ]
                discussion = "Compare objective direction, feasibility, and baseline fairness before making claims."
            elif numeric_cols:
                summary_stats = df[numeric_cols].describe().reset_index().rename(columns={"index": "stat"})
                table = "\n".join(
                    "- "
                    + ", ".join(
                        f"{col}={getattr(row, col) if col == 'stat' else float(getattr(row, col)):.4g}"
                        if col != "stat"
                        else f"{col}={getattr(row, col)}"
                        for col in summary_stats.columns
                    )
                    for row in summary_stats.itertuples(index=False)
                )
                effects = [f"Loaded {len(df)} result rows.", "Numeric summary:\n" + table]
                discussion = "Add method and objective columns to enable stronger automated comparison."
            else:
                effects = [f"Loaded {len(df)} result rows, but no numeric columns were detected."]
                discussion = "Add numeric metrics and domain-specific parameters to enable sensitivity analysis."
            summary = (
                "# Sensitivity Analysis\n\n"
                "## Tested Parameters\n"
                + "\n".join(f"- {p}" for p in tested)
                + "\n\n## Observed Effects\n"
                f"{effects[0]}\n\n{effects[1] if len(effects) > 1 else ''}\n\n"
                "## Robustness Discussion\n"
                f"{discussion} These findings are preliminary and generated from local experiments.\n\n"
                "## Suspicious or Unstable Findings\n"
                "Any surprising result should be manually checked against model assumptions, random seeds, bounds, feasibility, and data generation.\n"
            )
        return SensitivityReport(markdown=summary + human_verification_footer(), tested_parameters=tested, observed_effects=effects)

