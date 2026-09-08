from __future__ import annotations

from pathlib import Path

import pandas as pd


def _markdown_table(df: pd.DataFrame) -> str:
    headers = [str(col) for col in df.columns]
    rows = []
    for row in df.itertuples(index=False):
        formatted = []
        for value in row:
            if isinstance(value, float):
                formatted.append(f"{value:.4g}")
            else:
                formatted.append(str(value))
        rows.append(formatted)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def evaluate_results(results_csv: str | Path) -> str:
    path = Path(results_csv)
    if not path.exists():
        return "# Result Evaluation\n\nNo results.csv found.\n"
    df = pd.read_csv(path)
    lines = ["# Result Evaluation", ""]
    lines.append(f"- Rows analyzed: {len(df)}")
    lines.append("")
    if {"method", "objective"}.issubset(df.columns):
        metric_cols = [col for col in ["objective", "runtime_seconds", "gap", "feasible", "robustness_metric"] if col in df.columns]
        grouped = df.groupby("method", as_index=False)[metric_cols].mean()
        lines.append("## Method-Level Summary")
        lines.append(_markdown_table(grouped))
        lines.append("")
        lines.append("## Hypothesis Checks")
        lines.append("- Confirm whether the objective is minimized or maximized before declaring one method better.")
        lines.append("- Check feasibility and constraint violation columns, if available.")
        lines.append("- Check whether the baseline is strong enough for the research question.")
        lines.append("")
        lines.append("## Recommended Next Experiments")
        lines.append("- Add repeated seeds and confidence intervals.")
        lines.append("- Add stronger baselines or exact solver references where feasible.")
        lines.append("- Add sensitivity sweeps over domain-specific parameters.")
    else:
        lines.append("## Generic Summary")
        numeric = df.select_dtypes(include="number")
        if not numeric.empty:
            summary = numeric.describe().reset_index().rename(columns={"index": "stat"})
            lines.append(_markdown_table(summary))
        lines.append("")
        lines.append("## Recommended Next Experiments")
        lines.append("- Add domain-specific metrics for stronger automated interpretation.")
    lines.append("")
    lines.append("## Claim Constraint")
    lines.append("- Treat all conclusions as preliminary until literature claims and statistical intervals are verified.")
    return "\n".join(lines) + "\n"
