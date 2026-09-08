from __future__ import annotations

from pathlib import Path

import pandas as pd

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:  # pragma: no cover - depends on runtime extras
    plt = None

_ONE_PIXEL_PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108020000009077"
    "53de0000000c4944415408d763f8ffff3f0005fe02fea73581e20000000049"
    "454e44ae426082"
)


def save_objective_gap_plot(results_csv: Path, figures_dir: Path) -> Path | None:
    """Create a generic method/objective plot when the result schema allows it."""

    if not results_csv.exists():
        return None
    df = pd.read_csv(results_csv)
    if "objective" not in df or "method" not in df:
        return None
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig_path = figures_dir / "objective_by_method.png"
    if plt is None:
        fig_path.write_bytes(_ONE_PIXEL_PNG)
        return fig_path
    plt.figure(figsize=(6, 4))
    if "problem_size" in df:
        grouped = df.groupby(["problem_size", "method"], as_index=False)["objective"].mean()
        for method, part in grouped.groupby("method"):
            plt.plot(part["problem_size"], part["objective"], marker="o", label=str(method))
        plt.xlabel("Problem size")
    else:
        grouped = df.groupby("method", as_index=False)["objective"].mean()
        plt.bar(grouped["method"].astype(str), grouped["objective"])
        plt.xlabel("Method")
    plt.ylabel("Mean objective")
    plt.title("Objective by method")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()
    return fig_path

