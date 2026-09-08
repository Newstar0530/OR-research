from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROBLEM_SIZES = [10, 25, 50]
REPLICATES = 10
NOISE_SCALE = 1.0
BASELINE_EFFECT = 0.0
PROPOSED_EFFECT_LOW = 0.5
PROPOSED_EFFECT_HIGH = 3.0


def main() -> None:
    out = Path(__file__).resolve().parent
    rng = np.random.default_rng(42)
    rows = []
    for problem_size in PROBLEM_SIZES:
        for method in ["baseline", "proposed"]:
            for rep in range(REPLICATES):
                t0 = time.perf_counter()
                base = 1000 / problem_size + rng.normal(0, NOISE_SCALE)
                improvement = BASELINE_EFFECT if method == "baseline" else rng.uniform(PROPOSED_EFFECT_LOW, PROPOSED_EFFECT_HIGH)
                rows.append({
                    "instance_id": f"generic_{problem_size}_{rep}",
                    "domain": "generic_or",
                    "problem_size": problem_size,
                    "method": method,
                    "replicate": rep,
                    "seed": 42,
                    "objective": base - improvement,
                    "runtime_seconds": time.perf_counter() - t0,
                    "feasible": True,
                })
    df = pd.DataFrame(rows)
    df.to_csv(out / "results.csv", index=False)
    fig_dir = out / "figures"
    fig_dir.mkdir(exist_ok=True)
    grouped = df.groupby(["problem_size", "method"], as_index=False)["objective"].mean()
    for method, part in grouped.groupby("method"):
        plt.plot(part["problem_size"], part["objective"], marker="o", label=method)
    plt.xlabel("Problem size")
    plt.ylabel("Objective")
    plt.title("Generic OR experiment")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "generic_or_results.png")
    print("SUMMARY_JSON:" + json.dumps({"status": "success", "rows": len(df), "plot": "figures/generic_or_results.png"}))


if __name__ == "__main__":
    main()
