from __future__ import annotations

import json
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


N_JOBS_LIST = [10, 20, 40]
DUE_DATE_TIGHTNESS = [0.7, 1.0, 1.3]
REPLICATES = 5
PROCESSING_TIME_MAX = 20


def evaluate(order, processing, due_dates):
    t = 0
    tardiness = 0
    completion = []
    for job in order:
        t += processing[job]
        completion.append(t)
        tardiness += max(0, t - due_dates[job])
    return max(completion), tardiness


def main() -> None:
    out = Path(__file__).resolve().parent
    rng = random.Random(42)
    rows = []
    for n_jobs in N_JOBS_LIST:
        for tightness in DUE_DATE_TIGHTNESS:
            for rep in range(REPLICATES):
                processing = [rng.randint(1, PROCESSING_TIME_MAX) for _ in range(n_jobs)]
                due_dates = [int(sum(processing) * tightness * rng.uniform(0.3, 1.0)) for _ in range(n_jobs)]
                methods = {
                    "SPT": sorted(range(n_jobs), key=lambda j: processing[j]),
                    "EDD": sorted(range(n_jobs), key=lambda j: due_dates[j]),
                    "LPT": sorted(range(n_jobs), key=lambda j: processing[j], reverse=True),
                }
                for method, order in methods.items():
                    t0 = time.perf_counter()
                    makespan, tardiness = evaluate(order, processing, due_dates)
                    rows.append(
                        {
                            "domain": "scheduling",
                            "instance_id": f"sched_{n_jobs}_{tightness}_{rep}",
                            "n_jobs": n_jobs,
                            "problem_size": n_jobs,
                            "tightness": tightness,
                            "replicate": rep,
                            "seed": 42,
                            "method": method,
                            "objective": tardiness,
                            "makespan": makespan,
                            "total_tardiness": tardiness,
                            "feasible": True,
                            "runtime_seconds": time.perf_counter() - t0,
                        }
                    )
    df = pd.DataFrame(rows)
    df.to_csv(out / "results.csv", index=False)
    fig_dir = out / "figures"
    fig_dir.mkdir(exist_ok=True)
    plot = df.groupby(["n_jobs", "method"], as_index=False)["total_tardiness"].mean()
    for method, part in plot.groupby("method"):
        plt.plot(part["n_jobs"], part["total_tardiness"], marker="o", label=method)
    plt.xlabel("Number of jobs")
    plt.ylabel("Mean total tardiness")
    plt.title("Scheduling dispatching-rule comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "scheduling_tardiness.png")
    print("SUMMARY_JSON:" + json.dumps({"status": "success", "rows": len(df), "plot": "figures/scheduling_tardiness.png"}))


if __name__ == "__main__":
    main()
