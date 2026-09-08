from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ARRIVAL_RATES = [0.6, 0.8, 0.95]
SERVICE_RATE = 1.0
REPLICATES = 20
SIMULATION_HORIZON = 500


def main() -> None:
    out = Path(__file__).resolve().parent
    rng = np.random.default_rng(42)
    rows = []
    for arrival_rate in ARRIVAL_RATES:
        service_rate = SERVICE_RATE
        for rep in range(REPLICATES):
            t0 = time.perf_counter()
            interarrival = rng.exponential(1 / arrival_rate, size=SIMULATION_HORIZON)
            service = rng.exponential(1 / service_rate, size=SIMULATION_HORIZON)
            arrivals = np.cumsum(interarrival)
            finish = 0.0
            waits = []
            for a, s in zip(arrivals, service):
                start = max(a, finish)
                waits.append(start - a)
                finish = start + s
            mean_wait = float(np.mean(waits))
            rows.append({
                "domain": "stochastic_models",
                "instance_id": f"queue_{arrival_rate}_{rep}",
                "arrival_rate": arrival_rate,
                "problem_size": SIMULATION_HORIZON,
                "replicate": rep,
                "seed": 42,
                "method": "single_server_simulation",
                "objective": mean_wait,
                "mean_wait": mean_wait,
                "utilization": arrival_rate / service_rate,
                "feasible": True,
                "runtime_seconds": time.perf_counter() - t0,
            })
    df = pd.DataFrame(rows)
    df.to_csv(out / "results.csv", index=False)
    fig_dir = out / "figures"
    fig_dir.mkdir(exist_ok=True)
    grouped = df.groupby("arrival_rate", as_index=False)["mean_wait"].mean()
    plt.plot(grouped["arrival_rate"], grouped["mean_wait"], marker="o")
    plt.xlabel("Arrival rate")
    plt.ylabel("Mean waiting time")
    plt.title("Queueing sensitivity")
    plt.tight_layout()
    plt.savefig(fig_dir / "queue_waiting_time.png")
    print("SUMMARY_JSON:" + json.dumps({"status": "success", "rows": len(df), "plot": "figures/queue_waiting_time.png"}))


if __name__ == "__main__":
    main()
