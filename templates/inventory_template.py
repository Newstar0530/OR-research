from __future__ import annotations

import json
import math
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEMAND_SIGMAS = [5, 15, 30]
SHORTAGE_COSTS = [5, 20, 50]
DEMAND_SAMPLE_SIZE = 5000
ORDER_QUANTITY_START = 60
ORDER_QUANTITY_STOP = 151
ORDER_QUANTITY_STEP = 10


def main() -> None:
    out = Path(__file__).resolve().parent
    rng = np.random.default_rng(42)
    rows = []
    for demand_sigma in DEMAND_SIGMAS:
        for shortage_cost in SHORTAGE_COSTS:
            demand = rng.normal(100, demand_sigma, size=DEMAND_SAMPLE_SIZE).clip(min=0)
            for q in range(ORDER_QUANTITY_START, ORDER_QUANTITY_STOP, ORDER_QUANTITY_STEP):
                t0 = time.perf_counter()
                holding = np.maximum(q - demand, 0).mean()
                shortage = np.maximum(demand - q, 0).mean()
                cost = holding + shortage_cost * shortage
                rows.append(
                    {
                        "domain": "inventory",
                        "instance_id": f"inv_{demand_sigma}_{shortage_cost}_{q}",
                        "demand_sigma": demand_sigma,
                        "shortage_cost": shortage_cost,
                        "order_quantity": q,
                        "problem_size": len(demand),
                        "method": "order_quantity_policy",
                        "seed": 42,
                        "objective": cost,
                        "mean_cost": cost,
                        "service_level": float((demand <= q).mean()),
                        "feasible": True,
                        "runtime_seconds": time.perf_counter() - t0,
                    }
                )
    df = pd.DataFrame(rows)
    best = df.loc[df.groupby(["demand_sigma", "shortage_cost"])["mean_cost"].idxmin()]
    df.to_csv(out / "results.csv", index=False)
    fig_dir = out / "figures"
    fig_dir.mkdir(exist_ok=True)
    plt.scatter(best["shortage_cost"], best["order_quantity"], c=best["demand_sigma"])
    plt.xlabel("Shortage cost")
    plt.ylabel("Best order quantity")
    plt.title("Inventory sensitivity")
    plt.tight_layout()
    plt.savefig(fig_dir / "inventory_policy.png")
    print("SUMMARY_JSON:" + json.dumps({"status": "success", "rows": len(df), "plot": "figures/inventory_policy.png"}))


if __name__ == "__main__":
    main()
