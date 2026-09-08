from __future__ import annotations

import heapq
import json
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


N_NODES_LIST = [20, 40, 80]
EDGE_DENSITIES = [0.08, 0.14, 0.22]
REPLICATES = 5
EDGE_WEIGHT_MAX = 20


def shortest_path(n, edges, source, target):
    graph = [[] for _ in range(n)]
    for u, v, w in edges:
        graph[u].append((v, w))
        graph[v].append((u, w))
    dist = [float("inf")] * n
    dist[source] = 0
    pq = [(0, source)]
    while pq:
        d, u = heapq.heappop(pq)
        if u == target:
            return d
        if d != dist[u]:
            continue
        for v, w in graph[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(pq, (nd, v))
    return float("inf")


def main() -> None:
    out = Path(__file__).resolve().parent
    rng = random.Random(42)
    rows = []
    for n in N_NODES_LIST:
        for density in EDGE_DENSITIES:
            for rep in range(REPLICATES):
                edges = []
                for i in range(n):
                    for j in range(i + 1, n):
                        if rng.random() < density:
                            edges.append((i, j, rng.randint(1, EDGE_WEIGHT_MAX)))
                t0 = time.perf_counter()
                cost = shortest_path(n, edges, 0, n - 1)
                rows.append({
                    "domain": "network_analysis",
                    "instance_id": f"net_{n}_{density}_{rep}",
                    "n_nodes": n,
                    "problem_size": n,
                    "density": density,
                    "replicate": rep,
                    "seed": 42,
                    "method": "dijkstra_shortest_path",
                    "objective": cost,
                    "path_cost": cost,
                    "n_edges": len(edges),
                    "feasible": cost < float("inf"),
                    "runtime_seconds": time.perf_counter() - t0,
                })
    df = pd.DataFrame(rows)
    df.to_csv(out / "results.csv", index=False)
    fig_dir = out / "figures"
    fig_dir.mkdir(exist_ok=True)
    grouped = df.replace(float("inf"), None).dropna().groupby("density", as_index=False)["path_cost"].mean()
    plt.plot(grouped["density"], grouped["path_cost"], marker="o")
    plt.xlabel("Edge density")
    plt.ylabel("Mean shortest-path cost")
    plt.title("Network sensitivity")
    plt.tight_layout()
    plt.savefig(fig_dir / "network_shortest_path.png")
    print("SUMMARY_JSON:" + json.dumps({"status": "success", "rows": len(df), "plot": "figures/network_shortest_path.png"}))


if __name__ == "__main__":
    main()
