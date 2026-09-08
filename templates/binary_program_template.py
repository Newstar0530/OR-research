"""MI 0-1 binary programming experiment: real instances, real solvers, verified solutions.

Unlike a synthetic scaffold, nothing in this experiment is decided in advance.
For every (size, seed) pair it

1. generates a reproducible 0-1 integer program,
2. establishes a *proved* optimum (exhaustive enumeration, or CP-SAT reporting
   `optimal`) -- and records `None` when no proof was obtained,
3. solves it with each configured method,
4. re-substitutes every returned solution into every constraint independently,
5. writes the recomputed objective, the MIP gap, and the distance to the proved
   optimum.

So `proposed beats baseline` is an outcome that can come out either way, and a
method that returns an infeasible vector is recorded as a defect instead of a
score.

Contract: writes `results.csv`, populates `figures/`, and prints one
`SUMMARY_JSON:` line on stdout.
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

# --- experiment knobs (the autonomous search mutates these) -----------------
INSTANCE_FAMILY = "multi_knapsack"
DIFFICULTY = "weakly_correlated"
PROBLEM_SIZES = [14, 18, 22]
SEED_LIST = [0, 1, 2, 3, 4]
METHODS = ["exact_cp_sat", "greedy", "local_search"]
BASELINE_METHOD = "greedy"
PROPOSED_METHOD = "local_search"
SOLVER_TIME_LIMIT_SECONDS = 5.0
GROUND_TRUTH_TIME_LIMIT_SECONDS = 20.0
BRUTE_FORCE_MAX_VARS = 20
LOCAL_SEARCH_RESTARTS = 3
N_RESOURCES = 3
CAPACITY_RATIO = 0.5
SAVE_INSTANCES = True


def _project_root() -> Path:
    """Find the repository root so `src.core.*` is importable from any workspace.

    A node workspace lives several directories below the project root, so the
    script walks upwards looking for the engine it needs.
    """

    override = os.environ.get("RESEARCH_PROJECT_ROOT")
    if override and (Path(override) / "src" / "core" / "solver_backends.py").exists():
        return Path(override)
    here = Path(__file__).resolve()
    for candidate in [here.parent, *here.parents]:
        if (candidate / "src" / "core" / "solver_backends.py").exists():
            return candidate
    raise RuntimeError(
        "Could not locate the project root (no src/core/solver_backends.py found above "
        f"{here}). Set RESEARCH_PROJECT_ROOT."
    )


ROOT = _project_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.binary_program import generate_instance  # noqa: E402
from src.core.solve_result import summarize_results, write_results_csv  # noqa: E402
from src.core.solver_backends import (  # noqa: E402
    build_method_specs,
    compute_ground_truth,
    describe_backends,
    solve_instance,
)


def _family_kwargs() -> dict:
    if INSTANCE_FAMILY == "multi_knapsack":
        return {"n_resources": N_RESOURCES, "capacity_ratio": CAPACITY_RATIO}
    if INSTANCE_FAMILY == "knapsack":
        return {"capacity_ratio": CAPACITY_RATIO}
    return {}


def _plot_metric_by_size(
    df: pd.DataFrame, metric: str, ylabel: str, title: str, path: Path, log_y: bool = False
) -> bool:
    usable = df[df[metric].notna()]
    if usable.empty:
        return False
    grouped = usable.groupby(["problem_size", "method"], as_index=False)[metric].mean()
    plt.figure(figsize=(6.5, 4))
    for method, part in grouped.groupby("method"):
        part = part.sort_values("problem_size")
        plt.plot(part["problem_size"], part[metric], marker="o", label=str(method))
    if log_y:
        plt.yscale("log")
    plt.xlabel("Number of binary variables")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return True


def main() -> int:
    out = Path(__file__).resolve().parent
    figures = out / "figures"
    figures.mkdir(exist_ok=True)

    specs = build_method_specs(
        brute_force_max_vars=BRUTE_FORCE_MAX_VARS,
        local_search_restarts=LOCAL_SEARCH_RESTARTS,
    )
    backend_report = describe_backends(specs)
    (out / "solver_backend_report.json").write_text(
        json.dumps(backend_report, indent=2), encoding="utf-8"
    )

    unavailable = [row["method"] for row in backend_report if not row["available"]]
    requested_unavailable = [m for m in METHODS if m in unavailable]

    results = []
    instances_dir = out / "instances"
    if SAVE_INSTANCES:
        instances_dir.mkdir(exist_ok=True)

    for size in PROBLEM_SIZES:
        for seed in SEED_LIST:
            instance = generate_instance(
                INSTANCE_FAMILY, size, seed=seed, difficulty=DIFFICULTY, **_family_kwargs()
            )
            instance = compute_ground_truth(
                instance,
                time_limit=GROUND_TRUTH_TIME_LIMIT_SECONDS,
                brute_force_max_vars=BRUTE_FORCE_MAX_VARS,
                specs=specs,
            )
            if SAVE_INSTANCES:
                instance.to_json_file(instances_dir / f"{instance.instance_id}.json")
            for method in METHODS:
                results.append(
                    solve_instance(
                        instance,
                        method,
                        time_limit=SOLVER_TIME_LIMIT_SECONDS,
                        seed=seed,
                        specs=specs,
                    )
                )

    write_results_csv(results, out / "results.csv")
    df = pd.DataFrame([r.to_row() for r in results])
    trustworthy = [r for r in results if r.is_trustworthy]
    failures = [r for r in results if not r.verification_ok]

    plots: list[str] = []
    trust_df = df[df["verification_ok"] & df["feasible"]]
    if _plot_metric_by_size(
        trust_df,
        "gap_to_known_optimum",
        "Mean relative gap to proved optimum",
        f"Solution quality ({INSTANCE_FAMILY}, {DIFFICULTY})",
        figures / "gap_to_optimum_by_size.png",
    ):
        plots.append("figures/gap_to_optimum_by_size.png")
    if _plot_metric_by_size(
        trust_df,
        "runtime_seconds",
        "Mean runtime (s, log scale)",
        f"Solve time ({INSTANCE_FAMILY}, {DIFFICULTY})",
        figures / "runtime_by_size.png",
        log_y=True,
    ):
        plots.append("figures/runtime_by_size.png")

    lines = [
        "# Solution Verification Report",
        "",
        f"- instance family: `{INSTANCE_FAMILY}` (difficulty `{DIFFICULTY}`)",
        f"- rows: {len(results)}",
        f"- trustworthy rows (claimed a solution, verified feasible): {len(trustworthy)}",
        f"- verification failures: {len(failures)}",
        f"- instances with a proved optimum: "
        f"{int(df['known_optimum'].notna().sum())} / {len(df)} rows",
        "",
        "## Requested Methods That Were Unavailable",
    ]
    if requested_unavailable:
        lines.extend(
            f"- `{method}`: "
            + next(r["unavailable_reason"] for r in backend_report if r["method"] == method)
            for method in requested_unavailable
        )
    else:
        lines.append("- None.")
    lines.append("")
    lines.append("## Verification Failures")
    if not failures:
        lines.append("- None.")
    for result in failures[:30]:
        lines.append(f"- `{result.instance_id}` / `{result.method}`: {result.notes}")
    (out / "verification_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    summary = summarize_results(results)
    summary.update(
        {
            "instance_family": INSTANCE_FAMILY,
            "difficulty": DIFFICULTY,
            "problem_sizes": PROBLEM_SIZES,
            "seeds": SEED_LIST,
            "requested_methods": METHODS,
            "unavailable_requested_methods": requested_unavailable,
            "baseline_method": BASELINE_METHOD,
            "proposed_method": PROPOSED_METHOD,
            "primary_metric": "gap_to_known_optimum",
            "objective_direction": "minimize",
            "instances_with_proved_optimum": int(df["known_optimum"].notna().sum()),
            "plots": plots,
        }
    )

    if failures:
        print(
            f"VERIFICATION_FAILURE: {len(failures)} row(s) failed independent verification. "
            "See verification_report.md.",
            file=sys.stderr,
        )
    print("SUMMARY_JSON:" + json.dumps(summary, default=str))
    return 0 if trustworthy else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:  # surface the real traceback for the debug loop
        traceback.print_exc()
        print('SUMMARY_JSON:{"status": "error", "rows": 0}')
        raise SystemExit(1)
