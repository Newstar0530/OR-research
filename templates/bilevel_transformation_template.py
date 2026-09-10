"""Does the bilevel -> KKT -> MI 0-1 chain still solve the same problem?

For every generated bilevel program this experiment establishes the true
optimum with a method that shares nothing with the transformation under test
(vertex enumeration plus a direct solve of the follower's LP), then checks each
rewrite against it:

1. `bilevel_to_kkt`      - solved by enumerating complementarity patterns, so no
                           Big-M constant is involved. This step has nothing to
                           tune: if it disagrees with the oracle, the KKT
                           derivation itself is wrong.
2. `mi01_derived_bigM`   - the Fortuny-Amat linearisation with constants derived
                           rigorously (interval arithmetic for slacks, dual
                           vertex enumeration for multipliers).
3. control conditions    - the same linearisation with deliberately scaled or
                           flat Big-M values, to locate the threshold below
                           which the reformulation stops being exact and to test
                           what the folklore "just set M large" actually does.

Only steps 1 and 2 can fail the run. The control conditions are *supposed* to
break; that is the measurement.

A row's `solver_status` describes the bilevel problem, not the rewritten model:
a model that solves happily but returns a point the follower would never choose
did not produce a solution to the question being asked, and is recorded as
`no_solution` with the verdict in `equivalence_verdict`.

Contract: writes `results.csv`, populates `figures/`, prints one `SUMMARY_JSON:`
line on stdout.
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
INSTANCE_SIZES = [2, 3]
SEED_LIST = [0, 1, 2]
DIFFICULTIES = ["balanced", "coupled", "degenerate", "large_dual"]
BIG_M_SCALES = [0.1, 0.5, 1.0, 10.0]
UNIFORM_BIG_M_VALUES = [1000.0, 1000000.0]
SOLVER_TIME_LIMIT_SECONDS = 30.0
ORACLE_MAX_COMBINATIONS = 400000
MAX_COMPLEMENTARITY_PATTERNS = 65536
SAVE_INSTANCES = True

DERIVED_METHOD = "mi01_derived_bigM"
KKT_METHOD = "kkt_pattern_enumeration"


def _project_root() -> Path:
    override = os.environ.get("RESEARCH_PROJECT_ROOT")
    if override and (Path(override) / "src" / "core" / "equivalence.py").exists():
        return Path(override)
    here = Path(__file__).resolve()
    for candidate in [here.parent, *here.parents]:
        if (candidate / "src" / "core" / "equivalence.py").exists():
            return candidate
    raise RuntimeError(
        f"Could not locate the project root above {here}. Set RESEARCH_PROJECT_ROOT."
    )


ROOT = _project_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.bilevel import generate_bilevel_lp  # noqa: E402
from src.core.bilevel_oracle import solve_bilevel_by_vertex_enumeration  # noqa: E402
from src.core.equivalence import verify_transformation_chain  # noqa: E402
from src.core.solve_result import SolveResult, summarize_results, write_results_csv  # noqa: E402
from src.core.transformations import derive_big_m_bounds, to_kkt_one_level  # noqa: E402


def _status_for(step) -> str:
    """Status relative to the bilevel problem, not to the rewritten model."""

    if step.verdict == "model_infeasible":
        return "infeasible"
    if step.verdict in ("not_run", "reference_unavailable"):
        return "not_run"
    if not step.solution_verified:
        # It solved something, but not the problem we asked about.
        return "no_solution"
    return "optimal" if step.verdict == "equivalent" else "feasible"


def _row(
    blp,
    step,
    method: str,
    oracle_value: float | None,
    n_binaries: int,
    is_control: bool,
    big_m_label: str,
    slack_m: float,
    dual_m: float,
    backends_disagreed: bool,
) -> SolveResult:
    verified = bool(step.solution_verified)
    result = SolveResult(
        instance_id=blp.instance_id,
        method=method,
        solver_backend=step.backend or "n/a",
        instance_family=f"bilevel_{blp.difficulty}",
        seed=blp.seed,
        problem_size=blp.n_x + blp.n_y,
        n_vars=blp.n_x + blp.n_y,
        n_constraints=blp.n_upper_rows + blp.n_lower_rows,
        objective_sense="minimize" if blp.upper_sense == "minimize" else "maximize",
        solver_status=_status_for(step),
        objective=step.model_value,
        runtime_seconds=step.runtime_seconds,
        feasible=verified,
        # Only the two load-bearing conditions can invalidate the run; the
        # control conditions exist precisely to fail.
        verification_ok=True if is_control else (step.verdict == "equivalent" and verified),
        known_optimum=oracle_value,
        known_optimum_source="vertex_enumeration_proved",
        extra={
            "equivalence_verdict": step.verdict,
            "transformation_step": step.step,
            "big_m_setting": big_m_label,
            "big_m_slack": slack_m,
            "big_m_dual": dual_m,
            "n_binaries": n_binaries,
            "n_multipliers": n_binaries,
            "is_control_condition": is_control,
            "solution_bilevel_feasible": verified,
            "backends_disagreed": backends_disagreed,
            "difficulty": blp.difficulty,
        },
    )
    if step.notes:
        result.add_note(step.notes)
    return result.with_derived_metrics()


def main() -> int:
    out = Path(__file__).resolve().parent
    figures = out / "figures"
    figures.mkdir(exist_ok=True)
    instances_dir = out / "instances"
    if SAVE_INSTANCES:
        instances_dir.mkdir(exist_ok=True)

    results: list[SolveResult] = []
    chain_reports: list[str] = []
    oracle_unavailable = 0

    for size in INSTANCE_SIZES:
        for difficulty in DIFFICULTIES:
            for seed in SEED_LIST:
                blp = generate_bilevel_lp(size, seed=seed, difficulty=difficulty)
                if SAVE_INSTANCES:
                    (instances_dir / f"{blp.instance_id}.json").write_text(
                        blp.model_dump_json(indent=2), encoding="utf-8"
                    )
                oracle = solve_bilevel_by_vertex_enumeration(
                    blp, max_combinations=ORACLE_MAX_COMBINATIONS
                )
                if oracle.status != "optimal":
                    oracle_unavailable += 1
                    continue
                kkt = to_kkt_one_level(blp)
                bounds = derive_big_m_bounds(blp)

                conditions: list[tuple[str, str, object, object, bool]] = [
                    (DERIVED_METHOD, "derived", bounds.slack_bounds, bounds.dual_bounds, False)
                ]
                for scale in BIG_M_SCALES:
                    conditions.append(
                        (
                            f"mi01_scale_{scale:g}",
                            f"derived x {scale:g}",
                            [v * scale for v in bounds.slack_bounds],
                            [v * scale for v in bounds.dual_bounds],
                            True,
                        )
                    )
                for value in UNIFORM_BIG_M_VALUES:
                    conditions.append(
                        (f"mi01_uniform_{value:g}", f"uniform {value:g}", value, value, True)
                    )

                first = True
                for method, label, slack_m, dual_m, is_control in conditions:
                    report = verify_transformation_chain(
                        blp,
                        slack_big_m=slack_m,
                        dual_big_m=dual_m,
                        oracle=oracle,
                        kkt=kkt,
                        bounds=bounds,
                        time_limit=SOLVER_TIME_LIMIT_SECONDS,
                        max_patterns=MAX_COMPLEMENTARITY_PATTERNS,
                    )
                    step1 = report.step("bilevel_to_kkt")
                    step2 = report.step("kkt_to_mi01")
                    if first:
                        chain_reports.append(report.to_markdown())
                        results.append(
                            _row(
                                blp,
                                step1,
                                KKT_METHOD,
                                oracle.upper_objective,
                                report.n_lambda,
                                is_control=False,
                                big_m_label="none (pattern enumeration)",
                                slack_m=float("nan"),
                                dual_m=float("nan"),
                                backends_disagreed=False,
                            )
                        )
                        first = False
                    disagreed = "CROSS-CHECK FAILURE" in (step2.notes or "")
                    scalar_slack = float(slack_m) if isinstance(slack_m, (int, float)) else float(max(slack_m))
                    scalar_dual = float(dual_m) if isinstance(dual_m, (int, float)) else float(max(dual_m))
                    results.append(
                        _row(
                            blp,
                            step2,
                            method,
                            oracle.upper_objective,
                            report.n_binaries,
                            is_control,
                            label,
                            scalar_slack,
                            scalar_dual,
                            disagreed,
                        )
                    )

    write_results_csv(results, out / "results.csv")
    df = pd.DataFrame([r.to_row() for r in results])

    plots: list[str] = []
    controls = df[df["is_control_condition"] == True]  # noqa: E712
    if not controls.empty:
        share = (
            controls.assign(ok=(controls["equivalence_verdict"] == "equivalent").astype(float))
            .groupby("big_m_setting", as_index=False)["ok"]
            .mean()
            .sort_values("ok")
        )
        plt.figure(figsize=(7, 4))
        plt.bar(share["big_m_setting"].astype(str), share["ok"] * 100.0, color="#4c72b0")
        plt.ylabel("Instances where the optimum survives (%)")
        plt.xlabel("Big-M setting")
        plt.title("Big-M threshold for exactness of the MI 0-1 reformulation")
        plt.xticks(rotation=30, ha="right")
        plt.ylim(0, 105)
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(figures / "bigm_equivalence_by_setting.png", dpi=150)
        plt.close()
        plots.append("figures/bigm_equivalence_by_setting.png")

    usable = df[df["gap_to_known_optimum"].notna() & (df["feasible"] == True)]  # noqa: E712
    if not usable.empty:
        grouped = usable.groupby(["problem_size", "method"], as_index=False)[
            "gap_to_known_optimum"
        ].mean()
        plt.figure(figsize=(7, 4))
        for method, part in grouped.groupby("method"):
            part = part.sort_values("problem_size")
            plt.plot(part["problem_size"], part["gap_to_known_optimum"], marker="o", label=str(method))
        plt.xlabel("n_x + n_y")
        plt.ylabel("Mean relative gap to the true optimum")
        plt.title("Reformulation accuracy by problem size")
        plt.legend(fontsize=8)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(figures / "gap_by_size.png", dpi=150)
        plt.close()
        plots.append("figures/gap_by_size.png")

    verdicts = df["equivalence_verdict"].value_counts().to_dict()
    load_bearing = df[df["is_control_condition"] == False]  # noqa: E712
    failures = load_bearing[load_bearing["verification_ok"] == False]  # noqa: E712

    lines = [
        "# Transformation Equivalence Report",
        "",
        f"- instances: {df['instance_id'].nunique()}",
        f"- rows: {len(df)} ({int((df['is_control_condition'] == True).sum())} control)",  # noqa: E712
        f"- oracle unavailable on {oracle_unavailable} instance(s)",
        "",
        "## Verdicts",
        "",
        "| verdict | rows |",
        "| --- | --- |",
    ]
    lines.extend(f"| {k} | {v} |" for k, v in sorted(verdicts.items(), key=lambda kv: -kv[1]))
    lines.extend(
        [
            "",
            "## Load-Bearing Conditions",
            "",
            f"- `{KKT_METHOD}` and `{DERIVED_METHOD}` must be equivalent on every instance.",
            f"- failures: **{len(failures)}**",
            "",
            "## Big-M Setting vs Exactness",
            "",
            "| setting | equivalent | total | share |",
            "| --- | --- | --- | --- |",
        ]
    )
    if not controls.empty:
        for setting, part in controls.groupby("big_m_setting"):
            ok = int((part["equivalence_verdict"] == "equivalent").sum())
            lines.append(f"| {setting} | {ok} | {len(part)} | {ok / len(part):.0%} |")
    lines.extend(["", "## Per-Instance Chain Reports", ""])
    (out / "equivalence_report.md").write_text(
        "\n".join(lines) + "\n\n" + "\n\n---\n\n".join(chain_reports) + "\n", encoding="utf-8"
    )

    summary = summarize_results(results)
    derived = df[df["method"] == DERIVED_METHOD]
    summary.update(
        {
            "instances": int(df["instance_id"].nunique()),
            "sizes": INSTANCE_SIZES,
            "seeds": SEED_LIST,
            "difficulties": DIFFICULTIES,
            "big_m_scales": BIG_M_SCALES,
            "uniform_big_m_values": UNIFORM_BIG_M_VALUES,
            "verdict_counts": verdicts,
            "load_bearing_failures": int(len(failures)),
            "oracle_unavailable": oracle_unavailable,
            "derived_bigM_equivalent_share": (
                float((derived["equivalence_verdict"] == "equivalent").mean())
                if not derived.empty
                else None
            ),
            "backends_disagreed_rows": int(df["backends_disagreed"].sum()),
            "primary_metric": "gap_to_known_optimum",
            "objective_direction": "minimize",
            "plots": plots,
        }
    )
    if len(failures):
        print(
            f"TRANSFORMATION FAILURE: {len(failures)} load-bearing row(s) did not preserve the "
            "optimum. See equivalence_report.md.",
            file=sys.stderr,
        )
    print("SUMMARY_JSON:" + json.dumps(summary, default=str))
    return 0 if len(df) else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        print('SUMMARY_JSON:{"status": "error", "rows": 0}')
        raise SystemExit(1)
