"""Does the QUBO encoding still answer the original question, and at what price?

This is the last hop before a quantum sampler, and it spends two things that
fail in different ways, so the study measures them separately:

* **discretisation** - a continuous optimum that does not sit on the bit grid is
  simply unreachable. A cost, quantified, not a defect.
* **the penalty weight** - too small and the ground state is an infeasible
  bitstring that a sampler will report as the answer with no sign of trouble.

There is a third thing worth measuring that is nobody's bug: **how many bits the
encoding costs**. Every inequality needs its own slack register, so the bit count
grows with the number of constraints, and beyond roughly twenty bits nobody can
prove a ground state by enumeration any more. Where that happens the study says
`not_proved` rather than quietly accepting an annealer's best draw as the
optimum.

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
INSTANCE_FAMILIES = ["knapsack", "multi_knapsack", "set_cover"]
INSTANCE_SIZES = [6, 8]
SEED_LIST = [0, 1, 2]
PENALTY_SCALES = [0.001, 0.01, 1.0]
PRECISION = 1.0
MAX_BITS_FOR_PROOF = 22
ANNEALER_SEED = 0
SOLVER_TIME_LIMIT_SECONDS = 60.0
SAVE_INSTANCES = True

DERIVED_METHOD = "qubo_derived_penalty"

#: Verdicts that mean the encoding is wrong, as opposed to costly or unproved.
DEFECT_VERDICTS = {
    "penalty_too_small",
    "qubo_better_than_grid",
    "grid_better_than_true",
    "energy_encoding_mismatch",
    "error",
}


def _project_root() -> Path:
    override = os.environ.get("RESEARCH_PROJECT_ROOT")
    if override and (Path(override) / "src" / "core" / "qubo_equivalence.py").exists():
        return Path(override)
    here = Path(__file__).resolve()
    for candidate in [here.parent, *here.parents]:
        if (candidate / "src" / "core" / "qubo_equivalence.py").exists():
            return candidate
    raise RuntimeError(f"Could not locate the project root above {here}.")


ROOT = _project_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.binary_program import generate_instance  # noqa: E402
from src.core.mixed_integer_program import (  # noqa: E402
    MipConstraint,
    MipVariable,
    MixedIntegerProgram,
    solve_mip,
)
from src.core.qubo_equivalence import verify_qubo_chain  # noqa: E402
from src.core.qubo_transformations import (  # noqa: E402
    derive_penalty_bound,
    to_binary_grid_mip,
)
from src.core.solve_result import SolveResult, summarize_results, write_results_csv  # noqa: E402


def as_mip(instance) -> MixedIntegerProgram:
    """A binary program in the general mixed-integer representation."""

    return MixedIntegerProgram(
        name=instance.instance_id,
        sense=instance.objective_sense,
        variables=[MipVariable(name=f"x[{j}]", is_binary=True) for j in range(instance.n_vars)],
        objective=[float(c) for c in instance.objective_coefficients],
        constraints=[
            MipConstraint(
                name=c.name,
                coefficients=[float(a) for a in c.coefficients],
                sense=c.sense,
                rhs=float(c.rhs),
            )
            for c in instance.constraints
        ],
    )


def _status_for(report, step) -> str:
    """Status relative to the original problem, not to the QUBO."""

    if step is None or step.verdict in ("not_run",):
        return "not_run"
    if step.verdict in DEFECT_VERDICTS:
        # It produced a bitstring, but not a solution to the question asked.
        return "no_solution"
    if not step.solution_verified:
        return "no_solution"
    if step.verdict == "equivalent" and report.ground_state_proved:
        return "optimal"
    return "feasible"


def main() -> int:
    out = Path(__file__).resolve().parent
    figures = out / "figures"
    figures.mkdir(exist_ok=True)
    instances_dir = out / "instances"
    if SAVE_INSTANCES:
        instances_dir.mkdir(exist_ok=True)

    results: list[SolveResult] = []
    reports: list[str] = []

    for family in INSTANCE_FAMILIES:
        for size in INSTANCE_SIZES:
            for seed in SEED_LIST:
                instance = generate_instance(family, size, seed=seed)
                if SAVE_INSTANCES:
                    instance.to_json_file(instances_dir / f"{instance.instance_id}.json")
                mip = as_mip(instance)
                true_solution = solve_mip(mip, time_limit=SOLVER_TIME_LIMIT_SECONDS)
                if true_solution.status not in ("optimal", "feasible"):
                    continue
                grid_mip, _ = to_binary_grid_mip(mip, precision=PRECISION)
                bound = derive_penalty_bound(grid_mip)

                conditions = [(DERIVED_METHOD, "derived", bound.penalty, False)]
                for scale in PENALTY_SCALES:
                    if scale == 1.0:
                        continue
                    conditions.append(
                        (
                            f"qubo_scale_{scale:g}",
                            f"derived x {scale:g}",
                            bound.penalty * scale,
                            True,
                        )
                    )

                first = True
                for method, label, penalty, is_control in conditions:
                    report = verify_qubo_chain(
                        mip,
                        penalty=penalty,
                        precision=PRECISION,
                        max_bits_for_proof=MAX_BITS_FOR_PROOF,
                        time_limit=SOLVER_TIME_LIMIT_SECONDS,
                        seed=ANNEALER_SEED,
                        true_solution=true_solution,
                    )
                    if first:
                        reports.append(report.to_markdown())
                        first = False
                    step = report.step("binary_grid_to_qubo")
                    grid_step = report.step("mip_to_binary_grid")
                    verdict = step.verdict if step else "not_run"
                    is_defect = verdict in DEFECT_VERDICTS
                    result = SolveResult(
                        instance_id=instance.instance_id,
                        method=method,
                        solver_backend=(step.backend if step else "n/a"),
                        instance_family=f"qubo_{family}",
                        seed=seed,
                        problem_size=instance.n_vars,
                        n_vars=instance.n_vars,
                        n_constraints=instance.n_constraints,
                        objective_sense=instance.objective_sense,
                        solver_status=_status_for(report, step),
                        objective=report.qubo_value,
                        runtime_seconds=(step.runtime_seconds if step else 0.0),
                        feasible=bool(step and step.solution_verified),
                        verification_ok=True if is_control else not is_defect,
                        known_optimum=report.true_value,
                        known_optimum_source="mip_optimum_cross_checked",
                        extra={
                            "qubo_verdict": verdict,
                            "grid_verdict": grid_step.verdict if grid_step else "not_run",
                            "penalty": penalty,
                            "penalty_setting": label,
                            "penalty_to_objective_ratio": report.penalty_to_objective_ratio,
                            "n_bits": report.n_bits,
                            "n_grid_bits": report.n_grid_bits,
                            "n_slack_bits": report.n_slack_bits,
                            "qubo_density": report.qubo_density,
                            "qubo_dynamic_range": report.qubo_dynamic_range,
                            "ground_state_proved": report.ground_state_proved,
                            "discretisation_loss": report.discretisation_loss,
                            "grid_value": report.grid_value,
                            "is_control_condition": is_control,
                            "is_defect": is_defect,
                            "family": family,
                        },
                    )
                    if step and step.notes:
                        result.add_note(step.notes)
                    results.append(result.with_derived_metrics())

    write_results_csv(results, out / "results.csv")
    df = pd.DataFrame([r.to_row() for r in results])

    plots: list[str] = []
    derived = df[df["method"] == DERIVED_METHOD]
    if not derived.empty:
        grouped = derived.groupby(["family", "n_vars"], as_index=False)["n_bits"].mean()
        plt.figure(figsize=(7, 4))
        for family, part in grouped.groupby("family"):
            part = part.sort_values("n_vars")
            plt.plot(part["n_vars"], part["n_bits"], marker="o", label=str(family))
        plt.axhline(
            MAX_BITS_FOR_PROOF,
            color="crimson",
            linestyle="--",
            label=f"exhaustive proof limit ({MAX_BITS_FOR_PROOF} bits)",
        )
        plt.xlabel("Binary variables in the source problem")
        plt.ylabel("Bits required by the QUBO")
        plt.title("Cost of the QUBO encoding: one slack register per inequality")
        plt.legend(fontsize=8)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(figures / "qubo_bits_by_size.png", dpi=150)
        plt.close()
        plots.append("figures/qubo_bits_by_size.png")

    controls = df[df["is_control_condition"] == True]  # noqa: E712
    if not controls.empty:
        share = (
            controls.assign(ok=(~controls["is_defect"].astype(bool)).astype(float))
            .groupby("penalty_setting", as_index=False)["ok"]
            .mean()
        )
        plt.figure(figsize=(7, 4))
        plt.bar(share["penalty_setting"].astype(str), share["ok"] * 100.0, color="#4c72b0")
        plt.ylabel("Encodings that stayed sound (%)")
        plt.xlabel("Penalty setting")
        plt.title("Penalty threshold for a correct QUBO encoding")
        plt.ylim(0, 105)
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(figures / "penalty_threshold.png", dpi=150)
        plt.close()
        plots.append("figures/penalty_threshold.png")

    load_bearing = df[df["is_control_condition"] == False]  # noqa: E712
    defects = load_bearing[load_bearing["is_defect"] == True]  # noqa: E712
    verdicts = df["qubo_verdict"].value_counts().to_dict()
    provable = derived[derived["ground_state_proved"] == True]  # noqa: E712

    lines = [
        "# QUBO Encoding Report",
        "",
        f"- instances: {df['instance_id'].nunique()}",
        f"- rows: {len(df)}",
        f"- load-bearing defects: **{len(defects)}**",
        f"- ground state provable by exhaustive sweep on "
        f"{len(provable)} / {len(derived)} derived-penalty rows",
        "",
        "## Verdicts",
        "",
        "| verdict | rows |",
        "| --- | --- |",
    ]
    lines.extend(f"| {k} | {v} |" for k, v in sorted(verdicts.items(), key=lambda kv: -kv[1]))
    lines.extend(["", "## Encoding Cost", "", "| family | vars | constraints | bits | slack bits | provable |", "| --- | --- | --- | --- | --- | --- |"])
    if not derived.empty:
        cost = derived.groupby(["family", "n_vars"], as_index=False).agg(
            constraints=("n_constraints", "mean"),
            bits=("n_bits", "mean"),
            slack=("n_slack_bits", "mean"),
            provable=("ground_state_proved", "mean"),
        )
        for row in cost.itertuples(index=False):
            lines.append(
                f"| {row.family} | {row.n_vars:.0f} | {row.constraints:.0f} | {row.bits:.0f} | "
                f"{row.slack:.0f} | {row.provable:.0%} |"
            )
    lines.extend(["", "## Per-Instance Reports", ""])
    (out / "qubo_report.md").write_text(
        "\n".join(lines) + "\n\n" + "\n\n---\n\n".join(reports) + "\n", encoding="utf-8"
    )

    summary = summarize_results(results)
    summary.update(
        {
            "instances": int(df["instance_id"].nunique()),
            "families": INSTANCE_FAMILIES,
            "sizes": INSTANCE_SIZES,
            "seeds": SEED_LIST,
            "penalty_scales": PENALTY_SCALES,
            "precision": PRECISION,
            "verdict_counts": verdicts,
            "load_bearing_defects": int(len(defects)),
            "derived_penalty_sound_share": (
                float((~derived["is_defect"].astype(bool)).mean()) if not derived.empty else None
            ),
            "ground_state_proved_share": (
                float(derived["ground_state_proved"].astype(bool).mean())
                if not derived.empty
                else None
            ),
            "max_bits_observed": int(df["n_bits"].max()) if not df.empty else 0,
            "primary_metric": "gap_to_known_optimum",
            "objective_direction": "minimize",
            "plots": plots,
        }
    )
    if len(defects):
        print(
            f"ENCODING DEFECT: {len(defects)} load-bearing row(s) produced a broken QUBO. "
            "See qubo_report.md.",
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
