"""Explicit contract for the outcome of solving one optimization instance.

This module exists because "the experiment printed a number" is not evidence in
operations research. A usable record has to say which solver produced the
number, whether the solver *proved* anything, how far the incumbent is from a
bound, and -- critically -- whether an independent checker confirmed the
solution is feasible. `SolveResult` is that record.

The CSV column order deliberately starts with the legacy experiment-contract
columns (`instance_id`, `method`, `objective`, `runtime_seconds`, `seed`) so
existing tooling and the legacy contract keep working unchanged.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Iterable, Literal, Sequence

from pydantic import BaseModel, Field


SolverStatus = Literal[
    "optimal",      # proved optimal by an exact method
    "feasible",     # incumbent found, optimality not proved
    "time_limit",   # stopped by the time limit, incumbent available
    "infeasible",   # proved infeasible
    "unbounded",    # proved unbounded
    "no_solution",  # stopped without any incumbent (unknown)
    "error",        # backend failed, or a claim contradicted verification
    "not_run",      # backend unavailable / skipped
]

ObjectiveSense = Literal["maximize", "minimize"]

#: Statuses that assert "I am handing you a usable solution vector".
#: Any of these must survive independent feasibility verification.
CLAIMS_SOLUTION: frozenset[str] = frozenset({"optimal", "feasible", "time_limit"})

#: Statuses that assert a *proved* optimum.
CLAIMS_OPTIMALITY: frozenset[str] = frozenset({"optimal"})

_EPS = 1e-9


def compute_mip_gap(
    objective: float | None,
    dual_bound: float | None,
    objective_sense: ObjectiveSense = "minimize",
) -> float | None:
    """Relative MIP gap `|bound - incumbent| / max(|incumbent|, eps)`.

    Returns None when either side is missing. The value is clipped at zero:
    a bound that is *worse* than the incumbent means the bound is wrong, which
    is reported through `notes`, not by emitting a negative gap.
    """

    if objective is None or dual_bound is None:
        return None
    denominator = max(abs(float(objective)), _EPS)
    if objective_sense == "minimize":
        raw = (float(objective) - float(dual_bound)) / denominator
    else:
        raw = (float(dual_bound) - float(objective)) / denominator
    return max(0.0, raw)


def relative_shortfall(
    objective: float | None,
    reference: float | None,
    objective_sense: ObjectiveSense = "minimize",
) -> float | None:
    """Signed relative distance from `objective` to a reference optimum.

    Positive means the objective is *worse* than the reference. The sign is kept
    on purpose: a negative value on a verified-feasible solution means the
    reference itself is wrong, which is a finding, not a rounding artifact.
    """

    if objective is None or reference is None:
        return None
    denominator = max(abs(float(reference)), _EPS)
    if objective_sense == "minimize":
        return (float(objective) - float(reference)) / denominator
    return (float(reference) - float(objective)) / denominator


class SolveResult(BaseModel):
    """One (instance, method) solve attempt, with verification attached."""

    # --- identity -----------------------------------------------------------
    instance_id: str
    method: str
    solver_backend: str = "unknown"
    instance_family: str = "unknown"
    seed: int = 0
    problem_size: int = 0

    # --- model size ---------------------------------------------------------
    n_vars: int = 0
    n_constraints: int = 0
    objective_sense: ObjectiveSense = "minimize"

    # --- what the solver reported ------------------------------------------
    solver_status: SolverStatus = "not_run"
    objective: float | None = None
    dual_bound: float | None = None
    mip_gap: float | None = None
    node_count: int | None = None
    runtime_seconds: float = 0.0
    hit_time_limit: bool = False

    # --- what independent verification found --------------------------------
    feasible: bool = False
    max_violation: float = 0.0
    n_violated_constraints: int = 0
    verification_ok: bool = True

    # --- reference quality --------------------------------------------------
    known_optimum: float | None = None
    known_optimum_source: str = ""
    gap_to_known_optimum: float | None = None

    # --- payload (not written to CSV) ---------------------------------------
    solution: list[int] = Field(default_factory=list)
    #: Continuous or mixed solutions, for models that are not purely binary.
    solution_values: list[float] = Field(default_factory=list)
    notes: str = ""

    #: Domain-specific columns appended to the CSV row, e.g. the Big-M constant
    #: and the equivalence verdict of a transformation experiment. Kept separate
    #: from the fixed schema so a new experiment family cannot silently change
    #: the meaning of an existing column.
    extra: dict[str, float | int | str | bool | None] = Field(default_factory=dict)

    @property
    def claims_solution(self) -> bool:
        return self.solver_status in CLAIMS_SOLUTION

    @property
    def any_solution(self) -> list[float]:
        """Whichever payload this result carries, as floats."""

        return self.solution_values or [float(v) for v in self.solution]

    @property
    def is_trustworthy(self) -> bool:
        """A row you are allowed to draw a conclusion from."""

        return self.claims_solution and self.feasible and self.verification_ok

    @property
    def proved_optimal(self) -> bool:
        return self.solver_status in CLAIMS_OPTIMALITY and self.is_trustworthy

    def with_derived_metrics(self) -> "SolveResult":
        """Fill `mip_gap` and `gap_to_known_optimum` from the raw numbers."""

        if self.mip_gap is None:
            self.mip_gap = compute_mip_gap(self.objective, self.dual_bound, self.objective_sense)
        if self.gap_to_known_optimum is None:
            self.gap_to_known_optimum = relative_shortfall(
                self.objective, self.known_optimum, self.objective_sense
            )
        return self

    def add_note(self, note: str) -> None:
        note = note.strip()
        if not note:
            return
        self.notes = f"{self.notes} | {note}".strip(" |") if self.notes else note

    def to_row(self) -> dict[str, Any]:
        """Flat record for `results.csv`, in `SOLVE_RESULT_COLUMNS` order."""

        row: dict[str, Any] = {
            # legacy experiment-contract columns come first
            "instance_id": self.instance_id,
            "method": self.method,
            "objective": self.objective,
            "runtime_seconds": self.runtime_seconds,
            "seed": self.seed,
            # model / instance descriptors
            "problem_size": self.problem_size,
            "instance_family": self.instance_family,
            "objective_sense": self.objective_sense,
            "n_vars": self.n_vars,
            "n_constraints": self.n_constraints,
            # solver evidence
            "solver_backend": self.solver_backend,
            "solver_status": self.solver_status,
            "dual_bound": self.dual_bound,
            "mip_gap": self.mip_gap,
            "node_count": self.node_count,
            "hit_time_limit": self.hit_time_limit,
            # independent verification
            "feasible": self.feasible,
            "max_violation": self.max_violation,
            "constraint_violation": self.max_violation,
            "n_violated_constraints": self.n_violated_constraints,
            "verification_ok": self.verification_ok,
            # reference quality
            "known_optimum": self.known_optimum,
            "known_optimum_source": self.known_optimum_source,
            "gap_to_known_optimum": self.gap_to_known_optimum,
            "gap": self.gap_to_known_optimum,
            "notes": self.notes,
        }
        # Extras may never overwrite a fixed schema column.
        for key, value in self.extra.items():
            if key in row:
                raise ValueError(
                    f"extra column `{key}` collides with a fixed SolveResult column; "
                    "pick a different name rather than shadowing the schema."
                )
            row[key] = value
        return row


#: Canonical CSV column order for solver-backed experiments.
SOLVE_RESULT_COLUMNS: list[str] = list(
    SolveResult(instance_id="_", method="_").to_row().keys()
)


def write_results_csv(results: Sequence[SolveResult], path: str | Path) -> Path:
    """Write `results.csv`: the fixed schema first, then any extra columns.

    Extra columns are the union across rows, so a row that omits one still
    writes a blank rather than shifting every following field.
    """

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    extra_columns = sorted({key for result in results for key in result.extra})
    fieldnames = SOLVE_RESULT_COLUMNS + extra_columns
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, restval="")
        writer.writeheader()
        for result in results:
            writer.writerow(result.to_row())
    return target


def summarize_results(results: Iterable[SolveResult]) -> dict[str, Any]:
    """Compact machine-readable summary for the `SUMMARY_JSON:` stdout line."""

    rows = list(results)
    per_method: dict[str, dict[str, Any]] = {}
    for result in rows:
        bucket = per_method.setdefault(
            result.method,
            {
                "rows": 0,
                "trustworthy_rows": 0,
                "infeasible_rows": 0,
                "verification_failures": 0,
                "proved_optimal_rows": 0,
                "_gap_sum": 0.0,
                "_gap_n": 0,
                "_time_sum": 0.0,
            },
        )
        bucket["rows"] += 1
        bucket["_time_sum"] += float(result.runtime_seconds)
        if result.is_trustworthy:
            bucket["trustworthy_rows"] += 1
        if result.claims_solution and not result.feasible:
            bucket["infeasible_rows"] += 1
        if not result.verification_ok:
            bucket["verification_failures"] += 1
        if result.proved_optimal:
            bucket["proved_optimal_rows"] += 1
        if result.gap_to_known_optimum is not None and result.is_trustworthy:
            bucket["_gap_sum"] += float(result.gap_to_known_optimum)
            bucket["_gap_n"] += 1

    methods: dict[str, Any] = {}
    for method, bucket in per_method.items():
        gap_n = bucket.pop("_gap_n")
        gap_sum = bucket.pop("_gap_sum")
        time_sum = bucket.pop("_time_sum")
        bucket["mean_gap_to_known_optimum"] = (gap_sum / gap_n) if gap_n else None
        bucket["mean_runtime_seconds"] = (time_sum / bucket["rows"]) if bucket["rows"] else None
        methods[method] = bucket

    total_verification_failures = sum(1 for r in rows if not r.verification_ok)
    return {
        "status": "success" if rows and total_verification_failures == 0 else "attention_required",
        "rows": len(rows),
        "methods": methods,
        "verification_failures": total_verification_failures,
        "infeasible_claimed_solutions": sum(
            1 for r in rows if r.claims_solution and not r.feasible
        ),
    }
