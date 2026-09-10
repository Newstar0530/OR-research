"""Does the reformulation still solve the same problem?

A transformation chain is only useful if each rewrite preserves the answer, and
the failure mode that matters is silent: the rewritten model still solves, still
returns a number, and the number is wrong. This module makes that failure
visible by comparing every step against an oracle that shares none of its
machinery.

Four outcomes are distinguished, because they have different causes and
different fixes:

* `equivalent`               - values agree and the model's solution really is
                               bilevel-feasible.
* `model_worse_than_truth`   - the model cut the optimum off. For the Big-M step
                               this is the classic symptom of constants that are
                               too small.
* `model_better_than_truth`  - the model beat the true optimum, which is
                               impossible for a correct reformulation and means
                               it admits points the original forbids: the
                               complementarity is not actually being enforced.
* `model_infeasible`         - the model has no solution while the original does.

The last two are modelling errors, not tuning problems, and are reported as
such rather than being smoothed into a gap number.
"""

from __future__ import annotations

import time
from typing import Literal, Sequence

import numpy as np
from pydantic import BaseModel, Field

from src.core.bilevel import BilevelLinearProgram
from src.core.bilevel_oracle import (
    BilevelVerification,
    OracleResult,
    solve_bilevel_by_vertex_enumeration,
    verify_bilevel_solution,
)
from src.core.mixed_integer_program import MipSolution, solve_mip
from src.core.transformations import (
    BigMBounds,
    KKTOneLevelProgram,
    derive_big_m_bounds,
    solve_kkt_by_pattern_enumeration,
    to_kkt_one_level,
    to_mi01,
)


StepVerdict = Literal[
    "equivalent",
    "model_worse_than_truth",
    "model_better_than_truth",
    "model_infeasible",
    "reference_unavailable",
    "not_run",
    "error",
]

DEFAULT_ATOL = 1e-6
DEFAULT_RTOL = 1e-6


class StepVerification(BaseModel):
    step: str
    verdict: StepVerdict = "not_run"
    reference_value: float | None = None
    model_value: float | None = None
    absolute_difference: float | None = None
    relative_difference: float | None = None
    solution_verified: bool = False
    solution_check: BilevelVerification | None = None
    model_status: str = ""
    backend: str = ""
    runtime_seconds: float = 0.0
    notes: str = ""

    @property
    def passed(self) -> bool:
        return self.verdict == "equivalent" and self.solution_verified


class EquivalenceReport(BaseModel):
    instance_id: str
    n_x: int = 0
    n_y: int = 0
    n_lambda: int = 0
    n_binaries: int = 0
    oracle_status: str = "not_run"
    oracle_value: float | None = None
    oracle_runtime_seconds: float = 0.0
    big_m_source: str = "derived"
    big_m_rigorous: bool = False
    slack_big_m: list[float] = Field(default_factory=list)
    dual_big_m: list[float] = Field(default_factory=list)
    big_m_notes: list[str] = Field(default_factory=list)
    steps: list[StepVerification] = Field(default_factory=list)
    runtime_seconds: float = 0.0

    @property
    def chain_is_equivalent(self) -> bool:
        return bool(self.steps) and all(step.passed for step in self.steps)

    def step(self, name: str) -> StepVerification | None:
        return next((s for s in self.steps if s.step == name), None)

    def to_markdown(self) -> str:
        def fmt(value: float | None) -> str:
            return "n/a" if value is None else f"{value:.9g}"

        lines = [
            f"# Transformation Equivalence: `{self.instance_id}`",
            "",
            f"- chain preserves the optimum: **{self.chain_is_equivalent}**",
            f"- oracle (vertex enumeration, no KKT and no Big-M): {self.oracle_status}, "
            f"F = {fmt(self.oracle_value)}",
            f"- sizes: n_x={self.n_x}, n_y={self.n_y}, multipliers={self.n_lambda}, "
            f"binaries={self.n_binaries}",
            f"- Big-M source: {self.big_m_source} (rigorous: {self.big_m_rigorous})",
            "",
            "## Steps",
            "",
            "| step | verdict | reference | model | abs diff | solution verified |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for item in self.steps:
            lines.append(
                f"| {item.step} | {item.verdict} | {fmt(item.reference_value)} | "
                f"{fmt(item.model_value)} | {fmt(item.absolute_difference)} | "
                f"{item.solution_verified} |"
            )
        lines.append("")
        lines.append("## Big-M Constants")
        lines.append(f"- slack: {[round(v, 6) for v in self.slack_big_m]}")
        lines.append(f"- dual:  {[round(v, 6) for v in self.dual_big_m]}")
        lines.extend(f"- {note}" for note in self.big_m_notes)
        lines.append("")
        lines.append("## Notes")
        for item in self.steps:
            if item.notes:
                lines.append(f"- **{item.step}**: {item.notes}")
        lines.append("")
        lines.append("## Human Verification")
        lines.append(
            "- The oracle assumes the joint region is compact, which the box bounds guarantee, "
            "and uses the optimistic bilevel formulation -- the same one the KKT reformulation "
            "implements."
        )
        lines.append(
            "- `equivalent` on these instances is evidence about these instances. It is not a "
            "proof that the transformation is correct in general."
        )
        return "\n".join(lines) + "\n"


def _classify(
    reference: float | None,
    model: float | None,
    minimize: bool,
    model_status: str,
    atol: float = DEFAULT_ATOL,
    rtol: float = DEFAULT_RTOL,
) -> tuple[StepVerdict, float | None, float | None]:
    if reference is None:
        return "reference_unavailable", None, None
    if model is None:
        return ("model_infeasible" if model_status in ("infeasible", "no_solution") else "error"), None, None
    difference = model - reference
    tolerance = atol + rtol * max(1.0, abs(reference))
    relative = difference / max(1.0, abs(reference))
    if abs(difference) <= tolerance:
        return "equivalent", abs(difference), relative
    beat_the_truth = difference < 0 if minimize else difference > 0
    verdict: StepVerdict = "model_better_than_truth" if beat_the_truth else "model_worse_than_truth"
    return verdict, abs(difference), relative


def _verify_point(
    blp: BilevelLinearProgram, x: Sequence[float], y: Sequence[float], atol: float
) -> BilevelVerification:
    return verify_bilevel_solution(blp, x, y, tolerance=max(atol, 1e-6))


def verify_transformation_chain(
    blp: BilevelLinearProgram,
    slack_big_m: float | Sequence[float] | None = None,
    dual_big_m: float | Sequence[float] | None = None,
    oracle: OracleResult | None = None,
    kkt: KKTOneLevelProgram | None = None,
    bounds: BigMBounds | None = None,
    time_limit: float = 60.0,
    atol: float = DEFAULT_ATOL,
    rtol: float = DEFAULT_RTOL,
    max_patterns: int = 1 << 16,
) -> EquivalenceReport:
    """Verify `bilevel -> KKT` and `KKT -> MI 0-1` against an independent oracle."""

    started = time.perf_counter()
    minimize = blp.upper_sense == "minimize"
    kkt = kkt or to_kkt_one_level(blp)
    bounds = bounds or derive_big_m_bounds(blp)
    manual = slack_big_m is not None or dual_big_m is not None
    slack_m = bounds.slack_bounds if slack_big_m is None else slack_big_m
    dual_m = bounds.dual_bounds if dual_big_m is None else dual_big_m

    oracle = oracle or solve_bilevel_by_vertex_enumeration(blp, tolerance=atol)
    report = EquivalenceReport(
        instance_id=blp.instance_id,
        n_x=blp.n_x,
        n_y=blp.n_y,
        n_lambda=kkt.n_lambda,
        n_binaries=kkt.n_lambda,
        oracle_status=oracle.status,
        oracle_value=oracle.upper_objective,
        oracle_runtime_seconds=oracle.runtime_seconds,
        big_m_source="manual" if manual else "derived",
        big_m_rigorous=bounds.is_rigorous and not manual,
        slack_big_m=[float(v) for v in (slack_m if not np.isscalar(slack_m) else [slack_m] * kkt.n_lambda)],
        dual_big_m=[float(v) for v in (dual_m if not np.isscalar(dual_m) else [dual_m] * kkt.n_lambda)],
        big_m_notes=list(bounds.notes),
    )

    # -- step 1: bilevel -> KKT one-level ------------------------------------
    kkt_solution = solve_kkt_by_pattern_enumeration(
        kkt, max_patterns=max_patterns, time_limit=time_limit
    )
    step1 = StepVerification(
        step="bilevel_to_kkt",
        model_status=kkt_solution.status,
        backend=kkt_solution.backend,
        runtime_seconds=kkt_solution.runtime_seconds,
        reference_value=oracle.upper_objective,
        model_value=kkt_solution.objective,
    )
    if kkt_solution.status == "not_run":
        step1.verdict = "not_run"
        step1.notes = kkt_solution.notes
    else:
        verdict, absolute, relative = _classify(
            oracle.upper_objective, kkt_solution.objective, minimize, kkt_solution.status, atol, rtol
        )
        step1.verdict = verdict
        step1.absolute_difference = absolute
        step1.relative_difference = relative
        if kkt_solution.values:
            x, y, _ = kkt.split(kkt_solution.values)
            check = _verify_point(blp, x, y, atol)
            step1.solution_check = check
            step1.solution_verified = check.is_bilevel_feasible
            if not check.is_bilevel_feasible:
                step1.notes = (
                    "the KKT program's solution is not bilevel-feasible "
                    f"(joint violation {check.max_joint_violation:.3g}, follower suboptimality "
                    f"{check.follower_suboptimality!s}); the KKT derivation is wrong."
                )
        if verdict == "model_better_than_truth" and not step1.notes:
            step1.notes = (
                "the KKT program beat the true optimum, so it admits points the bilevel program "
                "forbids -- check the stationarity signs and which rows carry multipliers."
            )
    report.steps.append(step1)

    # -- step 2: KKT -> MI 0-1 ----------------------------------------------
    step2 = StepVerification(step="kkt_to_mi01", reference_value=oracle.upper_objective)
    try:
        mi01 = to_mi01(kkt, slack_m, dual_m)
    except ValueError as exc:
        step2.verdict = "error"
        step2.notes = str(exc)
        report.steps.append(step2)
        report.runtime_seconds = time.perf_counter() - started
        return report

    mip_solution = solve_mip(mi01, time_limit=time_limit, tolerance=atol)
    step2.model_status = mip_solution.status
    step2.backend = mip_solution.backend
    step2.runtime_seconds = mip_solution.runtime_seconds
    step2.model_value = mip_solution.objective
    report.n_binaries = mi01.n_binaries
    if mip_solution.status == "error":
        step2.verdict = "error"
        step2.notes = mip_solution.notes
    else:
        verdict, absolute, relative = _classify(
            oracle.upper_objective, mip_solution.objective, minimize, mip_solution.status, atol, rtol
        )
        step2.verdict = verdict
        step2.absolute_difference = absolute
        step2.relative_difference = relative
        if mip_solution.values:
            x, y, _ = kkt.split(mip_solution.values[: kkt.n_vars])
            check = _verify_point(blp, x, y, atol)
            step2.solution_check = check
            step2.solution_verified = check.is_bilevel_feasible
        if verdict == "model_worse_than_truth":
            step2.notes = (
                "the MI 0-1 model's optimum is worse than the true one, so the linearisation cut "
                "the optimal point off: at least one Big-M is too small."
            )
        elif verdict == "model_better_than_truth":
            step2.notes = (
                "the MI 0-1 model beat the true optimum, so complementarity is not being enforced "
                "-- check the switch direction of z and the sign of the slack rows."
            )
        elif verdict == "model_infeasible":
            step2.notes = (
                "the MI 0-1 model is infeasible although the bilevel program is not: the Big-M "
                "constants are too small to admit any complementarity pattern."
            )
    report.steps.append(step2)
    report.runtime_seconds = time.perf_counter() - started
    return report


# ---------------------------------------------------------------------------
# Big-M sensitivity
# ---------------------------------------------------------------------------


class SweepPoint(BaseModel):
    label: str
    scale: float | None = None
    uniform_value: float | None = None
    verdict: StepVerdict
    model_value: float | None = None
    reference_value: float | None = None
    absolute_difference: float | None = None
    solution_verified: bool = False
    runtime_seconds: float = 0.0
    notes: str = ""


class BigMSweep(BaseModel):
    instance_id: str
    reference_value: float | None = None
    derived_slack_max: float = 0.0
    derived_dual_max: float = 0.0
    points: list[SweepPoint] = Field(default_factory=list)
    smallest_equivalent_scale: float | None = None
    largest_failing_scale: float | None = None
    monotone: bool = True
    runtime_seconds: float = 0.0

    def to_markdown(self) -> str:
        lines = [
            f"# Big-M Sensitivity: `{self.instance_id}`",
            "",
            f"- true optimum (independent oracle): "
            f"{'n/a' if self.reference_value is None else f'{self.reference_value:.9g}'}",
            f"- derived bounds: max slack {self.derived_slack_max:.6g}, "
            f"max multiplier {self.derived_dual_max:.6g}",
            f"- smallest scale that preserves the optimum: "
            f"{'none tested' if self.smallest_equivalent_scale is None else self.smallest_equivalent_scale}",
            f"- verdicts are monotone in the scale: {self.monotone}",
            "",
            "| setting | verdict | model value | abs diff | solution verified |",
            "| --- | --- | --- | --- | --- |",
        ]
        for point in self.points:
            value = "n/a" if point.model_value is None else f"{point.model_value:.9g}"
            difference = (
                "n/a" if point.absolute_difference is None else f"{point.absolute_difference:.6g}"
            )
            lines.append(
                f"| {point.label} | {point.verdict} | {value} | {difference} | "
                f"{point.solution_verified} |"
            )
        lines.append("")
        lines.append(
            "A Big-M below the threshold does not merely slow the solver down: it removes the "
            "optimal point, and the model then reports a different answer with no warning."
        )
        return "\n".join(lines) + "\n"


DEFAULT_SCALES: tuple[float, ...] = (0.01, 0.05, 0.1, 0.25, 0.5, 0.9, 1.0, 2.0, 10.0)


def sweep_big_m(
    blp: BilevelLinearProgram,
    scales: Sequence[float] = DEFAULT_SCALES,
    uniform_values: Sequence[float] | None = None,
    time_limit: float = 60.0,
    atol: float = DEFAULT_ATOL,
    rtol: float = DEFAULT_RTOL,
) -> BigMSweep:
    """Locate the Big-M threshold below which the linearisation stops being exact.

    `scales` multiply the rigorously derived bounds. `uniform_values` instead
    applies one flat constant to every row, which is what a paper that writes
    "we set M = 1e6" is actually doing -- worth testing separately, because a
    flat constant can be simultaneously too small for one row and needlessly
    huge for another.
    """

    started = time.perf_counter()
    kkt = to_kkt_one_level(blp)
    bounds = derive_big_m_bounds(blp)
    oracle = solve_bilevel_by_vertex_enumeration(blp, tolerance=atol)
    sweep = BigMSweep(
        instance_id=blp.instance_id,
        reference_value=oracle.upper_objective,
        derived_slack_max=bounds.max_slack,
        derived_dual_max=bounds.max_dual,
    )

    def run(label: str, slack_m, dual_m, scale=None, uniform=None) -> None:
        report = verify_transformation_chain(
            blp,
            slack_big_m=slack_m,
            dual_big_m=dual_m,
            oracle=oracle,
            kkt=kkt,
            bounds=bounds,
            time_limit=time_limit,
            atol=atol,
            rtol=rtol,
        )
        step = report.step("kkt_to_mi01")
        assert step is not None
        sweep.points.append(
            SweepPoint(
                label=label,
                scale=scale,
                uniform_value=uniform,
                verdict=step.verdict,
                model_value=step.model_value,
                reference_value=step.reference_value,
                absolute_difference=step.absolute_difference,
                solution_verified=step.solution_verified,
                runtime_seconds=step.runtime_seconds,
                notes=step.notes,
            )
        )

    for scale in scales:
        run(
            f"derived x {scale:g}",
            [v * scale for v in bounds.slack_bounds],
            [v * scale for v in bounds.dual_bounds],
            scale=float(scale),
        )
    for value in uniform_values or ():
        run(f"uniform M = {value:g}", float(value), float(value), uniform=float(value))

    scaled = [p for p in sweep.points if p.scale is not None]
    passing = [p.scale for p in scaled if p.verdict == "equivalent" and p.solution_verified]
    failing = [p.scale for p in scaled if p.verdict != "equivalent"]
    sweep.smallest_equivalent_scale = min(passing) if passing else None
    sweep.largest_failing_scale = max(failing) if failing else None
    if passing and failing:
        sweep.monotone = max(failing) < min(passing)
    sweep.runtime_seconds = time.perf_counter() - started
    return sweep
