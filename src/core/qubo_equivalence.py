"""Did the QUBO keep the problem, and if not, which constant lost it?

The hop to a QUBO spends two things at once, and lumping them together makes
both undiagnosable. So this module measures them separately, against two
different references:

    source MIP  --(discretise)-->  binary grid MIP  --(penalise)-->  QUBO
                 |                                   |
                 +-- cost of the grid                +-- correctness of the penalty

The binary grid model is still constrained, so it can be solved exactly by the
ordinary MIP machinery. Its optimum can only be *worse* than the source
optimum, because the grid is a subset of the original feasible set -- if it ever
comes out better, the projection is wrong, and that is reported as a defect
rather than as good news.

The QUBO is then judged only against the grid model. A ground state that
decodes to an infeasible point means the penalty is too small; one that decodes
to a feasible point worth less than the grid optimum means the sampler did not
reach the ground state; one worth *more* is impossible and means the encoding
is broken.

Every decoded point is checked against the source constraints directly. The
energy is checked too: at a feasible point every residual is zero, so the energy
must equal the objective exactly. When it does not, the slack bits cannot
represent the slack that point actually needs -- a failure that is invisible in
the objective value alone.
"""

from __future__ import annotations

import time
from typing import Any, Literal, Sequence

import numpy as np
from pydantic import BaseModel, Field

from src.core.mixed_integer_program import (
    MipSolution,
    MixedIntegerProgram,
    solve_mip,
    verify_mip_solution,
)
from src.core.qubo import QUBOModel, QUBOSolution, solve_qubo
from src.core.qubo_transformations import (
    DEFAULT_PRECISION,
    PenaltyBound,
    QUBOEncoding,
    build_qubo,
    derive_penalty_bound,
    to_binary_grid_mip,
    to_qubo,
)


QUBOVerdict = Literal[
    "equivalent",
    "discretisation_loss",
    "grid_better_than_true",
    "penalty_too_small",
    "qubo_worse_than_grid",
    "qubo_better_than_grid",
    "energy_encoding_mismatch",
    "model_infeasible",
    "not_proved",
    "not_run",
    "error",
]

DEFAULT_ATOL = 1e-6
DEFAULT_RTOL = 1e-6


class QUBOStepVerification(BaseModel):
    step: str
    verdict: QUBOVerdict = "not_run"
    reference_value: float | None = None
    model_value: float | None = None
    absolute_difference: float | None = None
    relative_difference: float | None = None
    solution_verified: bool = False
    energy_consistent: bool = True
    backend: str = ""
    runtime_seconds: float = 0.0
    notes: str = ""

    @property
    def passed(self) -> bool:
        return self.verdict == "equivalent" and self.solution_verified and self.energy_consistent


class QUBOEquivalenceReport(BaseModel):
    source_name: str
    sense: str = "minimize"
    true_value: float | None = None
    grid_value: float | None = None
    qubo_value: float | None = None
    discretisation_loss: float | None = None
    precision: float = DEFAULT_PRECISION
    penalty: float = 0.0
    penalty_is_rigorous: bool = False
    penalty_source: str = "derived"
    n_bits: int = 0
    n_grid_bits: int = 0
    n_slack_bits: int = 0
    qubo_density: float = 0.0
    qubo_dynamic_range: float = 0.0
    #: How far the penalty towers over the objective it is protecting. This is
    #: the number that moves when the penalty is scaled, and the one that says
    #: how much resolution a finite-precision sampler must spend on constraints
    #: rather than on the thing being optimised.
    penalty_to_objective_ratio: float = 0.0
    ground_state_proved: bool = False
    steps: list[QUBOStepVerification] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    runtime_seconds: float = 0.0

    @property
    def chain_is_equivalent(self) -> bool:
        return bool(self.steps) and all(step.passed for step in self.steps)

    @property
    def penalty_is_sound(self) -> bool:
        """The penalty step held, whatever the grid cost."""

        step = self.step("binary_grid_to_qubo")
        return bool(step and step.passed)

    def step(self, name: str) -> QUBOStepVerification | None:
        return next((s for s in self.steps if s.step == name), None)

    def to_markdown(self) -> str:
        def fmt(value: float | None) -> str:
            return "n/a" if value is None else f"{value:.9g}"

        lines = [
            f"# QUBO Equivalence: `{self.source_name}`",
            "",
            f"- chain preserves the optimum: **{self.chain_is_equivalent}**",
            f"- penalty step is sound: **{self.penalty_is_sound}**",
            f"- source optimum: {fmt(self.true_value)}",
            f"- best representable on the grid: {fmt(self.grid_value)} "
            f"(precision {self.precision:g})",
            f"- QUBO ground state decodes to: {fmt(self.qubo_value)}",
            f"- cost of discretisation: {fmt(self.discretisation_loss)}",
            f"- penalty: {self.penalty:.6g} ({self.penalty_source}, rigorous: "
            f"{self.penalty_is_rigorous})",
            f"- bits: {self.n_bits} ({self.n_grid_bits} variable + {self.n_slack_bits} slack), "
            f"density {self.qubo_density:.2f}",
            f"- coefficient dynamic range: {self.qubo_dynamic_range:.3g}",
            f"- penalty / largest objective coefficient: {self.penalty_to_objective_ratio:.3g}",
            f"- ground state proved by exhaustive sweep: {self.ground_state_proved}",
            "",
            "| step | verdict | reference | model | abs diff | decoded point feasible | energy consistent |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
        for item in self.steps:
            lines.append(
                f"| {item.step} | {item.verdict} | {fmt(item.reference_value)} | "
                f"{fmt(item.model_value)} | {fmt(item.absolute_difference)} | "
                f"{item.solution_verified} | {item.energy_consistent} |"
            )
        lines.append("")
        for item in self.steps:
            if item.notes:
                lines.append(f"- **{item.step}**: {item.notes}")
        lines.extend(f"- {note}" for note in self.notes)
        lines.extend(
            [
                "",
                "## Human Verification",
                "- A large penalty buys exactness on paper and spends it on dynamic range. Hardware "
                "coefficient precision is finite, so check the range above before assuming a "
                "correct QUBO is also a programmable one.",
                "- `not_proved` means a sampler returned the best energy it found. That is not a "
                "ground state, and it is not evidence that the reformulation is exact.",
            ]
        )
        return "\n".join(lines) + "\n"


def _difference(model: float, reference: float, minimize: bool) -> float:
    """Signed so that positive always means the model is worse."""

    return model - reference if minimize else reference - model


def verify_qubo_chain(
    mip: MixedIntegerProgram,
    penalty: float | None = None,
    precision: float = DEFAULT_PRECISION,
    slack_precision: float | None = None,
    max_bits_for_proof: int = 22,
    time_limit: float = 120.0,
    seed: int = 0,
    true_solution: MipSolution | None = None,
    atol: float = DEFAULT_ATOL,
    rtol: float = DEFAULT_RTOL,
) -> QUBOEquivalenceReport:
    """Verify `MIP -> binary grid -> QUBO`, attributing any loss to its own cause."""

    started = time.perf_counter()
    minimize = mip.sense == "minimize"
    report = QUBOEquivalenceReport(
        source_name=mip.name,
        sense=mip.sense,
        precision=precision,
        penalty_source="manual" if penalty is not None else "derived",
    )

    true_solution = true_solution or solve_mip(mip, time_limit=time_limit, tolerance=atol)
    report.true_value = true_solution.objective
    if true_solution.status not in ("optimal", "feasible") or true_solution.objective is None:
        report.notes.append(
            f"the source model is `{true_solution.status}`, so there is nothing to preserve"
        )
        report.runtime_seconds = time.perf_counter() - started
        return report

    # -- step 1: discretisation ---------------------------------------------
    grid_mip, grid_encoding = to_binary_grid_mip(mip, precision=precision)
    grid_solution = solve_mip(grid_mip, time_limit=time_limit, tolerance=atol)
    report.grid_value = grid_solution.objective
    report.n_grid_bits = grid_mip.n_vars

    step1 = QUBOStepVerification(
        step="mip_to_binary_grid",
        reference_value=true_solution.objective,
        model_value=grid_solution.objective,
        backend=grid_solution.backend,
        runtime_seconds=grid_solution.runtime_seconds,
    )
    if grid_solution.status not in ("optimal", "feasible") or grid_solution.objective is None:
        step1.verdict = "model_infeasible"
        step1.notes = (
            f"the grid model is `{grid_solution.status}` although the source model is not; "
            "the discretisation removed every feasible point."
        )
    else:
        loss = _difference(grid_solution.objective, true_solution.objective, minimize)
        report.discretisation_loss = loss
        step1.absolute_difference = abs(loss)
        step1.relative_difference = loss / max(1.0, abs(true_solution.objective))
        tolerance = atol + rtol * max(1.0, abs(true_solution.objective))
        decoded = grid_encoding.decode(grid_solution.values)
        verification = verify_mip_solution(mip, decoded, tolerance=max(atol, 1e-6))
        step1.solution_verified = verification.feasible
        if loss < -tolerance:
            step1.verdict = "grid_better_than_true"
            step1.notes = (
                "the grid model beat the source optimum, which is impossible for a subset of the "
                "feasible set: the variable substitution is wrong."
            )
        elif abs(loss) <= tolerance:
            step1.verdict = "equivalent"
            if grid_encoding.is_lossless:
                step1.notes = "no variable needed discretising, so the grid cannot cost anything"
        else:
            step1.verdict = "discretisation_loss"
            step1.notes = (
                f"the optimum is not representable at precision {precision:g}; the grid costs "
                f"{loss:.6g}. This is a measured cost of the encoding, not a defect."
            )
    report.steps.append(step1)

    if step1.verdict in ("model_infeasible", "grid_better_than_true"):
        report.runtime_seconds = time.perf_counter() - started
        return report

    # -- step 2: penalisation ------------------------------------------------
    bound = derive_penalty_bound(grid_mip)
    resolved_penalty = bound.penalty if penalty is None else float(penalty)
    report.penalty = resolved_penalty
    report.penalty_is_rigorous = bound.is_rigorous and penalty is None
    report.notes.extend(bound.notes)

    step2 = QUBOStepVerification(step="binary_grid_to_qubo", reference_value=grid_solution.objective)
    try:
        qubo, encoding = to_qubo(grid_mip, resolved_penalty, slack_precision=slack_precision)
    except ValueError as exc:
        step2.verdict = "error"
        step2.notes = str(exc)
        report.steps.append(step2)
        report.runtime_seconds = time.perf_counter() - started
        return report

    report.n_bits = qubo.n_bits
    report.n_slack_bits = qubo.n_bits - grid_mip.n_vars
    report.qubo_density = qubo.density
    report.qubo_dynamic_range = qubo.dynamic_range
    largest_objective = max((abs(float(c)) for c in grid_mip.objective), default=0.0)
    report.penalty_to_objective_ratio = float(resolved_penalty / max(largest_objective, 1e-12))

    qubo_solution = solve_qubo(
        qubo, max_bits_for_proof=max_bits_for_proof, time_limit=time_limit, seed=seed
    )
    report.ground_state_proved = qubo_solution.status == "optimal"
    step2.backend = qubo_solution.backend
    step2.runtime_seconds = qubo_solution.runtime_seconds

    if not qubo_solution.has_solution:
        step2.verdict = "not_run" if qubo_solution.status == "not_run" else "error"
        step2.notes = qubo_solution.notes
        report.steps.append(step2)
        report.runtime_seconds = time.perf_counter() - started
        return report

    bits = qubo_solution.bits
    decoded = grid_encoding.decode(bits[: grid_mip.n_vars])
    verification = verify_mip_solution(mip, decoded, tolerance=max(atol, 1e-6))
    decoded_objective = mip.objective_value(decoded)
    report.qubo_value = decoded_objective
    step2.model_value = decoded_objective
    step2.solution_verified = verification.feasible

    # At a feasible point every residual is zero, so the energy is the objective.
    expected_energy = encoding.energy_sign * decoded_objective
    energy_gap = abs(float(qubo_solution.energy) - expected_energy)
    step2.energy_consistent = (
        not verification.feasible
        or energy_gap <= atol + rtol * max(1.0, abs(expected_energy))
    )

    difference = _difference(decoded_objective, grid_solution.objective, minimize)
    step2.absolute_difference = abs(difference)
    step2.relative_difference = difference / max(1.0, abs(grid_solution.objective))
    tolerance = atol + rtol * max(1.0, abs(grid_solution.objective))

    if not verification.feasible:
        step2.verdict = "penalty_too_small"
        detail = (
            verification.domain_errors[0]
            if verification.domain_errors
            else f"`{verification.violations[0].name}` violated by {verification.max_violation:.6g}"
        )
        step2.notes = (
            f"the ground state decodes to an infeasible point ({detail}), so breaking a constraint "
            f"paid at penalty {resolved_penalty:.6g}."
        )
    elif not step2.energy_consistent and report.ground_state_proved:
        step2.verdict = "energy_encoding_mismatch"
        step2.notes = (
            f"the proved ground state is feasible but its energy is off by {energy_gap:.6g}. Since "
            "a zero-residual encoding would have had lower energy, none exists: the slack bits "
            "cannot represent the slack this point needs."
        )
    elif not step2.energy_consistent:
        # A sampler can return a bitstring whose variables are feasible while its
        # slack register is simply wrong. That is non-convergence, not a broken
        # encoding, and calling it one would be the very misattribution this
        # module exists to prevent.
        step2.verdict = "not_proved"
        step2.notes = (
            f"the sampler's bitstring decodes to a feasible point but pays {energy_gap:.6g} in "
            "penalty because its slack register does not match; it did not reach the ground state."
        )
    elif difference > tolerance:
        step2.verdict = (
            "qubo_worse_than_grid" if report.ground_state_proved else "not_proved"
        )
        step2.notes = (
            "the proved ground state is worse than the grid optimum, so the penalty distorted the "
            "landscape."
            if report.ground_state_proved
            else "a sampler was used, so this is its best draw rather than a ground state."
        )
    elif difference < -tolerance:
        step2.verdict = "qubo_better_than_grid"
        step2.notes = (
            "the decoded point beats the grid optimum while satisfying every constraint, which is "
            "impossible: the encoding does not represent the model it claims to."
        )
    else:
        step2.verdict = "equivalent"
        if not report.ground_state_proved:
            step2.verdict = "not_proved"
            step2.notes = (
                "a sampler matched the grid optimum, but matching is not proving; the ground state "
                "was not established."
            )

    report.steps.append(step2)
    report.runtime_seconds = time.perf_counter() - started
    return report


# ---------------------------------------------------------------------------
# Penalty sensitivity
# ---------------------------------------------------------------------------


class PenaltySweepPoint(BaseModel):
    label: str
    scale: float | None = None
    penalty: float = 0.0
    verdict: QUBOVerdict
    model_value: float | None = None
    reference_value: float | None = None
    solution_verified: bool = False
    dynamic_range: float = 0.0
    penalty_to_objective_ratio: float = 0.0
    runtime_seconds: float = 0.0
    notes: str = ""


class PenaltySweep(BaseModel):
    source_name: str
    grid_value: float | None = None
    derived_penalty: float = 0.0
    points: list[PenaltySweepPoint] = Field(default_factory=list)
    smallest_sound_scale: float | None = None
    largest_failing_scale: float | None = None
    monotone: bool = True
    runtime_seconds: float = 0.0

    def to_markdown(self) -> str:
        lines = [
            f"# Penalty Sensitivity: `{self.source_name}`",
            "",
            f"- best representable objective: "
            f"{'n/a' if self.grid_value is None else f'{self.grid_value:.9g}'}",
            f"- derived penalty: {self.derived_penalty:.6g}",
            f"- smallest scale that keeps the ground state feasible: "
            f"{'none tested' if self.smallest_sound_scale is None else self.smallest_sound_scale}",
            f"- verdicts are monotone in the scale: {self.monotone}",
            "",
            "| setting | penalty | verdict | decoded value | feasible | penalty/objective |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for point in self.points:
            value = "n/a" if point.model_value is None else f"{point.model_value:.9g}"
            lines.append(
                f"| {point.label} | {point.penalty:.6g} | {point.verdict} | {value} | "
                f"{point.solution_verified} | {point.penalty_to_objective_ratio:.3g} |"
            )
        lines.append("")
        lines.append(
            "Below the threshold the ground state is an infeasible bitstring, and a sampler "
            "returns it with no indication that anything is wrong."
        )
        return "\n".join(lines) + "\n"


#: The derived bound is sufficient but not tight, so the sweep has to reach well
#: below it before anything breaks. Stopping at 0.01 would report "never fails"
#: and conclude the wrong thing.
DEFAULT_PENALTY_SCALES: tuple[float, ...] = (0.0001, 0.001, 0.003, 0.01, 0.1, 1.0, 10.0)


def sweep_penalty(
    mip: MixedIntegerProgram,
    scales: Sequence[float] = DEFAULT_PENALTY_SCALES,
    precision: float = DEFAULT_PRECISION,
    max_bits_for_proof: int = 22,
    time_limit: float = 120.0,
    seed: int = 0,
) -> PenaltySweep:
    """Find the penalty below which the ground state stops being a feasible point."""

    started = time.perf_counter()
    grid_mip, _ = to_binary_grid_mip(mip, precision=precision)
    bound = derive_penalty_bound(grid_mip)
    grid_solution = solve_mip(grid_mip, time_limit=time_limit)
    true_solution = solve_mip(mip, time_limit=time_limit)
    sweep = PenaltySweep(
        source_name=mip.name,
        grid_value=grid_solution.objective,
        derived_penalty=bound.penalty,
    )

    for scale in scales:
        report = verify_qubo_chain(
            mip,
            penalty=bound.penalty * float(scale),
            precision=precision,
            max_bits_for_proof=max_bits_for_proof,
            time_limit=time_limit,
            seed=seed,
            true_solution=true_solution,
        )
        step = report.step("binary_grid_to_qubo")
        sweep.points.append(
            PenaltySweepPoint(
                label=f"derived x {scale:g}",
                scale=float(scale),
                penalty=bound.penalty * float(scale),
                verdict=step.verdict if step else "not_run",
                model_value=step.model_value if step else None,
                reference_value=report.grid_value,
                solution_verified=bool(step and step.solution_verified),
                dynamic_range=report.qubo_dynamic_range,
                penalty_to_objective_ratio=report.penalty_to_objective_ratio,
                runtime_seconds=step.runtime_seconds if step else 0.0,
                notes=step.notes if step else "",
            )
        )

    sound = [p.scale for p in sweep.points if p.verdict == "equivalent" and p.solution_verified]
    failing = [p.scale for p in sweep.points if p.verdict != "equivalent"]
    sweep.smallest_sound_scale = min(sound) if sound else None
    sweep.largest_failing_scale = max(failing) if failing else None
    if sound and failing:
        sweep.monotone = max(failing) < min(sound)
    sweep.runtime_seconds = time.perf_counter() - started
    return sweep
