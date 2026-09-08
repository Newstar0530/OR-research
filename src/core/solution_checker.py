"""Independent verification of a claimed solution.

The rule this module enforces: **never let the solver grade its own homework.**
A solver (or a heuristic, or LLM-written code) hands back a vector and a number.
This module substitutes the vector back into every constraint, recomputes the
objective from the instance data, and reports what it found. Downstream code is
only allowed to trust rows that passed through here.
"""

from __future__ import annotations

from typing import Sequence

from pydantic import BaseModel, Field

from src.core.binary_program import BinaryProgramInstance
from src.core.solve_result import CLAIMS_SOLUTION


DEFAULT_TOLERANCE = 1e-6


class ConstraintViolation(BaseModel):
    name: str
    sense: str
    lhs: float
    rhs: float
    violation: float


class FeasibilityReport(BaseModel):
    """What an independent pass over the constraints actually found."""

    feasible: bool
    n_violated: int = 0
    max_violation: float = 0.0
    total_violation: float = 0.0
    violations: list[ConstraintViolation] = Field(default_factory=list)
    domain_errors: list[str] = Field(default_factory=list)
    recomputed_objective: float | None = None
    tolerance: float = DEFAULT_TOLERANCE

    def to_markdown(self) -> str:
        lines = [
            "# Feasibility Verification",
            "",
            f"- feasible: {self.feasible}",
            f"- violated_constraints: {self.n_violated}",
            f"- max_violation: {self.max_violation:.6g}",
            f"- recomputed_objective: "
            f"{self.recomputed_objective if self.recomputed_objective is None else f'{self.recomputed_objective:.6g}'}",
            f"- tolerance: {self.tolerance:.3g}",
            "",
        ]
        lines.append("## Domain Errors")
        lines.extend(f"- {item}" for item in self.domain_errors or ["None."])
        lines.append("")
        lines.append("## Violated Constraints")
        if not self.violations:
            lines.append("- None.")
        for item in self.violations[:50]:
            lines.append(
                f"- `{item.name}`: lhs={item.lhs:.6g} {item.sense} rhs={item.rhs:.6g} "
                f"(violation={item.violation:.6g})"
            )
        if len(self.violations) > 50:
            lines.append(f"- ... {len(self.violations) - 50} more violated constraint(s) not listed.")
        return "\n".join(lines) + "\n"


def _normalize_binary(
    solution: Sequence[int | float] | None,
    n_vars: int,
    tolerance: float,
) -> tuple[list[int], list[str]]:
    """Coerce a claimed solution to 0/1 integers, collecting domain errors."""

    errors: list[str] = []
    if solution is None:
        return [], ["No solution vector was provided."]
    values = list(solution)
    if len(values) != n_vars:
        errors.append(
            f"Solution length {len(values)} does not match the instance's {n_vars} variables."
        )
        return [], errors
    normalized: list[int] = []
    for index, raw in enumerate(values):
        try:
            value = float(raw)
        except (TypeError, ValueError):
            errors.append(f"Variable x[{index}] is not numeric: {raw!r}.")
            normalized.append(0)
            continue
        if abs(value - round(value)) > tolerance:
            errors.append(f"Variable x[{index}]={value:.6g} is not integral.")
        rounded = int(round(value))
        if rounded not in (0, 1):
            errors.append(f"Variable x[{index}]={value:.6g} is outside the binary domain {{0, 1}}.")
            rounded = 1 if rounded > 1 else 0
        normalized.append(rounded)
    return normalized, errors


def check_binary_solution(
    instance: BinaryProgramInstance,
    solution: Sequence[int | float] | None,
    tolerance: float = DEFAULT_TOLERANCE,
) -> FeasibilityReport:
    """Substitute `solution` into every constraint of `instance` and report."""

    normalized, domain_errors = _normalize_binary(solution, instance.n_vars, tolerance)
    if not normalized:
        return FeasibilityReport(
            feasible=False,
            domain_errors=domain_errors or ["Solution could not be interpreted."],
            tolerance=tolerance,
        )

    violations: list[ConstraintViolation] = []
    for constraint in instance.constraints:
        magnitude = constraint.violation(normalized)
        if magnitude > tolerance:
            violations.append(
                ConstraintViolation(
                    name=constraint.name,
                    sense=constraint.sense,
                    lhs=constraint.lhs(normalized),
                    rhs=float(constraint.rhs),
                    violation=float(magnitude),
                )
            )

    return FeasibilityReport(
        feasible=not violations and not domain_errors,
        n_violated=len(violations),
        max_violation=max((v.violation for v in violations), default=0.0),
        total_violation=float(sum(v.violation for v in violations)),
        violations=violations,
        domain_errors=domain_errors,
        recomputed_objective=instance.objective_value(normalized),
        tolerance=tolerance,
    )


def verify_solver_claim(
    instance: BinaryProgramInstance,
    solution: Sequence[int | float] | None,
    claimed_status: str,
    claimed_objective: float | None,
    tolerance: float = DEFAULT_TOLERANCE,
    objective_tolerance: float = 1e-4,
) -> tuple[FeasibilityReport, bool, list[str]]:
    """Cross-examine what a backend claimed against what verification found.

    Returns `(report, claim_is_consistent, problems)`. A False verdict means the
    backend is lying or buggy, which is a defect to surface loudly -- not a
    number to average into a results table.
    """

    report = check_binary_solution(instance, solution, tolerance=tolerance)
    problems: list[str] = []

    if claimed_status not in CLAIMS_SOLUTION:
        # No solution was claimed, so there is nothing to contradict.
        return report, True, problems

    if not solution:
        problems.append(
            f"Backend reported status `{claimed_status}` but returned no solution vector."
        )
    if report.domain_errors:
        problems.append(
            f"Backend reported status `{claimed_status}` but the solution violates the binary "
            f"domain: {report.domain_errors[0]}"
        )
    if report.n_violated:
        worst = report.violations[0]
        problems.append(
            f"Backend reported status `{claimed_status}` but independent verification found "
            f"{report.n_violated} violated constraint(s); worst `{worst.name}` by {worst.violation:.6g}."
        )
    if (
        claimed_objective is not None
        and report.recomputed_objective is not None
        and abs(float(claimed_objective) - report.recomputed_objective)
        > objective_tolerance * max(1.0, abs(report.recomputed_objective))
    ):
        problems.append(
            f"Backend reported objective {float(claimed_objective):.6g} but recomputing it from the "
            f"instance data gives {report.recomputed_objective:.6g}."
        )

    return report, not problems, problems
