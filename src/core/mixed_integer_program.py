"""Mixed-integer linear programs: representation, solving, and verification.

The binary programs in `binary_program.py` cover pure 0-1 models. Reformulating
a bilevel program produces something different: continuous primal variables,
continuous duals, *and* binaries for the complementarity switches. This module
is that representation, plus three independent ways to solve it.

Having more than one solver matters here. The whole point of the equivalence
work is to catch a reformulation that quietly changes the answer, so the
solving step itself must not be a single unverified oracle. `solve_mip` runs
HiGHS and, when the model is small enough, also enumerates every binary
assignment and solves the resulting LPs -- two independent computations that
have to agree before a value is reported.
"""

from __future__ import annotations

import itertools
import time
from typing import Any, Iterable, Literal, Sequence

import numpy as np
from pydantic import BaseModel, Field


MipStatus = Literal[
    "optimal",
    "feasible",
    "infeasible",
    "unbounded",
    "no_solution",
    "error",
    "not_run",
]

ObjectiveSense = Literal["minimize", "maximize"]
ConstraintSense = Literal["<=", ">=", "=="]

DEFAULT_TOLERANCE = 1e-6
INF = float("inf")


class MipVariable(BaseModel):
    name: str
    lower: float = 0.0
    upper: float | None = None  # None means +infinity
    is_binary: bool = False

    @property
    def effective_upper(self) -> float:
        if self.is_binary:
            return 1.0
        return INF if self.upper is None else float(self.upper)

    @property
    def effective_lower(self) -> float:
        return 0.0 if self.is_binary else float(self.lower)


class MipConstraint(BaseModel):
    name: str
    coefficients: list[float]
    sense: ConstraintSense
    rhs: float

    def lhs(self, values: Sequence[float]) -> float:
        return float(sum(float(a) * float(v) for a, v in zip(self.coefficients, values)))

    def violation(self, values: Sequence[float]) -> float:
        """True violation magnitude; 0.0 when the constraint holds."""

        value = self.lhs(values)
        if self.sense == "<=":
            return max(0.0, value - self.rhs)
        if self.sense == ">=":
            return max(0.0, self.rhs - value)
        return abs(value - self.rhs)


class MixedIntegerProgram(BaseModel):
    """`min/max c'v` over linear constraints, with some variables binary."""

    name: str = "mip"
    sense: ObjectiveSense = "minimize"
    variables: list[MipVariable] = Field(default_factory=list)
    objective: list[float] = Field(default_factory=list)
    #: Constant term. A substitution such as `x = lower + sum w_k b_k` moves part
    #: of the objective into a constant, and dropping it would silently shift
    #: every reported value.
    objective_offset: float = 0.0
    constraints: list[MipConstraint] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def n_vars(self) -> int:
        return len(self.variables)

    @property
    def n_constraints(self) -> int:
        return len(self.constraints)

    @property
    def binary_indices(self) -> list[int]:
        return [i for i, v in enumerate(self.variables) if v.is_binary]

    @property
    def n_binaries(self) -> int:
        return len(self.binary_indices)

    def index_of(self, name: str) -> int:
        for index, variable in enumerate(self.variables):
            if variable.name == name:
                return index
        raise KeyError(f"No variable named `{name}` in `{self.name}`.")

    def objective_value(self, values: Sequence[float]) -> float:
        if len(values) != self.n_vars:
            raise ValueError(
                f"solution has {len(values)} entries but the model has {self.n_vars} variables"
            )
        return float(
            sum(float(c) * float(v) for c, v in zip(self.objective, values)) + self.objective_offset
        )

    def is_better(self, candidate: float, incumbent: float | None) -> bool:
        if incumbent is None:
            return True
        return candidate > incumbent if self.sense == "maximize" else candidate < incumbent

    # -- matrix views --------------------------------------------------------

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        lower = np.array([v.effective_lower for v in self.variables], dtype=float)
        upper = np.array([v.effective_upper for v in self.variables], dtype=float)
        return lower, upper

    def inequality_rows(self) -> tuple[np.ndarray, np.ndarray]:
        """`A_ub v <= b_ub`, with `>=` rows negated."""

        rows, rhs = [], []
        for constraint in self.constraints:
            coefficients = np.asarray(constraint.coefficients, dtype=float)
            if constraint.sense == "<=":
                rows.append(coefficients)
                rhs.append(float(constraint.rhs))
            elif constraint.sense == ">=":
                rows.append(-coefficients)
                rhs.append(-float(constraint.rhs))
        if not rows:
            return np.zeros((0, self.n_vars)), np.zeros(0)
        return np.vstack(rows), np.asarray(rhs, dtype=float)

    def equality_rows(self) -> tuple[np.ndarray, np.ndarray]:
        rows = [
            np.asarray(c.coefficients, dtype=float) for c in self.constraints if c.sense == "=="
        ]
        rhs = [float(c.rhs) for c in self.constraints if c.sense == "=="]
        if not rows:
            return np.zeros((0, self.n_vars)), np.zeros(0)
        return np.vstack(rows), np.asarray(rhs, dtype=float)


class ConstraintViolation(BaseModel):
    name: str
    sense: str
    lhs: float
    rhs: float
    violation: float


class MipVerification(BaseModel):
    """Independent check of a claimed MIP solution."""

    feasible: bool
    n_violated: int = 0
    max_violation: float = 0.0
    violations: list[ConstraintViolation] = Field(default_factory=list)
    domain_errors: list[str] = Field(default_factory=list)
    recomputed_objective: float | None = None
    tolerance: float = DEFAULT_TOLERANCE


class MipSolution(BaseModel):
    status: MipStatus = "not_run"
    objective: float | None = None
    values: list[float] = Field(default_factory=list)
    backend: str = "unknown"
    runtime_seconds: float = 0.0
    lp_solves: int = 0
    notes: str = ""

    @property
    def has_solution(self) -> bool:
        return self.status in ("optimal", "feasible") and bool(self.values)

    def add_note(self, note: str) -> None:
        note = note.strip()
        if not note:
            return
        self.notes = f"{self.notes} | {note}".strip(" |") if self.notes else note


def verify_mip_solution(
    mip: MixedIntegerProgram,
    values: Sequence[float] | None,
    tolerance: float = DEFAULT_TOLERANCE,
) -> MipVerification:
    """Substitute a claimed solution back into every constraint and bound."""

    if values is None or len(values) != mip.n_vars:
        return MipVerification(
            feasible=False,
            domain_errors=[
                "No solution vector was provided."
                if values is None
                else f"Solution length {len(values)} does not match {mip.n_vars} variables."
            ],
            tolerance=tolerance,
        )

    domain_errors: list[str] = []
    for index, (variable, value) in enumerate(zip(mip.variables, values)):
        numeric = float(value)
        if numeric < variable.effective_lower - tolerance:
            domain_errors.append(
                f"`{variable.name}`={numeric:.6g} is below its lower bound "
                f"{variable.effective_lower:.6g}."
            )
        if numeric > variable.effective_upper + tolerance:
            domain_errors.append(
                f"`{variable.name}`={numeric:.6g} is above its upper bound "
                f"{variable.effective_upper:.6g}."
            )
        if variable.is_binary and abs(numeric - round(numeric)) > 1e-5:
            domain_errors.append(f"binary `{variable.name}`={numeric:.6g} is not integral.")

    violations = []
    for constraint in mip.constraints:
        magnitude = constraint.violation(values)
        if magnitude > tolerance:
            violations.append(
                ConstraintViolation(
                    name=constraint.name,
                    sense=constraint.sense,
                    lhs=constraint.lhs(values),
                    rhs=float(constraint.rhs),
                    violation=float(magnitude),
                )
            )

    return MipVerification(
        feasible=not violations and not domain_errors,
        n_violated=len(violations),
        max_violation=max((v.violation for v in violations), default=0.0),
        violations=violations,
        domain_errors=domain_errors,
        recomputed_objective=mip.objective_value(values),
        tolerance=tolerance,
    )


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------


def _scipy_constraints(mip: MixedIntegerProgram):
    """One LinearConstraint covering every row, with per-row lb/ub."""

    from scipy.optimize import LinearConstraint

    if not mip.constraints:
        return []
    matrix, lower, upper = [], [], []
    for constraint in mip.constraints:
        matrix.append([float(a) for a in constraint.coefficients])
        if constraint.sense == "<=":
            lower.append(-np.inf)
            upper.append(float(constraint.rhs))
        elif constraint.sense == ">=":
            lower.append(float(constraint.rhs))
            upper.append(np.inf)
        else:
            lower.append(float(constraint.rhs))
            upper.append(float(constraint.rhs))
    return [LinearConstraint(np.asarray(matrix, dtype=float), np.asarray(lower), np.asarray(upper))]


def _objective_vector(mip: MixedIntegerProgram) -> tuple[np.ndarray, float]:
    """scipy always minimizes, so a maximization is negated. Returns (c, sign)."""

    coefficients = np.asarray(mip.objective, dtype=float)
    if mip.sense == "maximize":
        return -coefficients, -1.0
    return coefficients, 1.0


def solve_lp(
    mip: MixedIntegerProgram,
    fixed: dict[int, float] | None = None,
    relax_binaries: bool = True,
) -> MipSolution:
    """Solve the continuous relaxation, optionally with some variables pinned.

    With every binary pinned in `fixed`, this is an exact solve of that branch.
    """

    try:
        from scipy.optimize import linprog
    except Exception as exc:  # pragma: no cover - environment dependent
        return MipSolution(status="not_run", backend="scipy_linprog", notes=f"scipy missing: {exc}")

    started = time.perf_counter()
    lower, upper = mip.bounds()
    lower, upper = lower.copy(), upper.copy()
    for index, value in (fixed or {}).items():
        lower[index] = float(value)
        upper[index] = float(value)
    if not relax_binaries:
        for index in mip.binary_indices:
            if index not in (fixed or {}):
                raise ValueError("solve_lp cannot enforce integrality; pin binaries via `fixed`.")

    coefficients, sign = _objective_vector(mip)
    a_ub, b_ub = mip.inequality_rows()
    a_eq, b_eq = mip.equality_rows()
    result = linprog(
        coefficients,
        A_ub=a_ub if a_ub.shape[0] else None,
        b_ub=b_ub if a_ub.shape[0] else None,
        A_eq=a_eq if a_eq.shape[0] else None,
        b_eq=b_eq if a_eq.shape[0] else None,
        bounds=list(zip(lower, upper)),
        method="highs",
    )
    runtime = time.perf_counter() - started
    if result.status == 0:
        return MipSolution(
            status="optimal",
            objective=float(result.fun) * sign + mip.objective_offset,
            values=[float(v) for v in result.x],
            backend="scipy_linprog",
            runtime_seconds=runtime,
            lp_solves=1,
        )
    mapping = {2: "infeasible", 3: "unbounded", 1: "no_solution", 4: "error"}
    return MipSolution(
        status=mapping.get(int(result.status), "error"),
        backend="scipy_linprog",
        runtime_seconds=runtime,
        lp_solves=1,
        notes=str(getattr(result, "message", "")).strip(),
    )


def solve_mip_scipy(mip: MixedIntegerProgram, time_limit: float = 30.0) -> MipSolution:
    """HiGHS branch-and-bound through `scipy.optimize.milp`."""

    try:
        from scipy.optimize import Bounds, milp
    except Exception as exc:  # pragma: no cover - environment dependent
        return MipSolution(status="not_run", backend="scipy_milp", notes=f"scipy.milp missing: {exc}")

    started = time.perf_counter()
    coefficients, sign = _objective_vector(mip)
    lower, upper = mip.bounds()
    integrality = np.array([1 if v.is_binary else 0 for v in mip.variables])
    try:
        result = milp(
            c=coefficients,
            constraints=_scipy_constraints(mip),
            integrality=integrality,
            bounds=Bounds(lower, upper),
            options={"time_limit": float(time_limit)} if time_limit and time_limit > 0 else None,
        )
    except Exception as exc:
        return MipSolution(
            status="error",
            backend="scipy_milp",
            runtime_seconds=time.perf_counter() - started,
            notes=f"scipy.milp raised {type(exc).__name__}: {exc}",
        )
    runtime = time.perf_counter() - started

    if result.status == 0 and result.x is not None:
        return MipSolution(
            status="optimal",
            objective=float(result.fun) * sign + mip.objective_offset,
            values=[float(v) for v in result.x],
            backend="scipy_milp",
            runtime_seconds=runtime,
        )
    if result.status == 1 and result.x is not None:
        return MipSolution(
            status="feasible",
            objective=float(result.fun) * sign + mip.objective_offset,
            values=[float(v) for v in result.x],
            backend="scipy_milp",
            runtime_seconds=runtime,
            notes="stopped on the time/iteration limit; optimality is not proved",
        )
    mapping = {2: "infeasible", 3: "unbounded", 1: "no_solution", 4: "error"}
    return MipSolution(
        status=mapping.get(int(result.status), "error"),
        backend="scipy_milp",
        runtime_seconds=runtime,
        notes=str(getattr(result, "message", "")).strip(),
    )


def solve_mip_by_enumeration(
    mip: MixedIntegerProgram,
    max_binaries: int = 18,
    time_limit: float = 60.0,
) -> MipSolution:
    """Enumerate every binary assignment and solve the resulting LP.

    Exponential on purpose. It depends on no branch-and-bound implementation,
    which is what makes it usable as an independent check on HiGHS.
    """

    binaries = mip.binary_indices
    if len(binaries) > max_binaries:
        return MipSolution(
            status="not_run",
            backend="binary_enumeration",
            notes=(
                f"enumeration skipped: {len(binaries)} binaries exceeds max_binaries="
                f"{max_binaries} ({2 ** len(binaries):.3g} branches)."
            ),
        )

    started = time.perf_counter()
    best_value: float | None = None
    best_values: list[float] = []
    lp_solves = 0
    any_feasible = False
    unbounded = False
    hit_limit = False

    for assignment in itertools.product((0.0, 1.0), repeat=len(binaries)):
        if time_limit and (time.perf_counter() - started) > time_limit:
            hit_limit = True
            break
        branch = solve_lp(mip, fixed=dict(zip(binaries, assignment)))
        lp_solves += 1
        if branch.status == "unbounded":
            unbounded = True
            break
        if branch.status != "optimal" or branch.objective is None:
            continue
        any_feasible = True
        if mip.is_better(branch.objective, best_value):
            best_value = branch.objective
            best_values = list(branch.values)

    runtime = time.perf_counter() - started
    if unbounded:
        return MipSolution(
            status="unbounded",
            backend="binary_enumeration",
            runtime_seconds=runtime,
            lp_solves=lp_solves,
            notes="an LP branch was unbounded",
        )
    if best_value is None:
        return MipSolution(
            status="no_solution" if hit_limit else "infeasible",
            backend="binary_enumeration",
            runtime_seconds=runtime,
            lp_solves=lp_solves,
            notes=(
                "stopped on the time limit before finding a feasible branch"
                if hit_limit
                else f"all {lp_solves} binary branches were infeasible"
            ),
        )
    return MipSolution(
        status="feasible" if hit_limit else "optimal",
        objective=best_value,
        values=best_values,
        backend="binary_enumeration",
        runtime_seconds=runtime,
        lp_solves=lp_solves,
        notes=(
            "stopped on the time limit; optimality is NOT proved"
            if hit_limit
            else f"enumerated all {lp_solves} binary branches; optimality proved"
        ),
    )


_PYOMO_CANDIDATES = ("gurobi_direct", "gurobi", "appsi_highs", "cbc", "glpk", "scip")


def _pyomo_solver_name(preferred: str | None = None) -> str | None:
    try:
        import pyomo.environ as pyo
    except Exception:
        return None
    for name in (preferred,) if preferred else _PYOMO_CANDIDATES:
        if not name:
            continue
        try:
            factory = pyo.SolverFactory(name)
            if factory is not None and factory.available(exception_flag=False):
                return name
        except Exception:
            continue
    return None


def solve_mip_pyomo(
    mip: MixedIntegerProgram,
    time_limit: float = 30.0,
    solver_name: str | None = None,
) -> MipSolution:
    """Route the model through Pyomo to whichever MILP solver is installed."""

    resolved = _pyomo_solver_name(solver_name)
    if resolved is None:
        return MipSolution(
            status="not_run",
            backend="pyomo",
            notes="pyomo is unavailable or no MILP solver is installed",
        )
    import pyomo.environ as pyo

    started = time.perf_counter()
    model = pyo.ConcreteModel()
    model.I = pyo.RangeSet(0, mip.n_vars - 1)

    def _domain(index: int):
        return pyo.Binary if mip.variables[index].is_binary else pyo.Reals

    def _bounds(_, index: int):
        variable = mip.variables[index]
        upper = variable.effective_upper
        return (variable.effective_lower, None if upper == INF else upper)

    model.v = pyo.Var(model.I, domain=pyo.Reals, bounds=_bounds)
    for index in mip.binary_indices:
        model.v[index].domain = pyo.Binary
    model.objective = pyo.Objective(
        expr=sum(float(mip.objective[i]) * model.v[i] for i in model.I) + mip.objective_offset,
        sense=pyo.maximize if mip.sense == "maximize" else pyo.minimize,
    )
    model.rows = pyo.ConstraintList()
    for constraint in mip.constraints:
        expression = sum(
            float(a) * model.v[i] for i, a in enumerate(constraint.coefficients) if float(a) != 0.0
        )
        if constraint.sense == "<=":
            model.rows.add(expression <= float(constraint.rhs))
        elif constraint.sense == ">=":
            model.rows.add(expression >= float(constraint.rhs))
        else:
            model.rows.add(expression == float(constraint.rhs))

    solver = pyo.SolverFactory(resolved)
    for option in ("TimeLimit", "sec", "tmlim", "limits/time", "time_limit"):
        try:
            solver.options[option] = float(time_limit)
            break
        except Exception:  # pragma: no cover - solver specific
            continue
    try:
        results = solver.solve(model, tee=False, load_solutions=True)
    except Exception as exc:
        return MipSolution(
            status="error",
            backend=f"pyomo:{resolved}",
            runtime_seconds=time.perf_counter() - started,
            notes=f"{resolved} raised: {exc}",
        )
    runtime = time.perf_counter() - started
    condition = str(getattr(results.solver, "termination_condition", "unknown")).lower()
    mapping = {
        "optimal": "optimal",
        "globallyoptimal": "optimal",
        "locallyoptimal": "feasible",
        "feasible": "feasible",
        "maxtimelimit": "feasible",
        "infeasible": "infeasible",
        "infeasibleorunbounded": "infeasible",
        "unbounded": "unbounded",
    }
    status: MipStatus = mapping.get(condition, "no_solution")  # type: ignore[assignment]
    if status in ("optimal", "feasible"):
        try:
            values = [float(pyo.value(model.v[i])) for i in model.I]
            objective = float(pyo.value(model.objective))
        except Exception as exc:
            return MipSolution(
                status="error",
                backend=f"pyomo:{resolved}",
                runtime_seconds=runtime,
                notes=f"solution could not be read: {exc}",
            )
        return MipSolution(
            status=status,
            objective=objective,
            values=values,
            backend=f"pyomo:{resolved}",
            runtime_seconds=runtime,
            notes=f"termination_condition={condition}",
        )
    return MipSolution(
        status=status,
        backend=f"pyomo:{resolved}",
        runtime_seconds=runtime,
        notes=f"termination_condition={condition}",
    )


def available_mip_backends() -> dict[str, tuple[bool, str]]:
    backends: dict[str, tuple[bool, str]] = {}
    try:
        from scipy.optimize import milp  # noqa: F401

        backends["scipy_milp"] = (True, "")
    except Exception as exc:
        backends["scipy_milp"] = (False, f"scipy.optimize.milp unavailable: {exc}")
    try:
        from scipy.optimize import linprog  # noqa: F401

        backends["binary_enumeration"] = (True, "")
    except Exception as exc:
        backends["binary_enumeration"] = (False, f"scipy.optimize.linprog unavailable: {exc}")
    resolved = _pyomo_solver_name()
    backends["pyomo"] = (
        (True, "") if resolved else (False, "pyomo is unavailable or no MILP solver is installed")
    )
    return backends


def solve_mip(
    mip: MixedIntegerProgram,
    backend: str = "auto",
    time_limit: float = 30.0,
    cross_check: bool = True,
    cross_check_max_binaries: int = 14,
    tolerance: float = 1e-6,
) -> MipSolution:
    """Solve, verify the solution, and cross-check the value against a second method.

    A disagreement between branch-and-bound and exhaustive binary enumeration
    is reported as `error`, never averaged away: on this project the reported
    optimum of a reformulated model *is* the experimental result, so an
    unreconciled second opinion invalidates it.
    """

    if backend == "auto":
        primary_fn = solve_mip_scipy
        if not available_mip_backends()["scipy_milp"][0]:
            primary_fn = lambda m, time_limit=time_limit: solve_mip_by_enumeration(  # noqa: E731
                m, time_limit=time_limit
            )
        solution = primary_fn(mip, time_limit=time_limit)
    elif backend == "scipy_milp":
        solution = solve_mip_scipy(mip, time_limit=time_limit)
    elif backend == "binary_enumeration":
        solution = solve_mip_by_enumeration(mip, time_limit=time_limit)
    elif backend == "pyomo":
        solution = solve_mip_pyomo(mip, time_limit=time_limit)
    else:
        return MipSolution(status="error", backend=backend, notes=f"unknown backend `{backend}`")

    if solution.has_solution:
        verification = verify_mip_solution(mip, solution.values, tolerance=tolerance)
        if not verification.feasible:
            detail = (
                verification.domain_errors[0]
                if verification.domain_errors
                else f"`{verification.violations[0].name}` violated by {verification.max_violation:.6g}"
            )
            solution.status = "error"
            solution.add_note(
                f"VERIFICATION FAILURE: {solution.backend} reported a solution that does not "
                f"satisfy the model ({detail})."
            )
            return solution
        if (
            verification.recomputed_objective is not None
            and solution.objective is not None
            and abs(verification.recomputed_objective - solution.objective)
            > 1e-4 * max(1.0, abs(verification.recomputed_objective))
        ):
            solution.add_note(
                f"objective recomputed from the model is {verification.recomputed_objective:.9g}, "
                f"not the reported {solution.objective:.9g}; using the recomputed value"
            )
        if verification.recomputed_objective is not None:
            solution.objective = verification.recomputed_objective

    if (
        cross_check
        and solution.backend != "binary_enumeration"
        and mip.n_binaries <= cross_check_max_binaries
    ):
        second = solve_mip_by_enumeration(mip, max_binaries=cross_check_max_binaries, time_limit=time_limit)
        if second.status in ("optimal", "infeasible", "unbounded"):
            if solution.status in ("optimal", "feasible") and second.status == "optimal":
                if second.objective is not None and solution.objective is not None:
                    delta = abs(second.objective - solution.objective)
                    if delta > tolerance * max(1.0, abs(second.objective)):
                        solution.status = "error"
                        solution.add_note(
                            f"CROSS-CHECK FAILURE: {solution.backend} reported "
                            f"{solution.objective:.9g} but exhaustive binary enumeration proves "
                            f"{second.objective:.9g} (difference {delta:.3g})."
                        )
                        return solution
                    solution.add_note(
                        f"cross-checked against exhaustive enumeration over {second.lp_solves} branches"
                    )
            elif solution.status != second.status:
                solution.status = "error"
                solution.add_note(
                    f"CROSS-CHECK FAILURE: {solution.backend} reported status `{solution.status}` "
                    f"but exhaustive binary enumeration reports `{second.status}`."
                )
                return solution
    return solution
