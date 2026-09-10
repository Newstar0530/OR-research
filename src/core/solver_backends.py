"""Real solver backends for 0-1 integer programs, plus one verified entry point.

Design rules, in priority order:

1. **Nothing is trusted.** Every backend's answer goes through
   `verify_solver_claim` before it becomes a `SolveResult`, and the objective
   written to the results table is the one *recomputed from the instance data*,
   not the one the backend printed.
2. **Absence degrades, it does not crash.** A missing solver yields
   `solver_status="not_run"` with a reason, so an experiment can still run and
   the report can still say what was unavailable.
3. **Exactness is labelled.** Only a backend that actually proves optimality is
   allowed to return `optimal`; heuristics top out at `feasible`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Literal, Sequence

import numpy as np

from src.core.binary_program import BinaryProgramInstance
from src.core.solution_checker import DEFAULT_TOLERANCE, verify_solver_claim
from src.core.solve_result import SolveResult, SolverStatus


BackendKind = Literal["exact", "heuristic"]

_SCALES = (1, 10, 100, 1_000, 10_000, 100_000, 1_000_000)


@dataclass
class BackendOutcome:
    """Raw, unverified output of one backend."""

    status: SolverStatus = "not_run"
    objective: float | None = None
    dual_bound: float | None = None
    solution: list[int] = field(default_factory=list)
    node_count: int | None = None
    runtime_seconds: float = 0.0
    hit_time_limit: bool = False
    notes: str = ""


def _integer_scale(values: Iterable[float], tolerance: float = 1e-9) -> tuple[int, bool]:
    """Smallest power-of-ten scale making every value integral. (scale, exact)."""

    values = [float(v) for v in values]
    for scale in _SCALES:
        if all(abs(v * scale - round(v * scale)) <= tolerance * max(1.0, abs(v * scale)) for v in values):
            return scale, True
    return _SCALES[-1], False


# ---------------------------------------------------------------------------
# Backend base
# ---------------------------------------------------------------------------


class SolverBackend:
    name: str = "abstract"
    kind: BackendKind = "heuristic"
    description: str = ""

    def availability(self) -> tuple[bool, str]:
        return True, ""

    def solve(
        self,
        instance: BinaryProgramInstance,
        time_limit: float = 10.0,
        seed: int = 0,
    ) -> BackendOutcome:  # pragma: no cover - abstract
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Exact: exhaustive enumeration (ground truth for small instances)
# ---------------------------------------------------------------------------


class BruteForceBackend(SolverBackend):
    """Chunked, vectorised enumeration of all 2**n binary vectors.

    This is the only backend that can establish ground truth without trusting
    any third-party solver, which is exactly why it is worth having: it is the
    reference used to check that CP-SAT/Pyomo are configured correctly.
    """

    name = "brute_force"
    kind = "exact"
    description = "Exhaustive enumeration; proves optimality for small instances."

    def __init__(self, max_vars: int = 22, chunk_size: int = 1 << 16) -> None:
        self.max_vars = int(max_vars)
        self.chunk_size = int(chunk_size)

    def availability(self) -> tuple[bool, str]:
        return True, ""

    def solve(
        self,
        instance: BinaryProgramInstance,
        time_limit: float = 10.0,
        seed: int = 0,
    ) -> BackendOutcome:
        n = instance.n_vars
        if n > self.max_vars:
            return BackendOutcome(
                status="not_run",
                notes=(
                    f"brute_force skipped: {n} variables exceeds max_vars={self.max_vars} "
                    f"({2 ** n:.3g} candidates)."
                ),
            )
        started = time.perf_counter()
        costs = np.asarray(instance.objective_coefficients, dtype=float)
        rows = [np.asarray(c.coefficients, dtype=float) for c in instance.constraints]
        senses = [c.sense for c in instance.constraints]
        rhs = np.asarray([float(c.rhs) for c in instance.constraints], dtype=float)
        matrix = np.vstack(rows) if rows else np.zeros((0, n))
        maximize = instance.objective_sense == "maximize"
        bit_weights = (1 << np.arange(n, dtype=np.uint64)).astype(np.uint64)

        best_value: float | None = None
        best_vector: list[int] = []
        total = 1 << n
        hit_limit = False
        for start in range(0, total, self.chunk_size):
            if time_limit and (time.perf_counter() - started) > time_limit:
                hit_limit = True
                break
            stop = min(total, start + self.chunk_size)
            candidates = np.arange(start, stop, dtype=np.uint64)
            # bits[k, j] = j-th bit of candidate k
            bits = ((candidates[:, None] & bit_weights[None, :]) > 0).astype(float)
            if matrix.shape[0]:
                lhs = bits @ matrix.T  # (chunk, n_constraints)
                ok = np.ones(bits.shape[0], dtype=bool)
                for index, sense in enumerate(senses):
                    column = lhs[:, index]
                    if sense == "<=":
                        ok &= column <= rhs[index] + DEFAULT_TOLERANCE
                    elif sense == ">=":
                        ok &= column >= rhs[index] - DEFAULT_TOLERANCE
                    else:
                        ok &= np.abs(column - rhs[index]) <= DEFAULT_TOLERANCE
            else:
                ok = np.ones(bits.shape[0], dtype=bool)
            if not ok.any():
                continue
            feasible_bits = bits[ok]
            values = feasible_bits @ costs
            position = int(np.argmax(values) if maximize else np.argmin(values))
            value = float(values[position])
            if best_value is None or (value > best_value if maximize else value < best_value):
                best_value = value
                best_vector = [int(v) for v in feasible_bits[position]]

        runtime = time.perf_counter() - started
        if best_value is None:
            return BackendOutcome(
                status="no_solution" if hit_limit else "infeasible",
                runtime_seconds=runtime,
                hit_time_limit=hit_limit,
                notes=(
                    "brute_force stopped on the time limit before finding a feasible point."
                    if hit_limit
                    else "brute_force enumerated every candidate: the instance is infeasible."
                ),
            )
        return BackendOutcome(
            status="time_limit" if hit_limit else "optimal",
            objective=best_value,
            dual_bound=None if hit_limit else best_value,
            solution=best_vector,
            node_count=total if not hit_limit else None,
            runtime_seconds=runtime,
            hit_time_limit=hit_limit,
            notes=(
                "brute_force hit the time limit; optimality is NOT proved."
                if hit_limit
                else f"brute_force enumerated all {total} candidates; optimality proved."
            ),
        )


# ---------------------------------------------------------------------------
# Exact: CP-SAT
# ---------------------------------------------------------------------------


class CpSatBackend(SolverBackend):
    """OR-Tools CP-SAT. Needs no external solver executable."""

    name = "cp_sat"
    kind = "exact"
    description = "OR-Tools CP-SAT exact solver for integral 0-1 programs."

    def __init__(self, workers: int = 1) -> None:
        self.workers = int(workers)

    def availability(self) -> tuple[bool, str]:
        try:
            from ortools.sat.python import cp_model  # noqa: F401
        except Exception as exc:  # pragma: no cover - environment dependent
            return False, f"ortools is not importable: {exc}"
        return True, ""

    def solve(
        self,
        instance: BinaryProgramInstance,
        time_limit: float = 10.0,
        seed: int = 0,
    ) -> BackendOutcome:
        available, reason = self.availability()
        if not available:
            return BackendOutcome(status="not_run", notes=reason)
        from ortools.sat.python import cp_model

        started = time.perf_counter()
        notes: list[str] = []
        model = cp_model.CpModel()
        variables = [model.NewBoolVar(f"x[{j}]") for j in range(instance.n_vars)]

        for constraint in instance.constraints:
            scale, exact = _integer_scale(list(constraint.coefficients) + [constraint.rhs])
            if not exact:
                notes.append(
                    f"constraint `{constraint.name}` was rounded at scale {scale}; "
                    "results may lose precision."
                )
            terms = [
                int(round(float(a) * scale)) * variables[j]
                for j, a in enumerate(constraint.coefficients)
                if round(float(a) * scale) != 0
            ]
            bound = int(round(float(constraint.rhs) * scale))
            expression = sum(terms) if terms else 0
            if constraint.sense == "<=":
                model.Add(expression <= bound)
            elif constraint.sense == ">=":
                model.Add(expression >= bound)
            else:
                model.Add(expression == bound)

        obj_scale, obj_exact = _integer_scale(instance.objective_coefficients)
        if not obj_exact:
            notes.append(f"objective was rounded at scale {obj_scale}; results may lose precision.")
        objective_terms = [
            int(round(float(c) * obj_scale)) * variables[j]
            for j, c in enumerate(instance.objective_coefficients)
            if round(float(c) * obj_scale) != 0
        ]
        objective_expression = sum(objective_terms) if objective_terms else 0
        if instance.objective_sense == "maximize":
            model.Maximize(objective_expression)
        else:
            model.Minimize(objective_expression)

        solver = cp_model.CpSolver()
        if time_limit and time_limit > 0:
            solver.parameters.max_time_in_seconds = float(time_limit)
        solver.parameters.num_workers = self.workers
        solver.parameters.random_seed = int(seed) % (2**31 - 1)
        status = solver.Solve(model)
        runtime = time.perf_counter() - started

        has_solution = status in (cp_model.OPTIMAL, cp_model.FEASIBLE)
        solution = [int(solver.Value(v)) for v in variables] if has_solution else []
        objective = float(solver.ObjectiveValue()) / obj_scale if has_solution else None
        try:
            dual_bound = float(solver.BestObjectiveBound()) / obj_scale if has_solution else None
        except Exception:  # pragma: no cover - defensive
            dual_bound = None

        hit_limit = bool(
            time_limit and runtime >= float(time_limit) * 0.98 and status != cp_model.OPTIMAL
        )
        if status == cp_model.OPTIMAL:
            mapped: SolverStatus = "optimal"
        elif status == cp_model.FEASIBLE:
            mapped = "time_limit" if hit_limit else "feasible"
        elif status == cp_model.INFEASIBLE:
            mapped = "infeasible"
        elif status == cp_model.MODEL_INVALID:
            mapped = "error"
            notes.append("CP-SAT reported MODEL_INVALID.")
        else:
            mapped = "no_solution"

        try:
            node_count: int | None = int(solver.NumBranches())
        except Exception:  # pragma: no cover - defensive
            node_count = None

        return BackendOutcome(
            status=mapped,
            objective=objective,
            dual_bound=dual_bound,
            solution=solution,
            node_count=node_count,
            runtime_seconds=runtime,
            hit_time_limit=hit_limit,
            notes="; ".join(notes),
        )


# ---------------------------------------------------------------------------
# Exact: Pyomo + whichever MILP solver is installed
# ---------------------------------------------------------------------------


_PYOMO_CANDIDATES: tuple[tuple[str, str, str], ...] = (
    # (solver name, time-limit option key, human label)
    ("gurobi_direct", "TimeLimit", "Gurobi (direct)"),
    ("gurobi", "TimeLimit", "Gurobi"),
    ("appsi_highs", "time_limit", "HiGHS (appsi)"),
    ("cbc", "sec", "CBC"),
    ("glpk", "tmlim", "GLPK"),
    ("scip", "limits/time", "SCIP"),
    ("cplex_direct", "timelimit", "CPLEX (direct)"),
)


class PyomoBackend(SolverBackend):
    """Algebraic model through Pyomo, solved by the first available MILP solver."""

    name = "pyomo"
    kind = "exact"
    description = "Pyomo model dispatched to Gurobi/HiGHS/CBC/GLPK/SCIP, whichever is installed."

    def __init__(self, solver_name: str | None = None) -> None:
        self.solver_name = solver_name
        self._resolved: tuple[str, str, str] | None = None

    def _resolve(self) -> tuple[str, str, str] | None:
        if self._resolved is not None:
            return self._resolved
        try:
            import pyomo.environ as pyo
        except Exception:
            return None
        candidates = (
            tuple(c for c in _PYOMO_CANDIDATES if c[0] == self.solver_name)
            if self.solver_name
            else _PYOMO_CANDIDATES
        )
        for candidate in candidates:
            try:
                factory = pyo.SolverFactory(candidate[0])
                if factory is not None and factory.available(exception_flag=False):
                    self._resolved = candidate
                    return candidate
            except Exception:
                continue
        return None

    def availability(self) -> tuple[bool, str]:
        try:
            import pyomo.environ  # noqa: F401
        except Exception as exc:
            return False, f"pyomo is not importable: {exc}"
        resolved = self._resolve()
        if resolved is None:
            return False, (
                "pyomo is installed but no MILP solver is available "
                f"(tried: {', '.join(c[0] for c in _PYOMO_CANDIDATES)})."
            )
        return True, ""

    def solve(
        self,
        instance: BinaryProgramInstance,
        time_limit: float = 10.0,
        seed: int = 0,
    ) -> BackendOutcome:
        available, reason = self.availability()
        if not available:
            return BackendOutcome(status="not_run", notes=reason)
        import pyomo.environ as pyo

        solver_name, time_option, label = self._resolve()  # type: ignore[misc]
        started = time.perf_counter()
        model = pyo.ConcreteModel()
        model.J = pyo.RangeSet(0, instance.n_vars - 1)
        model.x = pyo.Var(model.J, domain=pyo.Binary)
        sense = pyo.maximize if instance.objective_sense == "maximize" else pyo.minimize
        coefficients = [float(c) for c in instance.objective_coefficients]
        model.objective = pyo.Objective(
            expr=sum(coefficients[j] * model.x[j] for j in model.J), sense=sense
        )
        model.constraints = pyo.ConstraintList()
        for constraint in instance.constraints:
            expression = sum(
                float(a) * model.x[j] for j, a in enumerate(constraint.coefficients) if float(a) != 0.0
            )
            if constraint.sense == "<=":
                model.constraints.add(expression <= float(constraint.rhs))
            elif constraint.sense == ">=":
                model.constraints.add(expression >= float(constraint.rhs))
            else:
                model.constraints.add(expression == float(constraint.rhs))

        solver = pyo.SolverFactory(solver_name)
        if time_limit and time_limit > 0:
            try:
                solver.options[time_option] = float(time_limit)
            except Exception:  # pragma: no cover - solver specific
                pass
        try:
            results = solver.solve(model, tee=False, load_solutions=True)
        except Exception as exc:
            return BackendOutcome(
                status="error",
                runtime_seconds=time.perf_counter() - started,
                notes=f"{label} raised: {exc}",
            )
        runtime = time.perf_counter() - started

        condition = str(getattr(results.solver, "termination_condition", "unknown")).lower()
        mapping: dict[str, SolverStatus] = {
            "optimal": "optimal",
            "globallyoptimal": "optimal",
            "locallyoptimal": "feasible",
            "feasible": "feasible",
            "maxtimelimit": "time_limit",
            "maxiterations": "time_limit",
            "infeasible": "infeasible",
            "infeasibleorunbounded": "infeasible",
            "unbounded": "unbounded",
        }
        mapped: SolverStatus = mapping.get(condition, "no_solution")

        solution: list[int] = []
        objective: float | None = None
        if mapped in ("optimal", "feasible", "time_limit"):
            try:
                solution = [int(round(float(pyo.value(model.x[j])))) for j in model.J]
                objective = float(pyo.value(model.objective))
            except Exception as exc:
                return BackendOutcome(
                    status="error",
                    runtime_seconds=runtime,
                    notes=f"{label} finished as `{condition}` but its solution could not be read: {exc}",
                )

        dual_bound: float | None = None
        try:
            problem = results.problem[0]
            candidate = (
                problem.upper_bound if instance.objective_sense == "maximize" else problem.lower_bound
            )
            if candidate is not None and np.isfinite(float(candidate)):
                dual_bound = float(candidate)
        except Exception:  # pragma: no cover - solver specific
            dual_bound = None
        if dual_bound is None and mapped == "optimal":
            dual_bound = objective

        return BackendOutcome(
            status=mapped,
            objective=objective,
            dual_bound=dual_bound,
            solution=solution,
            node_count=None,
            runtime_seconds=runtime,
            hit_time_limit=mapped == "time_limit",
            notes=f"solved by {label} via pyomo (termination_condition={condition})",
        )


# ---------------------------------------------------------------------------
# Heuristics
# ---------------------------------------------------------------------------


class _ConstraintState:
    """Incremental constraint bookkeeping so flips are O(m), not O(m*n)."""

    def __init__(self, instance: BinaryProgramInstance, solution: Sequence[int]) -> None:
        n = instance.n_vars
        self.n = n
        self.matrix = (
            np.vstack([np.asarray(c.coefficients, dtype=float) for c in instance.constraints])
            if instance.constraints
            else np.zeros((0, n))
        )
        self.senses = [c.sense for c in instance.constraints]
        self.rhs = np.asarray([float(c.rhs) for c in instance.constraints], dtype=float)
        self.is_upper = np.asarray([s == "<=" for s in self.senses], dtype=bool)
        self.is_lower = np.asarray([s == ">=" for s in self.senses], dtype=bool)
        self.is_equal = np.asarray([s == "==" for s in self.senses], dtype=bool)
        self.x = np.asarray(list(solution), dtype=float)
        self.lhs = self.matrix @ self.x if self.matrix.shape[0] else np.zeros(0)

    def _violation(self, lhs: np.ndarray) -> np.ndarray:
        if lhs.size == 0:
            return lhs
        upper = np.where(self.is_upper, np.maximum(0.0, lhs - self.rhs), 0.0)
        lower = np.where(self.is_lower, np.maximum(0.0, self.rhs - lhs), 0.0)
        equal = np.where(self.is_equal, np.abs(lhs - self.rhs), 0.0)
        return upper + lower + equal

    @property
    def violation(self) -> np.ndarray:
        return self._violation(self.lhs)

    def total_violation(self) -> float:
        return float(self.violation.sum())

    def hard_deficit(self) -> float:
        """Violation of `>=` and `==` constraints only (what repair must fix)."""

        if self.lhs.size == 0:
            return 0.0
        lower = np.where(self.is_lower, np.maximum(0.0, self.rhs - self.lhs), 0.0)
        equal = np.where(self.is_equal, np.abs(self.lhs - self.rhs), 0.0)
        return float((lower + equal).sum())

    def lhs_after_flip(self, j: int) -> np.ndarray:
        if self.matrix.shape[0] == 0:
            return self.lhs
        delta = 1.0 - 2.0 * self.x[j]  # +1 when turning on, -1 when turning off
        return self.lhs + delta * self.matrix[:, j]

    def is_feasible_after_flip(self, j: int, tolerance: float = DEFAULT_TOLERANCE) -> bool:
        return bool(self._violation(self.lhs_after_flip(j)).max(initial=0.0) <= tolerance)

    def hard_deficit_after_flip(self, j: int) -> float:
        lhs = self.lhs_after_flip(j)
        if lhs.size == 0:
            return 0.0
        lower = np.where(self.is_lower, np.maximum(0.0, self.rhs - lhs), 0.0)
        equal = np.where(self.is_equal, np.abs(lhs - self.rhs), 0.0)
        return float((lower + equal).sum())

    def upper_violation_after_flip(self, j: int) -> float:
        lhs = self.lhs_after_flip(j)
        if lhs.size == 0:
            return 0.0
        return float(np.where(self.is_upper, np.maximum(0.0, lhs - self.rhs), 0.0).sum())

    def flip(self, j: int) -> None:
        if self.matrix.shape[0]:
            delta = 1.0 - 2.0 * self.x[j]
            self.lhs = self.lhs + delta * self.matrix[:, j]
        self.x[j] = 1.0 - self.x[j]

    def slack(self) -> np.ndarray:
        return self.rhs - self.lhs

    def solution(self) -> list[int]:
        return [int(round(v)) for v in self.x]

    def is_feasible(self, tolerance: float = DEFAULT_TOLERANCE) -> bool:
        return bool(self.violation.max(initial=0.0) <= tolerance)


def _repair(state: _ConstraintState, costs: np.ndarray, direction: float) -> None:
    """Turn variables on until `>=`/`==` constraints hold (or no progress is possible)."""

    guard = 0
    while state.hard_deficit() > DEFAULT_TOLERANCE and guard <= state.n:
        guard += 1
        current = state.hard_deficit()
        best: tuple[int, int, float] | None = None  # (tier, index, score)
        for j in range(state.n):
            if state.x[j] >= 0.5:
                continue
            gain = current - state.hard_deficit_after_flip(j)
            if gain <= DEFAULT_TOLERANCE:
                continue
            # tier 0 keeps every `<=` constraint satisfied; tier 1 does not.
            tier = 0 if state.upper_violation_after_flip(j) <= DEFAULT_TOLERANCE else 1
            cost = max(0.0, -direction * float(costs[j]))
            score = gain / (1.0 + cost)
            if best is None or (tier, -score) < (best[0], -best[2]):
                best = (tier, j, score)
        if best is None:
            return
        state.flip(best[1])


def _greedy_construct(instance: BinaryProgramInstance) -> tuple[list[int], str]:
    """Deterministic greedy construction: repair, then insert, then prune.

    Insertion uses the classic value-density rule (objective gain per unit of
    scarce resource consumed); pruning removes variables whose removal saves
    objective without breaking anything. Each phase is a single sweep-to-
    exhaustion, so the result is a *construction*, not a local optimum -- that
    separation is what makes `local_search` a meaningful comparison.
    """

    costs = np.asarray(instance.objective_coefficients, dtype=float)
    direction = 1.0 if instance.objective_sense == "maximize" else -1.0
    state = _ConstraintState(instance, [0] * instance.n_vars)
    _repair(state, costs, direction)

    # Insertion phase: variables whose activation improves the objective.
    guard = 0
    while guard <= state.n:
        guard += 1
        best_j, best_density = None, 0.0
        slack = state.slack()
        for j in range(state.n):
            if state.x[j] >= 0.5:
                continue
            gain = direction * float(costs[j])
            if gain <= 0:
                continue
            if not state.is_feasible_after_flip(j):
                continue
            pressure = 0.0
            if state.matrix.shape[0]:
                column = state.matrix[:, j]
                mask = state.is_upper & (column > 0)
                if mask.any():
                    pressure = float(
                        np.sum(column[mask] / np.maximum(slack[mask], DEFAULT_TOLERANCE))
                    )
            density = gain / (1.0 + pressure)
            if density > best_density:
                best_j, best_density = j, density
        if best_j is None:
            break
        state.flip(best_j)

    # Pruning phase: variables whose deactivation improves the objective.
    guard = 0
    while guard <= state.n:
        guard += 1
        best_j, best_gain = None, 0.0
        for j in range(state.n):
            if state.x[j] < 0.5:
                continue
            gain = -direction * float(costs[j])
            if gain <= 0:
                continue
            if not state.is_feasible_after_flip(j):
                continue
            if gain > best_gain:
                best_j, best_gain = j, gain
        if best_j is None:
            break
        state.flip(best_j)

    note = "greedy construction: repair -> density insertion -> redundancy pruning"
    if not state.is_feasible():
        note += "; WARNING: construction ended infeasible"
    return state.solution(), note


class GreedyBackend(SolverBackend):
    """Deterministic greedy construction heuristic (the honest baseline)."""

    name = "greedy"
    kind = "heuristic"
    description = "Repair + value-density insertion + redundancy pruning. No search."

    def solve(
        self,
        instance: BinaryProgramInstance,
        time_limit: float = 10.0,
        seed: int = 0,
    ) -> BackendOutcome:
        started = time.perf_counter()
        solution, note = _greedy_construct(instance)
        state = _ConstraintState(instance, solution)
        feasible = state.is_feasible()
        return BackendOutcome(
            status="feasible" if feasible else "no_solution",
            objective=instance.objective_value(solution) if feasible else None,
            dual_bound=None,
            solution=solution if feasible else [],
            runtime_seconds=time.perf_counter() - started,
            notes=note,
        )


class LocalSearchBackend(SolverBackend):
    """Greedy start, then 1-flip and 1-1 swap hill climbing with restarts.

    Starts from the greedy solution and keeps it as the incumbent, so this
    method can never report a worse objective than `greedy` on the same
    instance. It never claims `optimal`.
    """

    name = "local_search"
    kind = "heuristic"
    description = "Greedy start + best-improvement 1-flip and 1-1 swap, with seeded restarts."

    def __init__(self, restarts: int = 3, perturbation: int = 3) -> None:
        self.restarts = max(0, int(restarts))
        self.perturbation = max(1, int(perturbation))

    def _climb(
        self,
        instance: BinaryProgramInstance,
        state: _ConstraintState,
        costs: np.ndarray,
        direction: float,
        deadline: float | None,
    ) -> None:
        while True:
            if deadline is not None and time.perf_counter() > deadline:
                return
            best_move: tuple[float, tuple[int, ...]] | None = None
            # 1-flip
            for j in range(state.n):
                delta = direction * float(costs[j]) * (1.0 - 2.0 * float(state.x[j]))
                if delta <= DEFAULT_TOLERANCE:
                    continue
                if not state.is_feasible_after_flip(j):
                    continue
                if best_move is None or delta > best_move[0]:
                    best_move = (delta, (j,))
            if best_move is None:
                # 1-1 swap: turn one variable off and another on.
                ones = [j for j in range(state.n) if state.x[j] >= 0.5]
                zeros = [j for j in range(state.n) if state.x[j] < 0.5]
                for out_j in ones:
                    state.flip(out_j)
                    for in_j in zeros:
                        delta = direction * (float(costs[in_j]) - float(costs[out_j]))
                        if delta <= DEFAULT_TOLERANCE:
                            continue
                        if not state.is_feasible_after_flip(in_j):
                            continue
                        if best_move is None or delta > best_move[0]:
                            best_move = (delta, (out_j, in_j))
                    state.flip(out_j)
                    if deadline is not None and time.perf_counter() > deadline:
                        break
            if best_move is None:
                return
            for index in best_move[1]:
                state.flip(index)

    def solve(
        self,
        instance: BinaryProgramInstance,
        time_limit: float = 10.0,
        seed: int = 0,
    ) -> BackendOutcome:
        started = time.perf_counter()
        deadline = started + float(time_limit) if time_limit and time_limit > 0 else None
        costs = np.asarray(instance.objective_coefficients, dtype=float)
        direction = 1.0 if instance.objective_sense == "maximize" else -1.0
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), instance.n_vars]))

        start_solution, _ = _greedy_construct(instance)
        state = _ConstraintState(instance, start_solution)
        best_solution: list[int] | None = None
        best_value: float | None = None
        if state.is_feasible():
            best_solution = state.solution()
            best_value = instance.objective_value(best_solution)

        self._climb(instance, state, costs, direction, deadline)
        if state.is_feasible():
            value = instance.objective_value(state.solution())
            if instance.is_better(value, best_value):
                best_solution, best_value = state.solution(), value

        restarts_done = 0
        for restart in range(self.restarts):
            if deadline is not None and time.perf_counter() > deadline:
                break
            restarts_done += 1
            perturbed = list(best_solution or start_solution)
            for j in rng.choice(
                instance.n_vars, size=min(self.perturbation, instance.n_vars), replace=False
            ):
                perturbed[int(j)] = 1 - perturbed[int(j)]
            trial = _ConstraintState(instance, perturbed)
            _repair(trial, costs, direction)
            self._climb(instance, trial, costs, direction, deadline)
            if trial.is_feasible():
                value = instance.objective_value(trial.solution())
                if instance.is_better(value, best_value):
                    best_solution, best_value = trial.solution(), value

        runtime = time.perf_counter() - started
        hit_limit = bool(deadline is not None and time.perf_counter() > deadline)
        if best_solution is None:
            return BackendOutcome(
                status="no_solution",
                runtime_seconds=runtime,
                hit_time_limit=hit_limit,
                notes="local_search could not construct any feasible solution.",
            )
        return BackendOutcome(
            status="feasible",
            objective=best_value,
            dual_bound=None,
            solution=best_solution,
            runtime_seconds=runtime,
            hit_time_limit=hit_limit,
            notes=(
                f"local_search: greedy start + hill climbing, {restarts_done} restart(s); "
                "optimality is not claimed"
            ),
        )


# ---------------------------------------------------------------------------
# Method registry and the verified entry point
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MethodSpec:
    """A named method as an experiment refers to it."""

    name: str
    backend: SolverBackend
    role: Literal["reference", "exact", "baseline", "proposed"]
    description: str


def build_method_specs(
    brute_force_max_vars: int = 22,
    local_search_restarts: int = 3,
    cp_sat_workers: int = 1,
    pyomo_solver: str | None = None,
) -> dict[str, MethodSpec]:
    """Fresh, independently configurable method table."""

    specs = [
        MethodSpec(
            "brute_force",
            BruteForceBackend(max_vars=brute_force_max_vars),
            "reference",
            "Exhaustive enumeration; ground truth for small instances.",
        ),
        MethodSpec(
            "exact_cp_sat",
            CpSatBackend(workers=cp_sat_workers),
            "exact",
            "OR-Tools CP-SAT; proves optimality when it finishes.",
        ),
        MethodSpec(
            "exact_pyomo",
            PyomoBackend(solver_name=pyomo_solver),
            "exact",
            "Pyomo dispatched to an installed MILP solver.",
        ),
        MethodSpec(
            "greedy",
            GreedyBackend(),
            "baseline",
            "Deterministic greedy construction; the honest baseline.",
        ),
        MethodSpec(
            "local_search",
            LocalSearchBackend(restarts=local_search_restarts),
            "proposed",
            "Greedy start plus hill climbing; the method under test.",
        ),
    ]
    return {spec.name: spec for spec in specs}


METHOD_SPECS: dict[str, MethodSpec] = build_method_specs()


def available_methods(specs: dict[str, MethodSpec] | None = None) -> dict[str, tuple[bool, str]]:
    """`{method: (available, reason_if_not)}` for the current machine."""

    specs = specs or METHOD_SPECS
    return {name: spec.backend.availability() for name, spec in specs.items()}


def describe_backends(specs: dict[str, MethodSpec] | None = None) -> list[dict[str, Any]]:
    """Inspectable table for `solver_backend_report.*`."""

    specs = specs or METHOD_SPECS
    rows: list[dict[str, Any]] = []
    for name, spec in specs.items():
        available, reason = spec.backend.availability()
        rows.append(
            {
                "method": name,
                "backend": spec.backend.name,
                "kind": spec.backend.kind,
                "role": spec.role,
                "available": available,
                "unavailable_reason": reason,
                "description": spec.description,
            }
        )
    return rows


def solve_instance(
    instance: BinaryProgramInstance,
    method: str,
    time_limit: float = 10.0,
    seed: int = 0,
    known_optimum: float | None = None,
    known_optimum_source: str = "",
    tolerance: float = DEFAULT_TOLERANCE,
    specs: dict[str, MethodSpec] | None = None,
) -> SolveResult:
    """Solve one instance with one method and return a *verified* record.

    Every path through this function ends in `verify_solver_claim`. If a backend
    claims a solution that fails verification, the status is rewritten to
    `error` and the contradiction is recorded in `notes` -- the row is kept so
    the defect is visible, but `is_trustworthy` is False so no analysis will
    average it in.
    """

    specs = specs or METHOD_SPECS
    reference = known_optimum if known_optimum is not None else instance.known_optimum
    reference_source = known_optimum_source or instance.known_optimum_source
    result = SolveResult(
        instance_id=instance.instance_id,
        method=method,
        instance_family=instance.family,
        seed=seed,
        problem_size=instance.n_vars,
        n_vars=instance.n_vars,
        n_constraints=instance.n_constraints,
        objective_sense=instance.objective_sense,
        known_optimum=reference,
        known_optimum_source=reference_source,
    )

    spec = specs.get(method)
    if spec is None:
        result.solver_status = "error"
        result.verification_ok = False
        result.add_note(f"Unknown method `{method}`. Known: {list(specs)}")
        return result
    result.solver_backend = spec.backend.name

    available, reason = spec.backend.availability()
    if not available:
        result.solver_status = "not_run"
        result.add_note(reason)
        return result

    started = time.perf_counter()
    try:
        outcome = spec.backend.solve(instance, time_limit=time_limit, seed=seed)
    except Exception as exc:  # a backend crash is a finding, not a silent zero
        result.solver_status = "error"
        result.verification_ok = False
        result.runtime_seconds = time.perf_counter() - started
        result.add_note(f"{spec.backend.name} raised {type(exc).__name__}: {exc}")
        return result

    result.solver_status = outcome.status
    result.objective = outcome.objective
    result.dual_bound = outcome.dual_bound
    result.node_count = outcome.node_count
    result.runtime_seconds = outcome.runtime_seconds or (time.perf_counter() - started)
    result.hit_time_limit = outcome.hit_time_limit
    result.solution = list(outcome.solution)
    result.add_note(outcome.notes)

    report, claim_ok, problems = verify_solver_claim(
        instance,
        outcome.solution or None,
        claimed_status=outcome.status,
        claimed_objective=outcome.objective,
        tolerance=tolerance,
    )
    result.feasible = bool(report.feasible and outcome.solution)
    result.max_violation = report.max_violation
    result.n_violated_constraints = report.n_violated
    result.verification_ok = claim_ok

    # The objective that reaches the results table is always the recomputed one.
    if outcome.solution and report.recomputed_objective is not None:
        result.objective = report.recomputed_objective

    if not claim_ok:
        result.solver_status = "error"
        for problem in problems:
            result.add_note(f"VERIFICATION FAILURE: {problem}")

    if (
        result.feasible
        and reference is not None
        and instance.is_better(float(result.objective or 0.0), reference)
    ):
        result.add_note(
            "VERIFICATION FAILURE: a verified-feasible solution beat the recorded reference "
            f"optimum ({result.objective!r} vs {reference!r}); the reference is wrong."
        )
        result.verification_ok = False

    return result.with_derived_metrics()


def compute_ground_truth(
    instance: BinaryProgramInstance,
    time_limit: float = 30.0,
    brute_force_max_vars: int = 20,
    specs: dict[str, MethodSpec] | None = None,
) -> BinaryProgramInstance:
    """Attach a *proved* optimum to the instance, or leave it as None.

    Brute force is preferred because it depends on no third-party solver. CP-SAT
    is accepted only when it reports `optimal`. A time-limited or heuristic
    answer is never recorded as a known optimum -- an unproved number used as a
    reference would silently corrupt every gap in the results table.
    """

    specs = specs or METHOD_SPECS
    order = (
        ["brute_force", "exact_cp_sat", "exact_pyomo"]
        if instance.n_vars <= brute_force_max_vars
        else ["exact_cp_sat", "exact_pyomo"]
    )
    for method in order:
        spec = specs.get(method)
        if spec is None:
            continue
        available, _ = spec.backend.availability()
        if not available:
            continue
        candidate = solve_instance(
            instance, method, time_limit=time_limit, seed=0, specs=specs
        )
        if candidate.proved_optimal and candidate.objective is not None:
            instance.known_optimum = float(candidate.objective)
            instance.known_optimum_source = f"{method}_proved"
            return instance
    instance.known_optimum = None
    instance.known_optimum_source = "unproved"
    return instance
