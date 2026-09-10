"""An independent ground truth for linear bilevel programs.

This module exists to be *unlike* the thing it checks. The transformation chain
under study rewrites a bilevel program through KKT conditions and a Big-M
linearization; if the reference answer were computed the same way, the
verification would be circular and would confirm a sign error rather than catch
it.

So nothing here uses KKT, complementarity, duals, or Big-M. The optimum of a
linear bilevel program with a compact joint region is attained at a vertex of
that region, so the oracle

1. enumerates the vertices of `{(x, y) : all upper and lower rows hold}` by
   solving every d-subset of the rows as equalities, and
2. for each vertex, solves the follower's LP directly at that `x` and keeps the
   vertex only if its `y` attains the follower's optimal value.

Step 2 is the only definition of bilevel feasibility used, and it is the
textbook one. Vertices whose `y` ties the follower optimum are all admissible,
which makes this the *optimistic* formulation -- the same one the KKT
reformulation implements, so the comparison is fair.

A second, cheaper method (`solve_bilevel_by_sampling`) samples the leader's
box and solves the follower exactly at each sample. Every point it returns is
genuinely bilevel-feasible, so its value bounds the optimum and disagreements
with the vertex oracle are detectable.
"""

from __future__ import annotations

import itertools
import math
import time
from typing import Literal, Sequence

import numpy as np
from pydantic import BaseModel, Field

from src.core.bilevel import DEFAULT_TOLERANCE, BilevelLinearProgram


OracleStatus = Literal["optimal", "infeasible", "no_solution", "not_run", "error"]


class LowerLevelSolution(BaseModel):
    status: Literal["optimal", "infeasible", "unbounded", "error"]
    value: float | None = None
    y: list[float] = Field(default_factory=list)
    notes: str = ""


class BilevelVerification(BaseModel):
    """Independent check that a claimed (x, y) really solves the bilevel program."""

    jointly_feasible: bool
    lower_level_optimal: bool
    max_joint_violation: float = 0.0
    broken_rows: list[str] = Field(default_factory=list)
    follower_optimal_value: float | None = None
    follower_value_at_y: float | None = None
    follower_suboptimality: float | None = None
    upper_objective: float | None = None
    notes: str = ""

    @property
    def is_bilevel_feasible(self) -> bool:
        return self.jointly_feasible and self.lower_level_optimal


class OracleResult(BaseModel):
    status: OracleStatus = "not_run"
    method: str = "vertex_enumeration"
    upper_objective: float | None = None
    x: list[float] = Field(default_factory=list)
    y: list[float] = Field(default_factory=list)
    vertices_examined: int = 0
    vertices_total: int = 0
    vertices_feasible: int = 0
    alternative_optima: int = 0
    runtime_seconds: float = 0.0
    notes: str = ""

    @property
    def has_solution(self) -> bool:
        return self.status == "optimal" and self.upper_objective is not None


def solve_lower_level(
    blp: BilevelLinearProgram,
    x: Sequence[float],
    tolerance: float = DEFAULT_TOLERANCE,
) -> LowerLevelSolution:
    """Solve the follower's LP exactly at a fixed `x`."""

    try:
        from scipy.optimize import linprog
    except Exception as exc:  # pragma: no cover - environment dependent
        return LowerLevelSolution(status="error", notes=f"scipy missing: {exc}")

    E, G, h = blp.lower_system()
    rhs = h - E @ np.asarray(x, dtype=float)
    result = linprog(
        np.asarray(blp.d2, dtype=float),
        A_ub=G,
        b_ub=rhs,
        bounds=[(0.0, float(u)) for u in blp.y_upper],
        method="highs",
    )
    if result.status == 0:
        return LowerLevelSolution(
            status="optimal", value=float(result.fun), y=[float(v) for v in result.x]
        )
    mapping = {2: "infeasible", 3: "unbounded"}
    return LowerLevelSolution(
        status=mapping.get(int(result.status), "error"),  # type: ignore[arg-type]
        notes=str(getattr(result, "message", "")).strip(),
    )


def verify_bilevel_solution(
    blp: BilevelLinearProgram,
    x: Sequence[float],
    y: Sequence[float],
    tolerance: float = 1e-6,
) -> BilevelVerification:
    """Check a claimed solution against the definition, not against a solver."""

    if len(x) != blp.n_x or len(y) != blp.n_y:
        return BilevelVerification(
            jointly_feasible=False,
            lower_level_optimal=False,
            notes=f"expected {blp.n_x} leader and {blp.n_y} follower variables",
        )
    violation, broken = blp.joint_violation(x, y)
    follower = solve_lower_level(blp, x, tolerance=tolerance)
    value_at_y = blp.lower_objective(y)
    if follower.status != "optimal" or follower.value is None:
        return BilevelVerification(
            jointly_feasible=violation <= tolerance,
            lower_level_optimal=False,
            max_joint_violation=violation,
            broken_rows=broken,
            follower_value_at_y=value_at_y,
            upper_objective=blp.upper_objective(x, y),
            notes=f"the follower's problem is `{follower.status}` at this x",
        )
    scale = max(1.0, abs(follower.value))
    suboptimality = value_at_y - follower.value
    return BilevelVerification(
        jointly_feasible=violation <= tolerance,
        lower_level_optimal=suboptimality <= tolerance * scale,
        max_joint_violation=violation,
        broken_rows=broken,
        follower_optimal_value=follower.value,
        follower_value_at_y=value_at_y,
        follower_suboptimality=suboptimality,
        upper_objective=blp.upper_objective(x, y),
    )


def enumerate_vertices(
    blp: BilevelLinearProgram,
    tolerance: float = 1e-7,
    max_combinations: int = 400_000,
) -> tuple[list[np.ndarray], int, str]:
    """Every vertex of the joint region, by solving all d-subsets as equalities."""

    P, q, _ = blp.joint_system()
    n_rows, dimension = P.shape
    total = math.comb(n_rows, dimension)
    if total > max_combinations:
        return [], 0, (
            f"vertex enumeration skipped: C({n_rows}, {dimension}) = {total} exceeds "
            f"max_combinations={max_combinations}"
        )
    vertices: list[np.ndarray] = []
    seen: set[tuple[float, ...]] = set()
    examined = 0
    for rows in itertools.combinations(range(n_rows), dimension):
        examined += 1
        submatrix = P[list(rows)]
        if abs(np.linalg.det(submatrix)) < 1e-9:
            continue
        try:
            point = np.linalg.solve(submatrix, q[list(rows)])
        except np.linalg.LinAlgError:  # pragma: no cover - covered by the det guard
            continue
        if not np.all(P @ point - q <= tolerance):
            continue
        key = tuple(np.round(point, 7) + 0.0)
        if key in seen:
            continue
        seen.add(key)
        vertices.append(point)
    return vertices, examined, ""


def solve_bilevel_by_vertex_enumeration(
    blp: BilevelLinearProgram,
    tolerance: float = 1e-6,
    max_combinations: int = 400_000,
) -> OracleResult:
    """Exact optimistic bilevel optimum, with no KKT and no Big-M anywhere."""

    started = time.perf_counter()
    vertices, examined, skip_note = enumerate_vertices(
        blp, max_combinations=max_combinations
    )
    if skip_note:
        return OracleResult(
            status="not_run", runtime_seconds=time.perf_counter() - started, notes=skip_note
        )

    best_value: float | None = None
    best_x: list[float] = []
    best_y: list[float] = []
    feasible_count = 0
    ties = 0
    for point in vertices:
        x = point[: blp.n_x]
        y = point[blp.n_x :]
        check = verify_bilevel_solution(blp, x, y, tolerance=tolerance)
        if not check.is_bilevel_feasible:
            continue
        feasible_count += 1
        value = blp.upper_objective(x, y)
        if best_value is not None and abs(value - best_value) <= tolerance * max(1.0, abs(value)):
            ties += 1
            continue
        if blp.is_better_upper(value, best_value):
            best_value = value
            best_x = [float(v) for v in x]
            best_y = [float(v) for v in y]
            ties = 0

    runtime = time.perf_counter() - started
    if best_value is None:
        return OracleResult(
            status="infeasible",
            vertices_examined=examined,
            vertices_total=len(vertices),
            vertices_feasible=0,
            runtime_seconds=runtime,
            notes=(
                f"examined {len(vertices)} vertices from {examined} row subsets; none was "
                "bilevel-feasible"
            ),
        )
    return OracleResult(
        status="optimal",
        upper_objective=best_value,
        x=best_x,
        y=best_y,
        vertices_examined=examined,
        vertices_total=len(vertices),
        vertices_feasible=feasible_count,
        alternative_optima=ties,
        runtime_seconds=runtime,
        notes=(
            f"{len(vertices)} vertices from {examined} row subsets, {feasible_count} "
            "bilevel-feasible; optimum proved by enumeration"
        ),
    )


def solve_bilevel_by_sampling(
    blp: BilevelLinearProgram,
    samples: int = 2000,
    seed: int = 0,
    include_grid: bool = True,
    tolerance: float = 1e-6,
) -> OracleResult:
    """Sample the leader's box, solve the follower exactly, keep the best point.

    Every returned point is genuinely bilevel-feasible, so for a minimizing
    leader this value is an upper bound on the true optimum. It is a cross-check
    on the vertex oracle, not a replacement: sampling cannot prove optimality,
    and where the follower has ties it may return a `y` the leader dislikes.
    """

    started = time.perf_counter()
    rng = np.random.default_rng(seed)
    candidates: list[np.ndarray] = []
    upper = np.asarray(blp.x_upper, dtype=float)
    if include_grid:
        per_axis = max(2, int(round(samples ** (1.0 / max(1, blp.n_x)))))
        per_axis = min(per_axis, 21)
        axes = [np.linspace(0.0, upper[j], per_axis) for j in range(blp.n_x)]
        grid = np.array(list(itertools.product(*axes))) if blp.n_x else np.zeros((1, 0))
        candidates.extend(grid)
    candidates.extend(rng.uniform(0.0, upper, size=(samples, blp.n_x)))

    best_value: float | None = None
    best_x: list[float] = []
    best_y: list[float] = []
    feasible = 0
    for x in candidates:
        follower = solve_lower_level(blp, x, tolerance=tolerance)
        if follower.status != "optimal":
            continue
        y = np.asarray(follower.y, dtype=float)
        if not blp.is_jointly_feasible(x, y, tolerance=tolerance):
            continue
        feasible += 1
        value = blp.upper_objective(x, y)
        if blp.is_better_upper(value, best_value):
            best_value = value
            best_x = [float(v) for v in x]
            best_y = [float(v) for v in y]

    runtime = time.perf_counter() - started
    if best_value is None:
        return OracleResult(
            status="no_solution",
            method="sampling",
            vertices_examined=len(candidates),
            runtime_seconds=runtime,
            notes=f"no bilevel-feasible point found among {len(candidates)} sampled x values",
        )
    return OracleResult(
        status="optimal",
        method="sampling",
        upper_objective=best_value,
        x=best_x,
        y=best_y,
        vertices_examined=len(candidates),
        vertices_feasible=feasible,
        runtime_seconds=runtime,
        notes=(
            f"best of {feasible} bilevel-feasible points from {len(candidates)} sampled x values; "
            "this bounds the optimum but does not prove it"
        ),
    )
