"""The transformation chain: bilevel -> KKT one-level -> MI 0-1.

Two rewrites, each of which can be silently wrong:

**Step 1 -- bilevel to a single level via the follower's KKT conditions.**
The follower solves an LP, so its KKT conditions are necessary *and*
sufficient, and replacing `y in argmin{...}` by them is exact. What is easy to
get wrong is the algebra: which rows carry multipliers, the sign of the
stationarity equation, and whether the folded bound rows were handled. A sign
error here produces a model that still solves and still returns a number.

**Step 2 -- complementarity to binaries via Big-M (Fortuny-Amat).**
`lambda_i * slack_i = 0` becomes `slack_i <= M_s (1 - z_i)` and
`lambda_i <= M_l z_i`. This is exact *only if* both constants are true upper
bounds. Choose them too small and the model quietly removes the optimum;
"just use 1e6" trades that for a relaxation so weak the solver crawls.

So `derive_big_m_bounds` does not guess. Slack bounds come from interval
arithmetic over the box. Multiplier bounds need more care: folding the `y`
bounds into `G` makes the dual feasible set `D = {lambda >= 0 : G'lambda = -d2}`
unbounded in every coordinate (add the same amount to the multipliers of
`y_j <= u_j` and `-y_j <= 0` and nothing breaks), so simply maximising
`lambda_i` over `D` returns infinity and is useless. What *is* finite is the set
of **vertices** of `D`, and an LP whose feasible region is bounded always admits
an optimal dual that is one of them. Bounding `lambda_i` by its largest value
over those vertices is therefore both valid -- no bilevel-feasible `(x, y)`
loses its certificate -- and tight enough to be usable.

Variable ordering is fixed everywhere: `[x, y, lambda]`, plus `z` appended for
the MI 0-1 model.
"""

from __future__ import annotations

import itertools
import time
from typing import Any, Literal, Sequence

import numpy as np
from pydantic import BaseModel, Field

from src.core.bilevel import BilevelLinearProgram
from src.core.mixed_integer_program import (
    MipConstraint,
    MipSolution,
    MipVariable,
    MixedIntegerProgram,
    solve_lp,
)


DEFAULT_TOLERANCE = 1e-6


class ComplementarityPair(BaseModel):
    """`lambda_i >= 0`, `slack_i >= 0`, `lambda_i * slack_i = 0`."""

    name: str
    row_index: int
    lambda_index: int


class BigMBounds(BaseModel):
    """Upper bounds used to linearise complementarity, with their provenance."""

    slack_bounds: list[float] = Field(default_factory=list)
    dual_bounds: list[float] = Field(default_factory=list)
    dual_bounded: list[bool] = Field(default_factory=list)
    dual_set_unbounded: list[bool] = Field(default_factory=list)
    dual_vertices_found: int = 0
    slack_method: str = "interval_arithmetic_over_box"
    dual_method: str = "vertex_enumeration_of_the_dual_feasible_set"
    notes: list[str] = Field(default_factory=list)

    @property
    def all_duals_bounded(self) -> bool:
        return all(self.dual_bounded) if self.dual_bounded else False

    @property
    def is_rigorous(self) -> bool:
        """True when every constant is justified, not guessed."""

        return bool(self.slack_bounds) and self.all_duals_bounded

    @property
    def max_slack(self) -> float:
        return max(self.slack_bounds, default=0.0)

    @property
    def max_dual(self) -> float:
        return max((b for b, ok in zip(self.dual_bounds, self.dual_bounded) if ok), default=0.0)


class KKTOneLevelProgram(BaseModel):
    """The one-level program with complementarity constraints (still nonlinear)."""

    source_instance_id: str
    n_x: int
    n_y: int
    n_lambda: int
    upper_sense: Literal["minimize", "maximize"]
    objective: list[float]                       # over [x, y, lambda]
    linear_constraints: list[MipConstraint] = Field(default_factory=list)
    complementarity: list[ComplementarityPair] = Field(default_factory=list)
    variable_names: list[str] = Field(default_factory=list)
    lower_bounds: list[float] = Field(default_factory=list)
    upper_bounds: list[float] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def n_vars(self) -> int:
        return self.n_x + self.n_y + self.n_lambda

    def split(self, values: Sequence[float]) -> tuple[list[float], list[float], list[float]]:
        x = [float(v) for v in values[: self.n_x]]
        y = [float(v) for v in values[self.n_x : self.n_x + self.n_y]]
        lam = [float(v) for v in values[self.n_x + self.n_y :]]
        return x, y, lam

    def as_linear_program(self) -> MixedIntegerProgram:
        """The relaxation with complementarity *dropped* (not a valid reformulation)."""

        return MixedIntegerProgram(
            name=f"{self.source_instance_id}_kkt_relaxation",
            sense=self.upper_sense,
            variables=[
                MipVariable(name=name, lower=lo, upper=None if np.isinf(hi) else hi)
                for name, lo, hi in zip(self.variable_names, self.lower_bounds, self.upper_bounds)
            ],
            objective=list(self.objective),
            constraints=list(self.linear_constraints),
            metadata={"note": "complementarity constraints are NOT included"},
        )


# ---------------------------------------------------------------------------
# Step 1: bilevel -> KKT one-level
# ---------------------------------------------------------------------------


def to_kkt_one_level(blp: BilevelLinearProgram) -> KKTOneLevelProgram:
    """Replace `y in argmin{...}` by the follower's KKT system.

    Lower level: `min d2'y s.t. G y <= h - E x`, Lagrangian
    `L = d2'y + lambda'(G y - h + E x)` with `lambda >= 0`, so

    * stationarity          `d2 + G' lambda = 0`
    * primal feasibility    `E x + G y <= h`
    * dual feasibility      `lambda >= 0`
    * complementarity       `lambda_i (h_i - E_i x - G_i y) = 0`

    Exact because the follower's problem is a linear program.
    """

    E, G, h = blp.lower_system()
    m_g = G.shape[0]
    n_x, n_y = blp.n_x, blp.n_y
    total = n_x + n_y + m_g

    names = (
        [f"x[{j}]" for j in range(n_x)]
        + [f"y[{j}]" for j in range(n_y)]
        + [f"lambda[{i}]" for i in range(m_g)]
    )
    lower_bounds = [0.0] * total
    upper_bounds = (
        [float(v) for v in blp.x_upper]
        + [float(v) for v in blp.y_upper]
        + [float("inf")] * m_g
    )

    objective = [0.0] * total
    for j in range(n_x):
        objective[j] = float(blp.c1[j])
    for j in range(n_y):
        objective[n_x + j] = float(blp.d1[j])

    constraints: list[MipConstraint] = []

    # Leader's own constraints.
    for i in range(blp.n_upper_rows):
        row = [0.0] * total
        for j in range(n_x):
            row[j] = float(blp.A1[i][j])
        for j in range(n_y):
            row[n_x + j] = float(blp.B1[i][j])
        constraints.append(
            MipConstraint(name=f"upper_{i}", coefficients=row, sense="<=", rhs=float(blp.b1[i]))
        )

    # Follower primal feasibility: E x + G y <= h.
    row_names = blp.lower_row_names()
    for i in range(m_g):
        row = [0.0] * total
        for j in range(n_x):
            row[j] = float(E[i, j])
        for j in range(n_y):
            row[n_x + j] = float(G[i, j])
        constraints.append(
            MipConstraint(
                name=f"primal_{row_names[i]}", coefficients=row, sense="<=", rhs=float(h[i])
            )
        )

    # Stationarity: for each follower variable j, sum_i G[i, j] lambda_i = -d2[j].
    for j in range(n_y):
        row = [0.0] * total
        for i in range(m_g):
            row[n_x + n_y + i] = float(G[i, j])
        constraints.append(
            MipConstraint(
                name=f"stationarity_y{j}",
                coefficients=row,
                sense="==",
                rhs=-float(blp.d2[j]),
            )
        )

    complementarity = [
        ComplementarityPair(
            name=f"comp_{row_names[i]}", row_index=i, lambda_index=n_x + n_y + i
        )
        for i in range(m_g)
    ]

    return KKTOneLevelProgram(
        source_instance_id=blp.instance_id,
        n_x=n_x,
        n_y=n_y,
        n_lambda=m_g,
        upper_sense=blp.upper_sense,
        objective=objective,
        linear_constraints=constraints,
        complementarity=complementarity,
        variable_names=names,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        metadata={
            "n_upper_rows": blp.n_upper_rows,
            "n_lower_rows": m_g,
            "derivation": "follower KKT (necessary and sufficient for an LP follower)",
        },
    )


# ---------------------------------------------------------------------------
# Big-M derivation
# ---------------------------------------------------------------------------


def _dual_vertices(G: np.ndarray, d2: np.ndarray) -> list[np.ndarray]:
    """Basic feasible solutions of `D = {lambda >= 0 : G'lambda = -d2}`.

    A basic solution picks `n_y` columns, solves the resulting square system and
    zeroes the rest. `D` may be unbounded, but it has finitely many vertices and
    an LP with a bounded feasible region always has an optimal dual among them.
    """

    m_g, n_y = G.shape
    if n_y == 0:
        return [np.zeros(m_g)]
    vertices: list[np.ndarray] = []
    seen: set[tuple[float, ...]] = set()
    for columns in itertools.combinations(range(m_g), n_y):
        submatrix = G[list(columns), :].T  # (n_y, n_y)
        if abs(np.linalg.det(submatrix)) < 1e-9:
            continue
        try:
            basic = np.linalg.solve(submatrix, -d2)
        except np.linalg.LinAlgError:  # pragma: no cover - covered by the det guard
            continue
        if np.any(basic < -1e-9):
            continue
        candidate = np.zeros(m_g)
        candidate[list(columns)] = np.maximum(basic, 0.0)
        key = tuple(np.round(candidate, 7) + 0.0)
        if key in seen:
            continue
        seen.add(key)
        vertices.append(candidate)
    return vertices


def derive_big_m_bounds(
    blp: BilevelLinearProgram, safety_factor: float = 1.0
) -> BigMBounds:
    """Valid upper bounds for the slacks and the multipliers.

    Slacks: interval arithmetic over the box gives
    `max slack_i = h_i - min_{x,y}(E_i x + G_i y)`, exact and cheap.

    Multipliers: bounded by their largest value over the vertices of the dual
    feasible set. An LP probe is also run per coordinate purely as a diagnostic,
    to record that the naive "maximise over `D`" bound would have returned
    infinity -- which is why that method is not the one used.
    """

    E, G, h = blp.lower_system()
    m_g = G.shape[0]
    d2 = np.asarray(blp.d2, dtype=float)
    x_ub = np.asarray(blp.x_upper, dtype=float)
    y_ub = np.asarray(blp.y_upper, dtype=float)

    # min over the box of E_i x + G_i y: negative coefficients hit their upper bound.
    min_ex = np.minimum(E, 0.0) @ x_ub
    min_gy = np.minimum(G, 0.0) @ y_ub
    slack_bounds = [
        float(max(0.0, h[i] - min_ex[i] - min_gy[i]) * safety_factor) for i in range(m_g)
    ]

    notes: list[str] = []
    vertices = _dual_vertices(G, d2)
    if vertices:
        stacked = np.vstack(vertices)
        dual_bounds = [float(stacked[:, i].max() * safety_factor) for i in range(m_g)]
        dual_bounded = [True] * m_g
        # A multiplier that is zero at every vertex still needs a positive
        # switch constant, or `lambda_i <= M z_i` would pin it to zero outright.
        floor = max(1.0, max(dual_bounds) * 1e-6)
        dual_bounds = [max(bound, floor) for bound in dual_bounds]
        notes.append(
            f"multiplier bounds taken over {len(vertices)} vertices of the dual feasible set"
        )
    else:
        dual_bounds = [float("inf")] * m_g
        dual_bounded = [False] * m_g
        notes.append(
            "the dual feasible set has no vertex, so no finite multiplier bound is justified; "
            "the follower's KKT system may be infeasible."
        )

    # Diagnostic only: show that the naive LP bound is unusable here.
    dual_set_unbounded: list[bool] = []
    dual_program = MixedIntegerProgram(
        name=f"{blp.instance_id}_dual_feasible_set",
        sense="maximize",
        variables=[MipVariable(name=f"lambda[{i}]", lower=0.0, upper=None) for i in range(m_g)],
        objective=[0.0] * m_g,
        constraints=[
            MipConstraint(
                name=f"stationarity_y{j}",
                coefficients=[float(G[i, j]) for i in range(m_g)],
                sense="==",
                rhs=-float(d2[j]),
            )
            for j in range(blp.n_y)
        ],
    )
    for i in range(m_g):
        probe = dual_program.model_copy(deep=True)
        probe.objective = [1.0 if k == i else 0.0 for k in range(m_g)]
        dual_set_unbounded.append(solve_lp(probe).status == "unbounded")
    if any(dual_set_unbounded):
        notes.append(
            f"{sum(dual_set_unbounded)} of {m_g} multipliers are unbounded over the dual feasible "
            "set itself, so maximising over that set would have given no usable constant."
        )

    return BigMBounds(
        slack_bounds=slack_bounds,
        dual_bounds=dual_bounds,
        dual_bounded=dual_bounded,
        dual_set_unbounded=dual_set_unbounded,
        dual_vertices_found=len(vertices),
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Step 2: KKT -> MI 0-1
# ---------------------------------------------------------------------------


def to_mi01(
    kkt: KKTOneLevelProgram,
    slack_big_m: float | Sequence[float],
    dual_big_m: float | Sequence[float],
) -> MixedIntegerProgram:
    """Fortuny-Amat linearisation: one binary per complementarity pair.

    `z_i = 1` means "row i is active": the slack is forced to zero and the
    multiplier may be positive. `z_i = 0` forces the multiplier to zero and lets
    the slack grow. Exact when both constants are genuine upper bounds.
    """

    m_g = kkt.n_lambda
    base = kkt.n_vars
    total = base + m_g
    slack_m = (
        [float(slack_big_m)] * m_g if np.isscalar(slack_big_m) else [float(v) for v in slack_big_m]
    )
    dual_m = (
        [float(dual_big_m)] * m_g if np.isscalar(dual_big_m) else [float(v) for v in dual_big_m]
    )
    if len(slack_m) != m_g or len(dual_m) != m_g:
        raise ValueError("Big-M vectors must have one entry per complementarity pair")
    if not all(np.isfinite(slack_m)) or not all(np.isfinite(dual_m)):
        raise ValueError(
            "Big-M constants must be finite. An unbounded multiplier means no valid "
            "linearisation exists by this argument; substituting a huge number produces a "
            "numerically meaningless model rather than a reformulation."
        )

    def widen(coefficients: Sequence[float]) -> list[float]:
        return list(coefficients) + [0.0] * m_g

    variables = [
        MipVariable(
            name=name,
            lower=lo,
            upper=None if np.isinf(hi) else hi,
        )
        for name, lo, hi in zip(kkt.variable_names, kkt.lower_bounds, kkt.upper_bounds)
    ]
    for i in range(m_g):
        if np.isfinite(dual_m[i]):
            variables[kkt.complementarity[i].lambda_index].upper = dual_m[i]
    variables.extend(MipVariable(name=f"z[{i}]", is_binary=True) for i in range(m_g))

    constraints = [
        MipConstraint(
            name=c.name, coefficients=widen(c.coefficients), sense=c.sense, rhs=c.rhs
        )
        for c in kkt.linear_constraints
    ]

    primal_rows = {c.name: c for c in kkt.linear_constraints if c.name.startswith("primal_")}
    primal_list = list(primal_rows.values())
    for i, pair in enumerate(kkt.complementarity):
        primal = primal_list[i]
        # slack_i = rhs - (row . v) <= M_s (1 - z_i)   ->   -(row . v) + M_s z_i <= M_s - rhs
        row = [-float(a) for a in primal.coefficients] + [0.0] * m_g
        row[base + i] = slack_m[i]
        constraints.append(
            MipConstraint(
                name=f"bigM_slack_{pair.name}",
                coefficients=row,
                sense="<=",
                rhs=float(slack_m[i]) - float(primal.rhs),
            )
        )
        # lambda_i <= M_l z_i   ->   lambda_i - M_l z_i <= 0
        row = [0.0] * total
        row[pair.lambda_index] = 1.0
        row[base + i] = -float(dual_m[i])
        constraints.append(
            MipConstraint(name=f"bigM_dual_{pair.name}", coefficients=row, sense="<=", rhs=0.0)
        )

    return MixedIntegerProgram(
        name=f"{kkt.source_instance_id}_mi01",
        sense=kkt.upper_sense,
        variables=variables,
        objective=widen(kkt.objective),
        constraints=constraints,
        metadata={
            "n_x": kkt.n_x,
            "n_y": kkt.n_y,
            "n_lambda": m_g,
            "n_binaries": m_g,
            "slack_big_m": slack_m,
            "dual_big_m": dual_m,
            "linearisation": "Fortuny-Amat (z=1 means the row is active)",
        },
    )


# ---------------------------------------------------------------------------
# Solving the KKT program without any Big-M
# ---------------------------------------------------------------------------


def solve_kkt_by_pattern_enumeration(
    kkt: KKTOneLevelProgram,
    max_patterns: int = 1 << 16,
    time_limit: float = 120.0,
) -> MipSolution:
    """Exact optimum of the complementarity program, with no Big-M at all.

    For every subset of rows declared active, the complementarity constraints
    become linear (`slack_i = 0` inside the subset, `lambda_i = 0` outside), so
    each pattern is one LP. Enumerating all of them solves the program exactly
    and depends on no constant, which is what makes it a fair reference for
    judging a Big-M model.
    """

    m_g = kkt.n_lambda
    if (1 << m_g) > max_patterns:
        return MipSolution(
            status="not_run",
            backend="kkt_pattern_enumeration",
            notes=(
                f"skipped: 2^{m_g} complementarity patterns exceeds max_patterns={max_patterns}"
            ),
        )

    started = time.perf_counter()
    relaxation = kkt.as_linear_program()
    primal_rows = [c for c in kkt.linear_constraints if c.name.startswith("primal_")]
    best: MipSolution | None = None
    solved = 0
    hit_limit = False

    for pattern in itertools.product((0, 1), repeat=m_g):
        if time_limit and (time.perf_counter() - started) > time_limit:
            hit_limit = True
            break
        program = relaxation.model_copy(deep=True)
        fixed: dict[int, float] = {}
        extra: list[MipConstraint] = []
        for i, active in enumerate(pattern):
            pair = kkt.complementarity[i]
            if active:
                row = primal_rows[i]
                extra.append(
                    MipConstraint(
                        name=f"active_{pair.name}",
                        coefficients=list(row.coefficients),
                        sense="==",
                        rhs=float(row.rhs),
                    )
                )
            else:
                fixed[pair.lambda_index] = 0.0
        program.constraints = list(program.constraints) + extra
        branch = solve_lp(program, fixed=fixed)
        solved += 1
        if branch.status != "optimal" or branch.objective is None:
            continue
        if best is None or program.is_better(branch.objective, best.objective):
            best = branch

    runtime = time.perf_counter() - started
    if best is None:
        return MipSolution(
            status="no_solution" if hit_limit else "infeasible",
            backend="kkt_pattern_enumeration",
            runtime_seconds=runtime,
            lp_solves=solved,
            notes=(
                "stopped on the time limit before any pattern was feasible"
                if hit_limit
                else f"all {solved} complementarity patterns were infeasible"
            ),
        )
    return MipSolution(
        status="feasible" if hit_limit else "optimal",
        objective=best.objective,
        values=list(best.values),
        backend="kkt_pattern_enumeration",
        runtime_seconds=runtime,
        lp_solves=solved,
        notes=(
            "stopped on the time limit; optimality is NOT proved"
            if hit_limit
            else f"enumerated all {solved} complementarity patterns; no Big-M was used"
        ),
    )
