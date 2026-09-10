"""Bilevel linear programs: the object the transformation chain starts from.

Canonical form used throughout this project:

    upper:   min / max over x   c1'x + d1'y
             subject to         A1 x + B1 y <= b1
                                0 <= x <= x_upper
                                y in S(x)

    lower:   S(x) = argmin over y { d2'y : A2 x + B2 y <= b2, 0 <= y <= y_upper }

The lower level is always a minimization; a maximizing follower is expressed by
negating `d2`.

One representational choice matters for everything downstream. The bounds
`0 <= y <= y_upper` are *folded into the lower constraint matrix*, giving a
single system

    G y <= h - E x        with   G = [B2; I; -I],  h = [b2; y_upper; 0],  E = [A2; 0; 0]

so the KKT conditions have exactly one family of multipliers and one family of
complementarity pairs instead of three. This is an exact rewrite, not an
approximation: the multipliers of the folded rows are precisely the duals of
the bound constraints.

Box bounds on both x and y are mandatory. They make the joint region compact,
which is what lets the oracle enumerate vertices and lets `derive_big_m_bounds`
produce a *valid* slack bound instead of a guess.
"""

from __future__ import annotations

import zlib
from typing import Any, Literal, Sequence

import numpy as np
from pydantic import BaseModel, Field, model_validator


ObjectiveSense = Literal["minimize", "maximize"]

Difficulty = Literal["balanced", "coupled", "degenerate", "large_dual"]

DEFAULT_TOLERANCE = 1e-6

BILEVEL_DIFFICULTIES: tuple[str, ...] = ("balanced", "coupled", "degenerate", "large_dual")


def _code(text: str) -> int:
    return int(zlib.crc32(text.encode("utf-8")) & 0x7FFFFFFF)


def _rng(size: int, seed: int, difficulty: str) -> np.random.Generator:
    return np.random.default_rng(
        np.random.SeedSequence([int(seed), int(size), _code("bilevel"), _code(difficulty)])
    )


class BilevelLinearProgram(BaseModel):
    """A linear bilevel program with compact box bounds on both levels."""

    instance_id: str
    n_x: int
    n_y: int
    upper_sense: ObjectiveSense = "minimize"

    c1: list[float]                      # upper objective, x part
    d1: list[float]                      # upper objective, y part
    A1: list[list[float]] = Field(default_factory=list)   # upper rows, x part
    B1: list[list[float]] = Field(default_factory=list)   # upper rows, y part
    b1: list[float] = Field(default_factory=list)

    d2: list[float]                      # lower objective (minimized)
    A2: list[list[float]] = Field(default_factory=list)   # lower rows, x part
    B2: list[list[float]] = Field(default_factory=list)   # lower rows, y part
    b2: list[float] = Field(default_factory=list)

    x_upper: list[float]
    y_upper: list[float]

    seed: int = 0
    difficulty: str = "balanced"
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_shapes(self) -> "BilevelLinearProgram":
        if len(self.c1) != self.n_x or len(self.x_upper) != self.n_x:
            raise ValueError("c1 and x_upper must have length n_x")
        if len(self.d1) != self.n_y or len(self.d2) != self.n_y or len(self.y_upper) != self.n_y:
            raise ValueError("d1, d2 and y_upper must have length n_y")
        if not (len(self.A1) == len(self.B1) == len(self.b1)):
            raise ValueError("A1, B1 and b1 must have the same number of rows")
        if not (len(self.A2) == len(self.B2) == len(self.b2)):
            raise ValueError("A2, B2 and b2 must have the same number of rows")
        for row in self.A1:
            if len(row) != self.n_x:
                raise ValueError("every A1 row must have n_x entries")
        for row in self.B1:
            if len(row) != self.n_y:
                raise ValueError("every B1 row must have n_y entries")
        for row in self.A2:
            if len(row) != self.n_x:
                raise ValueError("every A2 row must have n_x entries")
        for row in self.B2:
            if len(row) != self.n_y:
                raise ValueError("every B2 row must have n_y entries")
        return self

    # -- shapes --------------------------------------------------------------

    @property
    def n_upper_rows(self) -> int:
        return len(self.b1)

    @property
    def n_lower_rows(self) -> int:
        """Rows of the folded lower system: original rows plus both y bounds."""

        return len(self.b2) + 2 * self.n_y

    @property
    def n_vars(self) -> int:
        return self.n_x + self.n_y

    # -- systems -------------------------------------------------------------

    def lower_system(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """`G y <= h - E x`, with the y box folded in. Returns (E, G, h)."""

        identity = np.eye(self.n_y)
        b2 = np.asarray(self.B2, dtype=float).reshape(len(self.b2), self.n_y)
        a2 = np.asarray(self.A2, dtype=float).reshape(len(self.b2), self.n_x)
        G = np.vstack([b2, identity, -identity])
        E = np.vstack([a2, np.zeros((self.n_y, self.n_x)), np.zeros((self.n_y, self.n_x))])
        h = np.concatenate(
            [np.asarray(self.b2, dtype=float), np.asarray(self.y_upper, dtype=float), np.zeros(self.n_y)]
        )
        return E, G, h

    def lower_row_names(self) -> list[str]:
        return (
            [f"lower_{i}" for i in range(len(self.b2))]
            + [f"y_ub_{j}" for j in range(self.n_y)]
            + [f"y_lb_{j}" for j in range(self.n_y)]
        )

    def joint_system(self) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Every constraint in (x, y) space: `P [x; y] <= q`. Returns (P, q, names)."""

        E, G, h = self.lower_system()
        blocks, rhs, names = [], [], []
        if self.n_upper_rows:
            a1 = np.asarray(self.A1, dtype=float).reshape(self.n_upper_rows, self.n_x)
            b1_block = np.asarray(self.B1, dtype=float).reshape(self.n_upper_rows, self.n_y)
            blocks.append(np.hstack([a1, b1_block]))
            rhs.append(np.asarray(self.b1, dtype=float))
            names.extend(f"upper_{i}" for i in range(self.n_upper_rows))
        blocks.append(np.hstack([E, G]))
        rhs.append(h)
        names.extend(self.lower_row_names())
        identity_x = np.eye(self.n_x)
        zeros_xy = np.zeros((self.n_x, self.n_y))
        blocks.append(np.hstack([identity_x, zeros_xy]))
        rhs.append(np.asarray(self.x_upper, dtype=float))
        names.extend(f"x_ub_{j}" for j in range(self.n_x))
        blocks.append(np.hstack([-identity_x, zeros_xy]))
        rhs.append(np.zeros(self.n_x))
        names.extend(f"x_lb_{j}" for j in range(self.n_x))
        return np.vstack(blocks), np.concatenate(rhs), names

    # -- evaluation ----------------------------------------------------------

    def upper_objective(self, x: Sequence[float], y: Sequence[float]) -> float:
        return float(
            np.dot(np.asarray(self.c1, dtype=float), np.asarray(x, dtype=float))
            + np.dot(np.asarray(self.d1, dtype=float), np.asarray(y, dtype=float))
        )

    def lower_objective(self, y: Sequence[float]) -> float:
        return float(np.dot(np.asarray(self.d2, dtype=float), np.asarray(y, dtype=float)))

    def is_better_upper(self, candidate: float, incumbent: float | None) -> bool:
        if incumbent is None:
            return True
        return candidate > incumbent if self.upper_sense == "maximize" else candidate < incumbent

    def joint_violation(
        self, x: Sequence[float], y: Sequence[float]
    ) -> tuple[float, list[str]]:
        """Largest joint-constraint violation and the names of the broken rows."""

        P, q, names = self.joint_system()
        point = np.concatenate([np.asarray(x, dtype=float), np.asarray(y, dtype=float)])
        residual = P @ point - q
        broken = [names[i] for i in range(len(q)) if residual[i] > DEFAULT_TOLERANCE]
        return float(max(residual.max(initial=0.0), 0.0)), broken

    def is_jointly_feasible(
        self, x: Sequence[float], y: Sequence[float], tolerance: float = DEFAULT_TOLERANCE
    ) -> bool:
        violation, _ = self.joint_violation(x, y)
        return violation <= tolerance


# ---------------------------------------------------------------------------
# Reference instance
# ---------------------------------------------------------------------------


def bard_example() -> BilevelLinearProgram:
    """A small textbook linear bilevel program used as a fixed reference.

    Its optimum is not asserted from memory anywhere in this project: the tests
    require three independent methods to agree on it.

        min  x - 4y   s.t.  y in argmin { y : -x - y <= -3, -2x + y <= 0,
                                              2x + y <= 12, 3x - 2y <= 4, y >= 0 }
    """

    return BilevelLinearProgram(
        instance_id="bard_example",
        n_x=1,
        n_y=1,
        upper_sense="minimize",
        c1=[1.0],
        d1=[-4.0],
        A1=[],
        B1=[],
        b1=[],
        d2=[1.0],
        A2=[[-1.0], [-2.0], [2.0], [3.0]],
        B2=[[-1.0], [1.0], [1.0], [-2.0]],
        b2=[-3.0, 0.0, 12.0, 4.0],
        x_upper=[8.0],
        y_upper=[12.0],
        difficulty="reference",
        metadata={"source": "classic linear bilevel textbook example"},
    )


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


def generate_bilevel_lp(
    size: int,
    seed: int = 0,
    difficulty: str = "balanced",
    **_: Any,
) -> BilevelLinearProgram:
    """Reproducible bilevel LP. `size` drives the lower-level dimension.

    Feasibility is structural rather than hoped for: every right-hand side is
    strictly positive, so `(x, y) = (0, 0)` always satisfies both levels, and the
    box bounds keep the joint region compact and hence the program bounded.

    `difficulty` changes the character of the instance, not just its size:

    * `balanced`      - generic coefficients
    * `coupled`       - x strongly shifts the follower's feasible set, which
                        widens the gap between the bilevel and single-level optima
    * `degenerate`    - the follower objective is parallel to a constraint, so the
                        lower level has multiple optima and the optimistic
                        formulation actually matters
    * `large_dual`    - tight right-hand sides and a steep follower objective,
                        which drives the KKT multipliers up and therefore raises
                        the Big-M a correct linearization needs
    """

    if size < 1:
        raise ValueError("size must be >= 1")
    if difficulty not in BILEVEL_DIFFICULTIES:
        # Silently treating an unknown name as `balanced` would mean an
        # experiment could report results for a condition it never ran.
        raise ValueError(
            f"Unknown difficulty `{difficulty}`. Known: {list(BILEVEL_DIFFICULTIES)}"
        )
    rng = _rng(size, seed, difficulty)
    n_y = int(size)
    n_x = max(1, int(size) - 1)
    m_lower = int(size) + 1
    m_upper = int(size)

    # Bilevel tension is the whole point: the follower minimises d2'y with
    # d2 > 0 and so pushes y down, while the leader's d1 < 0 rewards large y.
    # Without this opposition the origin is optimal and the instance says
    # nothing about whether a reformulation preserved the answer.
    c1 = rng.integers(-3, 4, size=n_x).astype(float)
    d1 = -rng.integers(1, 6, size=n_y).astype(float)
    d2 = rng.integers(1, 6, size=n_y).astype(float)

    A1 = rng.integers(-2, 4, size=(m_upper, n_x)).astype(float)
    B1 = rng.integers(-2, 4, size=(m_upper, n_y)).astype(float)
    A2 = rng.integers(-3, 4, size=(m_lower, n_x)).astype(float)
    B2 = rng.integers(-2, 5, size=(m_lower, n_y)).astype(float)

    x_upper = np.full(n_x, 10.0)
    y_upper = np.full(n_y, 10.0)
    b1 = rng.integers(5, 25, size=m_upper).astype(float)
    b2 = rng.integers(5, 25, size=m_lower).astype(float)

    if difficulty == "coupled":
        A2 = rng.integers(-6, 7, size=(m_lower, n_x)).astype(float)
        A2[A2 == 0] = 3.0
    elif difficulty == "degenerate":
        # Make the follower objective parallel to the first lower row so that
        # row's face is entirely optimal: multiple lower-level optima.
        B2[0] = np.maximum(1.0, np.abs(B2[0]))
        d2 = B2[0].copy()
    elif difficulty == "large_dual":
        d2 = rng.integers(20, 60, size=n_y).astype(float)
        b2 = rng.integers(1, 5, size=m_lower).astype(float)
        B2 = np.maximum(1.0, np.abs(B2))

    return BilevelLinearProgram(
        instance_id=f"bilevel_s{size}_{difficulty}_seed{seed}",
        n_x=n_x,
        n_y=n_y,
        upper_sense="minimize",
        c1=[float(v) for v in c1],
        d1=[float(v) for v in d1],
        A1=[[float(v) for v in row] for row in A1],
        B1=[[float(v) for v in row] for row in B1],
        b1=[float(v) for v in b1],
        d2=[float(v) for v in d2],
        A2=[[float(v) for v in row] for row in A2],
        B2=[[float(v) for v in row] for row in B2],
        b2=[float(v) for v in b2],
        x_upper=[float(v) for v in x_upper],
        y_upper=[float(v) for v in y_upper],
        seed=seed,
        difficulty=difficulty,
        metadata={"m_upper": m_upper, "m_lower": m_lower},
    )
