"""Reproducible MI 0-1 (pure binary) integer programming instances.

Everything is expressed in one general form so that a single solver layer, a
single checker and a single results contract cover every family:

    maximize / minimize   c' x
    subject to            a_i' x  {<=, >=, ==}  b_i      for each constraint i
                          x_j in {0, 1}

Instances are generated from an explicit `(family, size, difficulty, seed)`
tuple through `numpy.random.SeedSequence`, so the same tuple always produces
byte-identical data on any machine. Coefficients are integral by construction,
which keeps CP-SAT exact and keeps brute-force ground truth meaningful.
"""

from __future__ import annotations

import json
import zlib
from pathlib import Path
from typing import Any, Callable, Literal, Sequence

import numpy as np
from pydantic import BaseModel, Field

from src.core.solve_result import ObjectiveSense


ConstraintSense = Literal["<=", ">=", "=="]

Difficulty = Literal[
    "uncorrelated",
    "weakly_correlated",
    "strongly_correlated",
    "uniform_cost",
    "unicost",
]


def _code(text: str) -> int:
    """Process-stable integer code for a string (`hash()` is salted, so no)."""

    return int(zlib.crc32(text.encode("utf-8")) & 0x7FFFFFFF)


def _rng(family: str, size: int, seed: int, difficulty: str) -> np.random.Generator:
    sequence = np.random.SeedSequence([int(seed), int(size), _code(family), _code(difficulty)])
    return np.random.default_rng(sequence)


class LinearConstraint(BaseModel):
    """One linear constraint over binary variables."""

    name: str
    coefficients: list[float]
    sense: ConstraintSense
    rhs: float

    def lhs(self, solution: Sequence[int | float]) -> float:
        return float(sum(float(a) * float(x) for a, x in zip(self.coefficients, solution)))

    def violation(self, solution: Sequence[int | float]) -> float:
        """True violation magnitude (0.0 when the constraint holds exactly).

        No tolerance is applied here on purpose: the raw magnitude is what gets
        reported, and the caller decides what counts as "close enough".
        """

        value = self.lhs(solution)
        if self.sense == "<=":
            return max(0.0, value - self.rhs)
        if self.sense == ">=":
            return max(0.0, self.rhs - value)
        return abs(value - self.rhs)

    def is_satisfied(self, solution: Sequence[int | float], tolerance: float = 1e-6) -> bool:
        return self.violation(solution) <= tolerance

    @property
    def is_integral(self) -> bool:
        return all(float(a).is_integer() for a in self.coefficients) and float(self.rhs).is_integer()


class BinaryProgramInstance(BaseModel):
    """A concrete, solvable, reproducible 0-1 integer program."""

    instance_id: str
    family: str
    objective_sense: ObjectiveSense
    objective_coefficients: list[float]
    constraints: list[LinearConstraint] = Field(default_factory=list)
    seed: int = 0
    difficulty: str = "uncorrelated"
    known_optimum: float | None = None
    known_optimum_source: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def n_vars(self) -> int:
        return len(self.objective_coefficients)

    @property
    def n_constraints(self) -> int:
        return len(self.constraints)

    @property
    def is_integral(self) -> bool:
        """True when every coefficient is an integer (CP-SAT can be exact)."""

        objective_integral = all(float(c).is_integer() for c in self.objective_coefficients)
        return objective_integral and all(c.is_integral for c in self.constraints)

    def objective_value(self, solution: Sequence[int | float]) -> float:
        if len(solution) != self.n_vars:
            raise ValueError(
                f"solution has {len(solution)} entries but the instance has {self.n_vars} variables"
            )
        return float(
            sum(float(c) * float(x) for c, x in zip(self.objective_coefficients, solution))
        )

    def is_better(self, candidate: float, incumbent: float | None) -> bool:
        if incumbent is None:
            return True
        return candidate > incumbent if self.objective_sense == "maximize" else candidate < incumbent

    def to_json_file(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        return target

    @classmethod
    def from_json_file(cls, path: str | Path) -> "BinaryProgramInstance":
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))


# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------


def generate_knapsack(
    size: int,
    seed: int = 0,
    difficulty: str = "uncorrelated",
    capacity_ratio: float = 0.5,
    weight_max: int = 100,
    **_: Any,
) -> BinaryProgramInstance:
    """Single-constraint 0-1 knapsack.

    `difficulty` controls profit/weight correlation, which is what actually
    makes a knapsack easy or hard: strongly correlated instances defeat the
    ratio-greedy heuristic, uncorrelated ones do not.
    """

    if size < 1:
        raise ValueError("size must be >= 1")
    rng = _rng("knapsack", size, seed, difficulty)
    weights = rng.integers(1, weight_max + 1, size=size).astype(float)
    if difficulty == "strongly_correlated":
        profits = weights + 10.0
    elif difficulty == "weakly_correlated":
        noise = rng.integers(-10, 11, size=size).astype(float)
        profits = np.maximum(1.0, weights + noise)
    else:
        profits = rng.integers(1, weight_max + 1, size=size).astype(float)
    capacity = float(max(1, int(round(capacity_ratio * float(weights.sum())))))
    return BinaryProgramInstance(
        instance_id=f"knapsack_n{size}_{difficulty}_s{seed}",
        family="knapsack",
        objective_sense="maximize",
        objective_coefficients=[float(p) for p in profits],
        constraints=[
            LinearConstraint(
                name="capacity",
                coefficients=[float(w) for w in weights],
                sense="<=",
                rhs=capacity,
            )
        ],
        seed=seed,
        difficulty=difficulty,
        metadata={
            "capacity_ratio": capacity_ratio,
            "capacity": capacity,
            "total_weight": float(weights.sum()),
        },
    )


def generate_multi_knapsack(
    size: int,
    seed: int = 0,
    difficulty: str = "uncorrelated",
    n_resources: int = 3,
    capacity_ratio: float = 0.5,
    weight_max: int = 100,
    **_: Any,
) -> BinaryProgramInstance:
    """Multidimensional 0-1 knapsack (`n_resources` capacity constraints)."""

    if size < 1:
        raise ValueError("size must be >= 1")
    n_resources = max(1, int(n_resources))
    rng = _rng("multi_knapsack", size, seed, difficulty)
    weights = rng.integers(1, weight_max + 1, size=(n_resources, size)).astype(float)
    mean_weight = weights.mean(axis=0)
    if difficulty == "strongly_correlated":
        profits = mean_weight + 10.0
    elif difficulty == "weakly_correlated":
        profits = np.maximum(1.0, mean_weight + rng.integers(-10, 11, size=size))
    else:
        profits = rng.integers(1, weight_max + 1, size=size).astype(float)
    constraints = [
        LinearConstraint(
            name=f"resource_{i}",
            coefficients=[float(w) for w in weights[i]],
            sense="<=",
            rhs=float(max(1, int(round(capacity_ratio * float(weights[i].sum()))))),
        )
        for i in range(n_resources)
    ]
    return BinaryProgramInstance(
        instance_id=f"multi_knapsack_n{size}_m{n_resources}_{difficulty}_s{seed}",
        family="multi_knapsack",
        objective_sense="maximize",
        objective_coefficients=[float(round(p)) for p in profits],
        constraints=constraints,
        seed=seed,
        difficulty=difficulty,
        metadata={"n_resources": n_resources, "capacity_ratio": capacity_ratio},
    )


def generate_set_cover(
    size: int,
    seed: int = 0,
    difficulty: str = "uniform_cost",
    n_items: int | None = None,
    coverage_probability: float = 0.25,
    cost_max: int = 100,
    **_: Any,
) -> BinaryProgramInstance:
    """Minimum-cost set cover: `size` candidate sets, `n_items` items to cover.

    Every item is guaranteed to be coverable, so the instance is always
    feasible -- an infeasible answer from a method is therefore a real defect,
    not a property of the instance.
    """

    if size < 1:
        raise ValueError("size must be >= 1")
    rng = _rng("set_cover", size, seed, difficulty)
    n_items = int(n_items) if n_items else max(4, int(round(size * 0.75)))
    incidence = (rng.random((n_items, size)) < coverage_probability).astype(float)
    for item in range(n_items):
        if incidence[item].sum() == 0:
            incidence[item, int(rng.integers(0, size))] = 1.0
    if difficulty == "unicost":
        costs = np.ones(size, dtype=float)
    else:
        costs = rng.integers(1, cost_max + 1, size=size).astype(float)
    constraints = [
        LinearConstraint(
            name=f"cover_item_{item}",
            coefficients=[float(v) for v in incidence[item]],
            sense=">=",
            rhs=1.0,
        )
        for item in range(n_items)
    ]
    return BinaryProgramInstance(
        instance_id=f"set_cover_n{size}_i{n_items}_{difficulty}_s{seed}",
        family="set_cover",
        objective_sense="minimize",
        objective_coefficients=[float(c) for c in costs],
        constraints=constraints,
        seed=seed,
        difficulty=difficulty,
        metadata={"n_items": n_items, "coverage_probability": coverage_probability},
    )


INSTANCE_FAMILIES: dict[str, Callable[..., BinaryProgramInstance]] = {
    "knapsack": generate_knapsack,
    "multi_knapsack": generate_multi_knapsack,
    "set_cover": generate_set_cover,
}

DEFAULT_DIFFICULTY: dict[str, str] = {
    "knapsack": "uncorrelated",
    "multi_knapsack": "weakly_correlated",
    "set_cover": "uniform_cost",
}


def list_families() -> list[str]:
    return sorted(INSTANCE_FAMILIES)


def generate_instance(
    family: str,
    size: int,
    seed: int = 0,
    difficulty: str | None = None,
    **kwargs: Any,
) -> BinaryProgramInstance:
    """Dispatch to a family generator. `size` is the number of binary variables."""

    if family not in INSTANCE_FAMILIES:
        raise KeyError(f"Unknown instance family `{family}`. Known: {list_families()}")
    resolved_difficulty = difficulty or DEFAULT_DIFFICULTY.get(family, "uncorrelated")
    return INSTANCE_FAMILIES[family](
        size=size, seed=seed, difficulty=resolved_difficulty, **kwargs
    )
