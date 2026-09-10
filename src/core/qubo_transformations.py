"""MI 0-1 -> binary grid -> QUBO, with the two lossy constants kept separate.

A QUBO has no constraints and no continuous variables, so getting there costs
two things, and they fail differently:

* **Discretisation.** A continuous variable becomes a fixed-point sum of bits,
  so an optimum that does not sit on the grid is simply not representable. That
  is a *quantified cost*, not a bug, and it must not be confused with one --
  which is why `to_binary_grid_mip` produces a standalone model that can still
  be solved exactly. The gap between it and the original measures the cost of
  the grid and nothing else.

* **Penalty weight.** Constraints survive only as `P * residual^2` added to the
  energy. Too small and an infeasible bitstring wins; the sampler then returns
  a confident answer to a problem nobody posed. This is the same failure as a
  Big-M that is too small, in a different coordinate system, and
  `derive_penalty_bound` treats it the same way: derive a constant that is
  justified, and say plainly when it is not.

The reverse pressure is real too. Penalties inflate the coefficient spread, and
hardware has finite precision, so `QUBOModel.dynamic_range` is reported next to
correctness rather than after it.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Literal, Sequence

import numpy as np
from pydantic import BaseModel, Field

from src.core.mixed_integer_program import (
    MipConstraint,
    MipVariable,
    MixedIntegerProgram,
)
from src.core.qubo import QuadraticTerm, QUBOModel


DEFAULT_PRECISION = 1.0
DEFAULT_TOLERANCE = 1e-6


class BitExpansion(BaseModel):
    """`value = lower + sum_k weights[k] * bit[k]`."""

    name: str
    lower: float
    upper: float
    precision: float
    weights: list[float] = Field(default_factory=list)
    bit_indices: list[int] = Field(default_factory=list)
    is_original_binary: bool = False

    @property
    def n_bits(self) -> int:
        return len(self.weights)

    @property
    def representable_upper(self) -> float:
        return float(self.lower + sum(self.weights))

    def value(self, bits: Sequence[int | float]) -> float:
        return float(
            self.lower + sum(w * float(bits[i]) for w, i in zip(self.weights, self.bit_indices))
        )


def binary_expansion(
    lower: float, upper: float, precision: float = DEFAULT_PRECISION, name: str = "v"
) -> BitExpansion:
    """Tight fixed-point expansion of `[lower, upper]` at the given precision.

    The last weight is trimmed so the representable maximum lands exactly on
    `upper` rather than overshooting it: an expansion that can express values
    the original variable could not is a relaxation, and would let the QUBO
    "find" points the source model forbids.
    """

    if not math.isfinite(lower) or not math.isfinite(upper):
        raise ValueError(f"`{name}` needs finite bounds to be discretised, got [{lower}, {upper}]")
    if upper < lower:
        raise ValueError(f"`{name}` has an empty range [{lower}, {upper}]")
    if precision <= 0:
        raise ValueError("precision must be positive")

    span = float(upper) - float(lower)
    if span <= 0:
        return BitExpansion(name=name, lower=float(lower), upper=float(upper), precision=precision)
    steps = span / precision
    n_bits = max(1, int(math.ceil(math.log2(steps + 1.0) - 1e-12)))
    weights = [precision * (2.0**k) for k in range(n_bits - 1)]
    remainder = span - sum(weights)
    if remainder <= 0:  # pragma: no cover - defensive; the bit count guarantees it
        weights = weights[:-1]
        remainder = span - sum(weights)
    weights.append(remainder)
    return BitExpansion(
        name=name,
        lower=float(lower),
        upper=float(upper),
        precision=float(precision),
        weights=[float(w) for w in weights],
    )


class GridEncoding(BaseModel):
    """How every variable of a MIP became bits."""

    source_name: str
    precision: float
    expansions: list[BitExpansion] = Field(default_factory=list)
    n_bits: int = 0
    discretised_variables: list[str] = Field(default_factory=list)

    @property
    def is_lossless(self) -> bool:
        """True when nothing was discretised, so the grid cannot cost anything."""

        return not self.discretised_variables

    def decode(self, bits: Sequence[int | float]) -> list[float]:
        return [expansion.value(bits) for expansion in self.expansions]


def to_binary_grid_mip(
    mip: MixedIntegerProgram, precision: float = DEFAULT_PRECISION
) -> tuple[MixedIntegerProgram, GridEncoding]:
    """Replace every variable by its bits, keeping the constraints intact.

    The result is still a constrained model, so it can be solved exactly by the
    ordinary MIP machinery. That is the point: it isolates the cost of the grid
    from the cost of penalising the constraints away.
    """

    expansions: list[BitExpansion] = []
    discretised: list[str] = []
    cursor = 0
    for index, variable in enumerate(mip.variables):
        if variable.is_binary:
            expansion = BitExpansion(
                name=variable.name,
                lower=0.0,
                upper=1.0,
                precision=1.0,
                weights=[1.0],
                is_original_binary=True,
            )
        else:
            expansion = binary_expansion(
                variable.effective_lower, variable.effective_upper, precision, variable.name
            )
            if expansion.n_bits:
                discretised.append(variable.name)
        expansion.bit_indices = list(range(cursor, cursor + expansion.n_bits))
        cursor += expansion.n_bits
        expansions.append(expansion)

    encoding = GridEncoding(
        source_name=mip.name,
        precision=precision,
        expansions=expansions,
        n_bits=cursor,
        discretised_variables=discretised,
    )

    def project(coefficients: Sequence[float]) -> tuple[list[float], float]:
        """Rewrite `sum_j a_j v_j` over bits; returns (bit coefficients, constant)."""

        row = [0.0] * cursor
        constant = 0.0
        for index, coefficient in enumerate(coefficients):
            coefficient = float(coefficient)
            if coefficient == 0.0:
                continue
            expansion = expansions[index]
            constant += coefficient * expansion.lower
            for weight, bit in zip(expansion.weights, expansion.bit_indices):
                row[bit] += coefficient * weight
        return row, constant

    objective_row, objective_constant = project(mip.objective)
    constraints: list[MipConstraint] = []
    for constraint in mip.constraints:
        row, constant = project(constraint.coefficients)
        constraints.append(
            MipConstraint(
                name=constraint.name,
                coefficients=row,
                sense=constraint.sense,
                rhs=float(constraint.rhs) - constant,
            )
        )

    grid_mip = MixedIntegerProgram(
        name=f"{mip.name}_binary_grid",
        sense=mip.sense,
        variables=[
            MipVariable(name=f"{expansion.name}#b{k}", is_binary=True)
            for expansion in expansions
            for k in range(expansion.n_bits)
        ],
        objective=objective_row,
        objective_offset=float(mip.objective_offset) + objective_constant,
        constraints=constraints,
        metadata={
            "source": mip.name,
            "precision": precision,
            "discretised_variables": discretised,
            "n_bits": cursor,
        },
    )
    return grid_mip, encoding


# ---------------------------------------------------------------------------
# Penalty derivation
# ---------------------------------------------------------------------------


class PenaltyBound(BaseModel):
    penalty: float
    objective_range: float
    minimum_violation: float
    is_rigorous: bool
    integral_data: bool
    method: str = "objective_range_over_minimum_squared_violation"
    notes: list[str] = Field(default_factory=list)


def _is_integral(values: Sequence[float], tolerance: float = 1e-9) -> bool:
    return all(abs(float(v) - round(float(v))) <= tolerance for v in values)


def derive_penalty_bound(
    grid_mip: MixedIntegerProgram, safety_factor: float = 1.0
) -> PenaltyBound:
    """A penalty large enough that breaking a constraint can never pay.

    The largest the objective can swing over all bitstrings is `sum_k |c_k|`.
    If every coefficient and right-hand side is an integer then every residual
    is an integer too, so the smallest non-zero violation contributes at least
    `1` to the squared penalty, and `P > sum_k |c_k|` is sufficient -- a real
    bound, not a habit.

    With fractional data the smallest non-zero residual cannot be bounded this
    cheaply, so the function says the constant is heuristic instead of implying
    a guarantee it has not got.
    """

    objective_range = float(sum(abs(float(c)) for c in grid_mip.objective))
    data: list[float] = list(grid_mip.objective)
    for constraint in grid_mip.constraints:
        data.extend(constraint.coefficients)
        data.append(constraint.rhs)
    integral = _is_integral(data)
    notes: list[str] = []
    if integral:
        minimum_violation = 1.0
        penalty = (objective_range + 1.0) * safety_factor
        notes.append(
            "all coefficients and right-hand sides are integers, so any violated constraint "
            "contributes at least 1 to the squared penalty; the bound is rigorous."
        )
    else:
        minimum_violation = 0.0
        penalty = (objective_range + 1.0) * 10.0 * safety_factor
        notes.append(
            "the data is not integral, so the smallest non-zero residual cannot be bounded by "
            "this argument; the constant is heuristic and the equivalence check is what decides."
        )
    return PenaltyBound(
        penalty=penalty,
        objective_range=objective_range,
        minimum_violation=minimum_violation,
        is_rigorous=integral,
        integral_data=integral,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Grid MIP -> QUBO
# ---------------------------------------------------------------------------


class SlackEncoding(BaseModel):
    constraint_name: str
    constraint_index: int
    expansion: BitExpansion


class QUBOEncoding(BaseModel):
    source_name: str
    grid: GridEncoding
    slacks: list[SlackEncoding] = Field(default_factory=list)
    penalty: float = 0.0
    sense: Literal["minimize", "maximize"] = "minimize"
    energy_sign: float = 1.0
    n_bits: int = 0
    penalty_is_rigorous: bool = False
    notes: list[str] = Field(default_factory=list)

    def decode_variables(self, bits: Sequence[int | float]) -> list[float]:
        return self.grid.decode(bits)

    def decode_slacks(self, bits: Sequence[int | float]) -> list[float]:
        return [slack.expansion.value(bits) for slack in self.slacks]

    def objective_from_energy(self, energy: float, penalty_cost: float = 0.0) -> float:
        """Recover the source objective from an energy, given the penalty paid."""

        return float((energy - penalty_cost) * self.energy_sign)


def to_qubo(
    grid_mip: MixedIntegerProgram,
    penalty: float,
    slack_precision: float | None = None,
) -> tuple[QUBOModel, QUBOEncoding]:
    """Add slack bits, fold every constraint into the energy as `P * residual^2`.

    `>=` rows are negated into `<=` rows; `==` rows get no slack. Slack ranges
    come from interval arithmetic over the bits, so a constraint that no
    bitstring can satisfy is reported here rather than becoming an unsolvable
    QUBO nobody can diagnose.
    """

    if not all(v.is_binary for v in grid_mip.variables):
        raise ValueError("to_qubo expects an all-binary model; run to_binary_grid_mip first")
    if not math.isfinite(penalty) or penalty <= 0:
        raise ValueError(
            "the penalty must be a positive finite number; an unbounded penalty is not a model."
        )

    n_grid = grid_mip.n_vars
    cursor = n_grid
    slacks: list[SlackEncoding] = []
    rows: list[tuple[list[float], float]] = []  # (coefficients over all bits, constant r)
    notes: list[str] = []

    integral_data = _is_integral(
        [c for row in grid_mip.constraints for c in row.coefficients]
        + [row.rhs for row in grid_mip.constraints]
    )
    resolved_slack_precision = slack_precision or (1.0 if integral_data else DEFAULT_PRECISION)

    prepared: list[tuple[MipConstraint, list[float], float, str]] = []
    for index, constraint in enumerate(grid_mip.constraints):
        coefficients = [float(a) for a in constraint.coefficients]
        rhs = float(constraint.rhs)
        sense = constraint.sense
        if sense == ">=":
            coefficients = [-a for a in coefficients]
            rhs = -rhs
            sense = "<="
        prepared.append((constraint, coefficients, rhs, sense))

    for index, (constraint, coefficients, rhs, sense) in enumerate(prepared):
        if sense == "<=":
            minimum = sum(a for a in coefficients if a < 0)
            slack_upper = rhs - minimum
            if slack_upper < -DEFAULT_TOLERANCE:
                raise ValueError(
                    f"constraint `{constraint.name}` cannot be satisfied by any bitstring "
                    f"(its left-hand side is at least {minimum:.6g} > {rhs:.6g})"
                )
            slack_upper = max(0.0, slack_upper)
            expansion = binary_expansion(
                0.0, slack_upper, resolved_slack_precision, f"slack[{constraint.name}]"
            )
            expansion.bit_indices = list(range(cursor, cursor + expansion.n_bits))
            cursor += expansion.n_bits
            slacks.append(
                SlackEncoding(
                    constraint_name=constraint.name, constraint_index=index, expansion=expansion
                )
            )
            rows.append((coefficients, rhs, expansion))  # type: ignore[arg-type]
        else:
            rows.append((coefficients, rhs, None))  # type: ignore[arg-type]

    total_bits = cursor
    sign = 1.0 if grid_mip.sense == "minimize" else -1.0
    linear = np.zeros(total_bits, dtype=float)
    quadratic: dict[tuple[int, int], float] = defaultdict(float)
    offset = sign * float(grid_mip.objective_offset)

    for bit, coefficient in enumerate(grid_mip.objective):
        linear[bit] += sign * float(coefficient)

    for coefficients, rhs, expansion in rows:  # type: ignore[misc]
        combined = np.zeros(total_bits, dtype=float)
        combined[:n_grid] = np.asarray(coefficients, dtype=float)
        constant = float(rhs)
        if expansion is not None:
            for weight, bit in zip(expansion.weights, expansion.bit_indices):
                combined[bit] += float(weight)
            constant -= expansion.lower
        # (combined . b - constant)^2 with b_k^2 = b_k
        support = np.nonzero(combined)[0]
        for k in support:
            linear[k] += penalty * (combined[k] ** 2 - 2.0 * constant * combined[k])
        for a in range(len(support)):
            for b in range(a + 1, len(support)):
                k, l = int(support[a]), int(support[b])
                quadratic[(k, l)] += penalty * 2.0 * combined[k] * combined[l]
        offset += penalty * constant**2

    bit_names = [v.name for v in grid_mip.variables] + [
        f"{slack.expansion.name}#b{k}"
        for slack in slacks
        for k in range(slack.expansion.n_bits)
    ]
    qubo = QUBOModel(
        name=f"{grid_mip.name}_qubo",
        n_bits=total_bits,
        linear=[float(v) for v in linear],
        quadratic=[
            QuadraticTerm(i=i, j=j, coefficient=float(v))
            for (i, j), v in sorted(quadratic.items())
            if v != 0.0
        ],
        offset=float(offset),
        bit_names=bit_names,
        metadata={
            "source": grid_mip.name,
            "penalty": penalty,
            "n_grid_bits": n_grid,
            "n_slack_bits": total_bits - n_grid,
            "sense": grid_mip.sense,
            "slack_precision": resolved_slack_precision,
        },
    )
    if not integral_data:
        notes.append(
            "constraint data is not integral, so slack bits only approximate the slack range"
        )
    encoding = QUBOEncoding(
        source_name=grid_mip.name,
        grid=GridEncoding(
            source_name=grid_mip.metadata.get("source", grid_mip.name),
            precision=float(grid_mip.metadata.get("precision", DEFAULT_PRECISION)),
            expansions=[],
            n_bits=n_grid,
        ),
        slacks=slacks,
        penalty=float(penalty),
        sense=grid_mip.sense,
        energy_sign=sign,
        n_bits=total_bits,
        notes=notes,
    )
    return qubo, encoding


def build_qubo(
    mip: MixedIntegerProgram,
    penalty: float | None = None,
    precision: float = DEFAULT_PRECISION,
    slack_precision: float | None = None,
) -> tuple[QUBOModel, QUBOEncoding, MixedIntegerProgram, PenaltyBound]:
    """The whole hop: source MIP -> binary grid -> QUBO, with a derived penalty."""

    grid_mip, grid_encoding = to_binary_grid_mip(mip, precision=precision)
    bound = derive_penalty_bound(grid_mip)
    resolved_penalty = bound.penalty if penalty is None else float(penalty)
    qubo, encoding = to_qubo(grid_mip, resolved_penalty, slack_precision=slack_precision)
    encoding.grid = grid_encoding
    encoding.penalty_is_rigorous = bound.is_rigorous and penalty is None
    encoding.notes = list(encoding.notes) + list(bound.notes)
    return qubo, encoding, grid_mip, bound
