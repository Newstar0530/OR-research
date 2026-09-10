"""QUBO models, the last hop before a quantum sampler.

A QUBO has no constraints and no continuous variables: everything is a bit, and
everything the original problem insisted on has to survive as a penalty in the
energy. That is a lossy-looking rewrite, and two constants decide whether it is
actually lossy -- the discretisation precision and the penalty weight. This
module is the model and the solvers; `qubo_transformations.py` builds one and
`qubo_equivalence.py` decides whether it still answers the original question.

Solvers here come in two kinds on purpose:

* `solve_qubo_by_enumeration` sweeps every bitstring and therefore *proves* the
  ground state. Exponential, so only small models.
* `solve_qubo_by_annealing` is a seeded simulated annealer standing in for the
  sampler a real machine would be. It can only ever report `feasible`, never
  `optimal`, because a sampler that returns a low energy has not proved it is
  the lowest -- and treating a sampler's best draw as the optimum is precisely
  how a quantum pipeline convinces itself of a result it has not got.
"""

from __future__ import annotations

import time
from typing import Any, Literal, Sequence

import numpy as np
from pydantic import BaseModel, Field


QUBOStatus = Literal["optimal", "feasible", "no_solution", "not_run", "error"]


class QuadraticTerm(BaseModel):
    i: int
    j: int
    coefficient: float


class QUBOModel(BaseModel):
    """`E(b) = offset + sum_i a_i b_i + sum_{i<j} q_ij b_i b_j`, `b` in {0,1}^n."""

    name: str = "qubo"
    n_bits: int = 0
    linear: list[float] = Field(default_factory=list)
    quadratic: list[QuadraticTerm] = Field(default_factory=list)
    offset: float = 0.0
    bit_names: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_matrix(self) -> np.ndarray:
        """Upper-triangular matrix with the linear terms on the diagonal."""

        matrix = np.zeros((self.n_bits, self.n_bits), dtype=float)
        np.fill_diagonal(matrix, np.asarray(self.linear, dtype=float))
        for term in self.quadratic:
            i, j = (term.i, term.j) if term.i <= term.j else (term.j, term.i)
            matrix[i, j] += float(term.coefficient)
        return matrix

    def energy(self, bits: Sequence[int | float]) -> float:
        vector = np.asarray(bits, dtype=float)
        if vector.shape[0] != self.n_bits:
            raise ValueError(f"expected {self.n_bits} bits, got {vector.shape[0]}")
        return float(vector @ self.to_matrix() @ vector + self.offset)

    def energies(self, bit_rows: np.ndarray) -> np.ndarray:
        """Vectorised energy for a `(k, n_bits)` block of candidate bitstrings."""

        matrix = self.to_matrix()
        return np.einsum("ki,ij,kj->k", bit_rows, matrix, bit_rows) + self.offset

    @property
    def n_quadratic(self) -> int:
        return len(self.quadratic)

    @property
    def density(self) -> float:
        possible = self.n_bits * (self.n_bits - 1) / 2
        return float(self.n_quadratic / possible) if possible else 0.0

    @property
    def max_abs_coefficient(self) -> float:
        values = [abs(v) for v in self.linear] + [abs(t.coefficient) for t in self.quadratic]
        return max(values, default=0.0)

    @property
    def dynamic_range(self) -> float:
        """Largest over smallest non-zero magnitude.

        Real annealers have finite coefficient precision, so a QUBO whose terms
        span many orders of magnitude cannot be programmed faithfully even when
        it is mathematically correct.

        Note what this does and does not capture. When both extremes are penalty
        terms the weight cancels, so scaling the penalty leaves this ratio
        unchanged; the spread is then set by the squared slack weights. The
        quantity that does move with the penalty is its size relative to the
        objective, reported separately as `penalty_to_objective_ratio`.
        """

        values = [abs(v) for v in self.linear] + [abs(t.coefficient) for t in self.quadratic]
        nonzero = [v for v in values if v > 0]
        return float(max(nonzero) / min(nonzero)) if nonzero else 0.0

    def to_ising(self) -> tuple[np.ndarray, np.ndarray, float]:
        """Convert to Ising form with `b = (1 + s) / 2`, `s` in {-1, +1}.

        Returns `(h, J, offset)` where `E = offset + sum_i h_i s_i +
        sum_{i<j} J_ij s_i s_j`. Quantum hardware speaks Ising, so this is the
        boundary where the model stops being ours.
        """

        matrix = self.to_matrix()
        diagonal = np.diag(matrix).copy()
        couplings = matrix - np.diag(diagonal)
        symmetric = couplings + couplings.T
        h = diagonal / 2.0 + symmetric.sum(axis=1) / 4.0
        J = couplings / 4.0
        offset = float(self.offset + diagonal.sum() / 2.0 + couplings.sum() / 4.0)
        return h, J, offset


class QUBOSolution(BaseModel):
    status: QUBOStatus = "not_run"
    energy: float | None = None
    bits: list[int] = Field(default_factory=list)
    backend: str = "unknown"
    runtime_seconds: float = 0.0
    evaluations: int = 0
    notes: str = ""

    @property
    def has_solution(self) -> bool:
        return self.status in ("optimal", "feasible") and bool(self.bits)

    def add_note(self, note: str) -> None:
        note = note.strip()
        if not note:
            return
        self.notes = f"{self.notes} | {note}".strip(" |") if self.notes else note


def solve_qubo_by_enumeration(
    qubo: QUBOModel,
    max_bits: int = 22,
    chunk_size: int = 1 << 15,
    time_limit: float = 120.0,
) -> QUBOSolution:
    """Sweep every bitstring. The only method here that can prove a ground state."""

    if qubo.n_bits > max_bits:
        return QUBOSolution(
            status="not_run",
            backend="qubo_enumeration",
            notes=(
                f"skipped: {qubo.n_bits} bits exceeds max_bits={max_bits} "
                f"({2 ** qubo.n_bits:.3g} bitstrings)."
            ),
        )
    if qubo.n_bits == 0:
        return QUBOSolution(
            status="optimal", energy=qubo.offset, bits=[], backend="qubo_enumeration"
        )

    started = time.perf_counter()
    matrix = qubo.to_matrix()
    weights = (1 << np.arange(qubo.n_bits, dtype=np.uint64)).astype(np.uint64)
    total = 1 << qubo.n_bits
    best_energy: float | None = None
    best_bits: list[int] = []
    evaluated = 0
    hit_limit = False

    for start in range(0, total, chunk_size):
        if time_limit and (time.perf_counter() - started) > time_limit:
            hit_limit = True
            break
        stop = min(total, start + chunk_size)
        candidates = np.arange(start, stop, dtype=np.uint64)
        rows = ((candidates[:, None] & weights[None, :]) > 0).astype(float)
        values = np.einsum("ki,ij,kj->k", rows, matrix, rows) + qubo.offset
        evaluated += rows.shape[0]
        position = int(np.argmin(values))
        if best_energy is None or values[position] < best_energy:
            best_energy = float(values[position])
            best_bits = [int(v) for v in rows[position]]

    runtime = time.perf_counter() - started
    if best_energy is None:
        return QUBOSolution(
            status="no_solution", backend="qubo_enumeration", runtime_seconds=runtime
        )
    return QUBOSolution(
        status="feasible" if hit_limit else "optimal",
        energy=best_energy,
        bits=best_bits,
        backend="qubo_enumeration",
        runtime_seconds=runtime,
        evaluations=evaluated,
        notes=(
            "stopped on the time limit; the ground state is NOT proved"
            if hit_limit
            else f"swept all {total} bitstrings; ground state proved"
        ),
    )


def solve_qubo_by_annealing(
    qubo: QUBOModel,
    sweeps: int = 2000,
    restarts: int = 8,
    seed: int = 0,
    initial_temperature: float | None = None,
    final_temperature: float = 1e-3,
    time_limit: float = 60.0,
) -> QUBOSolution:
    """Seeded simulated annealing, standing in for a hardware sampler.

    Reports `feasible`, never `optimal`: a low energy is not a proof of the
    lowest, and this project does not let a sampler grade its own homework.
    """

    if qubo.n_bits == 0:
        return QUBOSolution(status="feasible", energy=qubo.offset, bits=[], backend="qubo_annealing")

    started = time.perf_counter()
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(qubo.n_bits)]))
    matrix = qubo.to_matrix()
    symmetric = matrix + matrix.T - np.diag(np.diag(matrix))
    scale = initial_temperature or max(1e-6, qubo.max_abs_coefficient)

    best_energy: float | None = None
    best_bits: np.ndarray | None = None
    evaluations = 0

    for _ in range(max(1, restarts)):
        if time_limit and (time.perf_counter() - started) > time_limit:
            break
        bits = rng.integers(0, 2, size=qubo.n_bits).astype(float)
        energy = float(bits @ matrix @ bits + qubo.offset)
        for step in range(max(1, sweeps)):
            temperature = scale * (final_temperature / scale) ** (step / max(1, sweeps - 1))
            index = int(rng.integers(0, qubo.n_bits))
            direction = 1.0 - 2.0 * bits[index]
            # delta for flipping one bit, using the symmetrised couplings
            delta = direction * (matrix[index, index] + symmetric[index] @ bits - matrix[index, index] * bits[index])
            evaluations += 1
            if delta <= 0 or rng.random() < np.exp(-delta / max(temperature, 1e-12)):
                bits[index] += direction
                energy += delta
        energy = float(bits @ matrix @ bits + qubo.offset)  # recompute, never trust the running sum
        if best_energy is None or energy < best_energy:
            best_energy = energy
            best_bits = bits.copy()

    runtime = time.perf_counter() - started
    if best_bits is None:
        return QUBOSolution(
            status="no_solution", backend="qubo_annealing", runtime_seconds=runtime
        )
    return QUBOSolution(
        status="feasible",
        energy=best_energy,
        bits=[int(round(v)) for v in best_bits],
        backend="qubo_annealing",
        runtime_seconds=runtime,
        evaluations=evaluations,
        notes=f"{restarts} restart(s) x {sweeps} sweeps; optimality is not claimed",
    )


def solve_qubo(
    qubo: QUBOModel,
    max_bits_for_proof: int = 22,
    time_limit: float = 120.0,
    seed: int = 0,
) -> QUBOSolution:
    """Prove the ground state when the model is small enough, otherwise anneal."""

    if qubo.n_bits <= max_bits_for_proof:
        return solve_qubo_by_enumeration(qubo, max_bits=max_bits_for_proof, time_limit=time_limit)
    solution = solve_qubo_by_annealing(qubo, seed=seed, time_limit=time_limit)
    solution.add_note(
        f"{qubo.n_bits} bits is beyond exhaustive proof; this energy is a sampler's best draw"
    )
    return solution
