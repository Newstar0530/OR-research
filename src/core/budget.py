from __future__ import annotations

from dataclasses import dataclass
from time import monotonic


@dataclass
class ResearchBudget:
    max_iterations: int
    max_branches: int
    timeout_seconds: int
    patience: int = 2
    min_improvement: float = 0.0
    started_at: float = monotonic()

    def allow_iteration(self, iteration: int) -> bool:
        return iteration < self.max_iterations

    def allow_branch(self, branch_index: int) -> bool:
        return branch_index < self.max_branches

    @property
    def elapsed_seconds(self) -> float:
        return monotonic() - self.started_at
