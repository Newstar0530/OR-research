from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


MutationKind = Literal[
    "baseline_strengthening",
    "proposed_intensification",
    "component_ablation",
    "stress_test",
    "replication_check",
]


class MutationSpec(BaseModel):
    kind: MutationKind
    description: str
    replacements: dict[str, str] = Field(default_factory=dict)
    metadata: dict[str, str | float | int | bool] = Field(default_factory=dict)

    def to_plan_suffix(self) -> str:
        return f"mutation={self.kind}; {self.description}"

