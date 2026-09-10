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


class MutationOutcome(BaseModel):
    """What actually happened when a mutation was applied to a script.

    The old refiner returned a bare string, so a mutation that changed nothing
    was indistinguishable from one that changed everything. On any template
    without the exact constant names the replacements name, every replacement
    was a no-op and the node became a copy of its parent -- scored and ranked
    as an explored branch.
    """

    code: str
    #: `llm` when a model rewrote the script, `string_replacement` when the
    #: bounded find/replace applied, `none` when nothing changed the behaviour.
    method: str = "none"
    applied: bool = False
    #: Replacement keys that were present in the parent, and those that were not.
    matched_replacements: list[str] = Field(default_factory=list)
    unmatched_replacements: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

    @property
    def is_effective(self) -> bool:
        return self.applied and self.method != "none"

    def summary(self) -> str:
        if self.is_effective:
            detail = (
                f"{len(self.matched_replacements)} replacement(s) applied"
                if self.method == "string_replacement"
                else "a model rewrote the script"
            )
            return f"{self.method}: {detail}."
        return "no effect: " + ("; ".join(self.notes) or "nothing changed the script's behaviour.")
