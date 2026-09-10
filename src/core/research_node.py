from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field

from src.schemas import ExecutionResult


NodeStatus = Literal["planned", "success", "failed", "timeout", "contract_failed"]

#: How this node came to exist. `draft` starts a fresh tree from the base code,
#: `improve` mutates a successful ancestor, `debug` attempts to repair a failed
#: one. The distinction is what makes a search tree more than a chain.
NodeKind = Literal["draft", "improve", "debug"]


class ResearchNode(BaseModel):
    id: str = Field(default_factory=lambda: uuid4().hex[:12])
    parent_id: str | None = None
    iteration: int
    branch_index: int
    plan: str
    code_path: str | None = None
    work_dir: str | None = None
    status: NodeStatus = "planned"
    metric_name: str | None = None
    metric_value: float | None = None
    maximize: bool = False
    analysis: str = ""
    kind: NodeKind = "improve"
    depth: int = 0
    #: How many consecutive repair attempts led here. Bounded so the search
    #: cannot spend itself re-fixing one broken branch forever.
    debug_depth: int = 0
    exception_type: str | None = None
    exception_message: str | None = None
    failing_line: int | None = None
    failure_summary: str = ""
    created_at: datetime = Field(default_factory=datetime.now)
    execution: ExecutionResult | None = None
    artifacts: dict[str, str] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def is_successful(self) -> bool:
        return self.status == "success" and self.metric_value is not None

    @property
    def is_buggy(self) -> bool:
        """Ran and failed, so there is something concrete to repair."""

        return self.status in ("failed", "timeout", "contract_failed")

    def better_than(self, other: "ResearchNode | None") -> bool:
        if other is None:
            return self.is_successful
        if not self.is_successful:
            return False
        if not other.is_successful:
            return True
        assert self.metric_value is not None and other.metric_value is not None
        return self.metric_value > other.metric_value if self.maximize else self.metric_value < other.metric_value

    def relative_code_path(self, root: Path) -> str | None:
        if not self.code_path:
            return None
        try:
            return str(Path(self.code_path).resolve().relative_to(root.resolve()))
        except ValueError:
            return self.code_path

