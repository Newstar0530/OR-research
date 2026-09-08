from __future__ import annotations

from pydantic import BaseModel, Field


class SearchStrategy(BaseModel):
    iteration: int
    branch_plans: list[str] = Field(default_factory=list)
    reasoning: str
    should_stop: bool = False
    stop_reason: str | None = None

    def to_markdown(self) -> str:
        lines = [f"## Iteration {self.iteration}", ""]
        lines.append(f"- should_stop: {self.should_stop}")
        if self.stop_reason:
            lines.append(f"- stop_reason: {self.stop_reason}")
        lines.append(f"- reasoning: {self.reasoning}")
        if self.branch_plans:
            lines.append("")
            lines.append("Branch plans:")
            lines.extend(f"- {plan}" for plan in self.branch_plans)
        lines.append("")
        return "\n".join(lines)

