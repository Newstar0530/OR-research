from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class ResearchState(BaseModel):
    project_name: str
    research_goal: str
    domain_profile: str
    run_dir: str
    hypotheses: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    proposed_experiments: list[str] = Field(default_factory=list)
    search_directives: list[str] = Field(default_factory=list)
    mutation_directives: list[str] = Field(default_factory=list)
    stopping_rules: list[str] = Field(default_factory=list)
    human_checks: list[str] = Field(default_factory=list)
    artifacts: dict[str, str] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def run_path(self) -> Path:
        return Path(self.run_dir)

    def brief(self) -> str:
        return (
            f"Project: {self.project_name}\n"
            f"Goal: {self.research_goal}\n"
            f"Domain profile: {self.domain_profile}\n"
            f"Hypotheses: {self.hypotheses}\n"
            f"Open questions: {self.open_questions}\n"
            f"Evidence: {self.evidence}\n"
            f"Risks: {self.risks}\n"
            f"Search directives: {self.search_directives}\n"
            f"Mutation directives: {self.mutation_directives}\n"
        )
