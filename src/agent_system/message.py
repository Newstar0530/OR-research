from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from src.agent_system.action import AgentAction, PolicyDecision


class AgentMessage(BaseModel):
    agent_name: str
    role: str
    action: AgentAction
    policy_decision: PolicyDecision | None = None
    timestamp: datetime = Field(default_factory=datetime.now)

