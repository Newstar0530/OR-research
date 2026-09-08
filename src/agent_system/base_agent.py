from __future__ import annotations

from abc import ABC, abstractmethod

from src.agent_system.action import AgentAction
from src.agent_system.state import ResearchState
from src.llm_client import LLMClient


class BaseResearchAgent(ABC):
    name: str = "base_agent"
    role: str = "generic"

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    @abstractmethod
    def decide(self, state: ResearchState) -> AgentAction:
        raise NotImplementedError

    def _json_action(self, system_prompt: str, user_prompt: str, fallback: AgentAction) -> AgentAction:
        if self.llm.use_mock:
            return fallback
        try:
            payload = self.llm.chat_json(system_prompt, user_prompt)
            return AgentAction(**payload)
        except Exception:
            return fallback

