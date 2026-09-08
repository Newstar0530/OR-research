from __future__ import annotations

import json
import time
from pathlib import Path

from pydantic import BaseModel, Field


class LLMInteraction(BaseModel):
    index: int
    provider: str
    model_name: str
    system_prompt_preview: str
    user_prompt_preview: str
    response_preview: str
    prompt_chars: int
    response_chars: int
    approximate_prompt_tokens: int
    approximate_response_tokens: int
    duration_seconds: float


class LLMInteractionTracker(BaseModel):
    interactions: list[LLMInteraction] = Field(default_factory=list)

    def record(
        self,
        provider: str,
        model_name: str,
        system_prompt: str,
        user_prompt: str,
        response: str,
        duration_seconds: float,
    ) -> None:
        prompt_chars = len(system_prompt) + len(user_prompt)
        response_chars = len(response)
        self.interactions.append(
            LLMInteraction(
                index=len(self.interactions),
                provider=provider,
                model_name=model_name,
                system_prompt_preview=_preview(system_prompt),
                user_prompt_preview=_preview(user_prompt),
                response_preview=_preview(response),
                prompt_chars=prompt_chars,
                response_chars=response_chars,
                approximate_prompt_tokens=max(1, prompt_chars // 4),
                approximate_response_tokens=max(1, response_chars // 4),
                duration_seconds=duration_seconds,
            )
        )

    def to_markdown(self) -> str:
        total_prompt = sum(item.approximate_prompt_tokens for item in self.interactions)
        total_response = sum(item.approximate_response_tokens for item in self.interactions)
        lines = [
            "# LLM Interactions",
            "",
            f"- calls: {len(self.interactions)}",
            f"- approximate_prompt_tokens: {total_prompt}",
            f"- approximate_response_tokens: {total_response}",
            "",
        ]
        for item in self.interactions:
            lines.append(f"## Call {item.index}")
            lines.append(f"- provider: {item.provider}")
            lines.append(f"- model: {item.model_name}")
            lines.append(f"- duration_seconds: {item.duration_seconds:.4f}")
            lines.append(f"- prompt_tokens_est: {item.approximate_prompt_tokens}")
            lines.append(f"- response_tokens_est: {item.approximate_response_tokens}")
            lines.append(f"- system_preview: {item.system_prompt_preview}")
            lines.append(f"- user_preview: {item.user_prompt_preview}")
            lines.append(f"- response_preview: {item.response_preview}")
            lines.append("")
        lines.append("## Note")
        lines.append("- Token counts are rough local estimates, not provider billing records.")
        return "\n".join(lines) + "\n"


def write_llm_tracker(tracker: LLMInteractionTracker, output_dir: str | Path) -> None:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    (root / "llm_interactions.json").write_text(json.dumps(tracker.model_dump(), indent=2), encoding="utf-8")
    (root / "llm_interactions.md").write_text(tracker.to_markdown(), encoding="utf-8")


def _preview(text: str, limit: int = 240) -> str:
    compact = " ".join(text.split())
    return compact[:limit] + ("..." if len(compact) > limit else "")


def now() -> float:
    return time.perf_counter()
