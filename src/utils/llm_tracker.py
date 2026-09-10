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
    #: `ok` or `failed`. Failures used to be absent entirely, so a run where
    #: six of eight calls failed reported two calls.
    status: str = "ok"
    #: Which pipeline stage asked, when the caller said.
    stage: str = ""
    #: 1 for a first attempt; higher after a retry.
    attempt: int = 1
    error_kind: str = ""
    error_detail: str = ""
    permanent: bool = False
    #: None when no prices were configured. A stale built-in price table
    #: printed as a dollar figure would be a confident wrong number.
    estimated_cost_usd: float | None = None


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
        stage: str = "",
        attempt: int = 1,
        price_per_1k_prompt_tokens: float | None = None,
        price_per_1k_response_tokens: float | None = None,
    ) -> None:
        prompt_chars = len(system_prompt) + len(user_prompt)
        response_chars = len(response)
        prompt_tokens = max(1, prompt_chars // 4)
        response_tokens = max(1, response_chars // 4)
        cost: float | None = None
        if price_per_1k_prompt_tokens is not None or price_per_1k_response_tokens is not None:
            cost = (prompt_tokens / 1000.0) * (price_per_1k_prompt_tokens or 0.0) + (
                response_tokens / 1000.0
            ) * (price_per_1k_response_tokens or 0.0)
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
                approximate_prompt_tokens=prompt_tokens,
                approximate_response_tokens=response_tokens,
                duration_seconds=duration_seconds,
                status="ok",
                stage=stage,
                attempt=attempt,
                estimated_cost_usd=cost,
            )
        )

    def record_failure(
        self,
        provider: str,
        model_name: str,
        system_prompt: str,
        user_prompt: str,
        duration_seconds: float,
        stage: str = "",
        attempt: int = 1,
        error_kind: str = "",
        error_detail: str = "",
        permanent: bool = False,
    ) -> None:
        """A call that did not return. The artifact has to show these too."""

        prompt_chars = len(system_prompt) + len(user_prompt)
        self.interactions.append(
            LLMInteraction(
                index=len(self.interactions),
                provider=provider,
                model_name=model_name,
                system_prompt_preview=_preview(system_prompt),
                user_prompt_preview=_preview(user_prompt),
                response_preview="",
                prompt_chars=prompt_chars,
                response_chars=0,
                approximate_prompt_tokens=max(1, prompt_chars // 4),
                approximate_response_tokens=0,
                duration_seconds=duration_seconds,
                status="failed",
                stage=stage,
                attempt=attempt,
                error_kind=error_kind,
                error_detail=_preview(error_detail, 400),
                permanent=permanent,
            )
        )

    # -- what actually happened -------------------------------------------

    @property
    def successful(self) -> list[LLMInteraction]:
        return [item for item in self.interactions if item.status == "ok"]

    @property
    def failures(self) -> list[LLMInteraction]:
        return [item for item in self.interactions if item.status == "failed"]

    @property
    def successful_calls(self) -> int:
        return len(self.successful)

    @property
    def estimated_cost_usd(self) -> float | None:
        """None when nothing was priced, so a zero is never mistaken for free."""

        priced = [item.estimated_cost_usd for item in self.successful if item.estimated_cost_usd is not None]
        return sum(priced) if priced else None

    def failure_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for item in self.failures:
            counts[item.error_kind or "unknown"] = counts.get(item.error_kind or "unknown", 0) + 1
        return counts

    def degraded_stages(self) -> list[str]:
        """Stages whose every attempt failed, so they produced nothing."""

        succeeded = {item.stage for item in self.successful if item.stage}
        failed = {item.stage for item in self.failures if item.stage}
        return sorted(failed - succeeded)

    def headline(self) -> str:
        """One sentence a reader can trust before reading the artifacts."""

        if not self.interactions:
            return "No model call was made in this run."
        ok, failed = len(self.successful), len(self.failures)
        cost = self.estimated_cost_usd
        spend = (
            f" Estimated spend ${cost:.4g} from local token counts."
            if cost is not None
            else " No prices were configured, so the spend is unknown rather than zero."
        )
        if not failed:
            return f"{ok} model call(s), none failed.{spend}"
        kinds = ", ".join(f"{kind} x{count}" for kind, count in sorted(self.failure_counts().items()))
        degraded = self.degraded_stages()
        tail = (
            f" The following stage(s) produced nothing as a result: {', '.join(degraded)}."
            if degraded
            else ""
        )
        return (
            f"{ok} model call(s) succeeded and {failed} failed ({kinds}).{spend}{tail}"
        )

    def to_markdown(self) -> str:
        total_prompt = sum(item.approximate_prompt_tokens for item in self.interactions)
        total_response = sum(item.approximate_response_tokens for item in self.interactions)
        lines = [
            "# LLM Interactions",
            "",
            self.headline(),
            "",
            f"- attempts recorded: {len(self.interactions)} ({len(self.successful)} ok, {len(self.failures)} failed)",
            f"- approximate_prompt_tokens: {total_prompt}",
            f"- approximate_response_tokens: {total_response}",
            "",
        ]
        if self.failures:
            lines.append("## Failures")
            lines.append("")
            for item in self.failures:
                marker = " **(permanent -- configuration, not weather)**" if item.permanent else ""
                lines.append(
                    f"- call {item.index}, stage `{item.stage or 'unnamed'}`, attempt {item.attempt}: "
                    f"`{item.error_kind}`{marker} -- {item.error_detail}"
                )
            lines.append("")
        for item in self.interactions:
            lines.append(f"## Call {item.index} ({item.status})")
            lines.append(f"- provider: {item.provider}")
            lines.append(f"- model: {item.model_name}")
            lines.append(f"- stage: {item.stage or 'unnamed'}")
            lines.append(f"- attempt: {item.attempt}")
            if item.status == "failed":
                lines.append(f"- error: `{item.error_kind}` {item.error_detail}")
            if item.estimated_cost_usd is not None:
                lines.append(f"- estimated_cost_usd: {item.estimated_cost_usd:.6f}")
            lines.append(f"- duration_seconds: {item.duration_seconds:.4f}")
            lines.append(f"- prompt_tokens_est: {item.approximate_prompt_tokens}")
            lines.append(f"- response_tokens_est: {item.approximate_response_tokens}")
            lines.append(f"- system_preview: {item.system_prompt_preview}")
            lines.append(f"- user_preview: {item.user_prompt_preview}")
            lines.append(f"- response_preview: {item.response_preview}")
            lines.append("")
        lines.append("## Note")
        lines.append("- Token counts are rough local estimates (characters / 4), not provider billing records.")
        lines.append(
            "- A cost figure here is those estimates multiplied by prices you configured. It is "
            "not an invoice, and it is absent rather than zero when no prices were set."
        )
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
