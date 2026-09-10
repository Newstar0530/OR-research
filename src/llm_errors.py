"""What went wrong with an LLM call, and whether trying again could help.

Every stage in the front half degrades to `not_generated` when its model call
fails. That is the right behaviour for a rate limit at three in the morning. It
is the wrong behaviour for a mistyped API key, because then all six stages
degrade for the same reason, the run finishes in ten minutes of wasted work,
and the report says "no stage used a language model" -- which is true, and
tells you nothing about why.

So a failure is classified before it is handled. A permanent failure is a
configuration problem: no amount of retrying fixes it, every later call will
fail identically, and the run should stop and say so. A transient failure is
weather: retry with backoff, and degrade honestly if it persists.
"""

from __future__ import annotations

from typing import Literal


ErrorKind = Literal[
    "auth",           # 401/403 -- the key is missing, wrong, or lacks access
    "not_found",      # 404 -- the model name does not exist at this provider
    "bad_request",    # 400/422 -- the request itself is malformed
    "budget",         # this run's own call or cost cap was reached
    "rate_limit",     # 429
    "server",         # 5xx
    "timeout",        # the request did not come back
    "network",        # DNS, refused connection, TLS
    "empty",          # a 200 with no usable content
    "malformed_json", # a 200 whose body was not the JSON that was asked for
]

#: Kinds no retry can fix. Every one of them is something a human must change.
PERMANENT_KINDS: frozenset[str] = frozenset({"auth", "not_found", "bad_request", "budget"})

#: What to tell the user for each permanent kind, since "it failed" is useless.
REMEDIES: dict[str, str] = {
    "auth": (
        "The provider rejected the credentials. Check the API key in your environment "
        "(OPENAI_API_KEY / GEMINI_API_KEY) and that it has access to this model."
    ),
    "not_found": (
        "The provider does not have a model by that name. Check `model_name` in the run config "
        "against the provider's current model list."
    ),
    "bad_request": (
        "The provider rejected the request itself. This is usually a prompt that exceeds the "
        "model's context window, or an unsupported parameter."
    ),
    "budget": (
        "This run reached its own LLM cap. Raise `max_llm_calls` or `max_llm_cost_usd` in the "
        "run config, or accept the degraded stages."
    ),
}


class LLMCallError(RuntimeError):
    """A failed model call, with enough detail to act on."""

    def __init__(
        self,
        kind: ErrorKind,
        message: str,
        status: int | None = None,
        detail: str = "",
        retry_after_seconds: float | None = None,
    ) -> None:
        super().__init__(message)
        self.kind: ErrorKind = kind
        self.status = status
        self.detail = detail[:2000]
        self.retry_after_seconds = retry_after_seconds

    @property
    def is_permanent(self) -> bool:
        return self.kind in PERMANENT_KINDS

    @property
    def remedy(self) -> str:
        return REMEDIES.get(self.kind, "")

    def summary(self) -> str:
        parts = [f"{self.kind}"]
        if self.status is not None:
            parts.append(f"HTTP {self.status}")
        parts.append(str(self))
        text = ": ".join(parts)
        return f"{text} -- {self.remedy}" if self.remedy else text


def classify_status(status: int) -> ErrorKind:
    """Map an HTTP status onto what a caller should do about it."""

    if status in (401, 403):
        return "auth"
    if status == 404:
        return "not_found"
    if status == 429:
        return "rate_limit"
    if 400 <= status < 500:
        return "bad_request"
    return "server"
