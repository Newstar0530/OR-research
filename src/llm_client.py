"""One place where every model call is made, retried, priced and recorded.

Three things were missing, and all three made the system look better than it
was behaving.

A failed call was never recorded. `tracker.record` ran only on success, so a run
where six of eight calls failed produced an `llm_interactions.json` showing two
calls. The artifact that exists to say what the models did was silent about
everything that went wrong.

There was no retry. A single 429 killed a stage, which then degraded to
`not_generated` -- indistinguishable in the final report from a stage that had
no model configured at all.

And a permanent failure was treated like a transient one. A mistyped key
degraded all six front-half stages, one after another, and the run finished
with a report saying no stage had used a language model. True, and useless.
Permanent failures now raise, so the run stops on the first one and says what
to change.

Costs are estimated only when the caller supplies prices. There is no built-in
price table, because a stale one printed as a dollar figure is exactly the kind
of confident wrong number this project keeps removing.
"""

from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass, field
from urllib import request
from urllib.error import HTTPError, URLError

from src.llm_errors import LLMCallError, classify_status
from src.utils.llm_tracker import LLMInteractionTracker


@dataclass
class LLMClient:
    provider: str = "mock"
    model_name: str = "mock-ie-or"
    use_mock: bool = True
    base_url: str | None = None
    api_key: str | None = None
    tracker: LLMInteractionTracker | None = None
    #: Attempts per call, including the first. Only transient failures retry.
    max_attempts: int = 3
    #: Seconds before the first retry; doubled each time, with jitter.
    backoff_seconds: float = 2.0
    timeout_seconds: int = 90
    temperature: float = 0.2
    #: Hard caps for one run. A search loop can otherwise grow its own bill.
    max_calls: int | None = None
    max_cost_usd: float | None = None
    #: Supplied by the caller, because a built-in price table goes stale and a
    #: stale price printed as a dollar figure is a confident wrong number.
    price_per_1k_prompt_tokens: float | None = None
    price_per_1k_response_tokens: float | None = None
    _sleep: object = field(default=None, repr=False)

    # -- the call ----------------------------------------------------------

    def chat(self, system_prompt: str, user_prompt: str, stage: str = "") -> str:
        """One logical call. Retries transient failures; raises permanent ones."""

        if self.use_mock or self.provider == "mock":
            start = time.perf_counter()
            response = self._mock_chat(system_prompt, user_prompt)
            self._record(stage, system_prompt, user_prompt, response, time.perf_counter() - start, 1)
            return response

        self._check_budget(stage, system_prompt, user_prompt)
        last: LLMCallError | None = None
        for attempt in range(1, self.max_attempts + 1):
            start = time.perf_counter()
            try:
                response = self._dispatch(system_prompt, user_prompt)
            except LLMCallError as error:
                self._record_failure(
                    stage, system_prompt, user_prompt, time.perf_counter() - start, attempt, error
                )
                if error.is_permanent:
                    # Retrying cannot help and every later call fails the same
                    # way. Stopping here is the only useful thing to do.
                    raise
                last = error
                if attempt < self.max_attempts:
                    self._wait(error, attempt)
                continue
            self._record(
                stage, system_prompt, user_prompt, response, time.perf_counter() - start, attempt
            )
            return response

        assert last is not None
        raise LLMCallError(
            last.kind,
            f"{last} (gave up after {self.max_attempts} attempt(s))",
            status=last.status,
            detail=last.detail,
        )

    def _dispatch(self, system_prompt: str, user_prompt: str) -> str:
        provider = self.provider.lower()
        if provider in {"ollama", "local-ollama"}:
            return self._ollama_chat(system_prompt, user_prompt)
        if provider in {"gemini", "google", "google-gemini"}:
            return self._gemini_chat(system_prompt, user_prompt)
        return self._openai_compatible_chat(system_prompt, user_prompt)

    def _wait(self, error: LLMCallError, attempt: int) -> None:
        """Exponential backoff, but obey the provider when it names a delay."""

        delay = error.retry_after_seconds
        if delay is None:
            delay = self.backoff_seconds * (2 ** (attempt - 1))
            delay += random.uniform(0, delay * 0.25)
        sleeper = self._sleep or time.sleep
        sleeper(min(float(delay), 60.0))

    def chat_json(self, system_prompt: str, user_prompt: str, stage: str = "") -> dict:
        """A JSON call. One extra attempt when the body is not JSON.

        Malformed JSON is a different failure from a network error: the model
        answered, it just answered in prose. Asking again with a blunter
        instruction fixes it often enough to be worth one attempt.
        """

        instruction = "\nReturn only valid JSON. Do not wrap it in markdown."
        raw = self.chat(system_prompt + instruction, user_prompt, stage=stage)
        try:
            return self._extract_json(raw)
        except (json.JSONDecodeError, ValueError):
            if self.use_mock:
                raise LLMCallError(
                    "malformed_json",
                    "The mock client does not return JSON.",
                    detail=raw[:500],
                )
            retry = self.chat(
                system_prompt
                + instruction
                + " Your previous reply was not valid JSON. Return a single JSON object and nothing else.",
                user_prompt,
                stage=stage,
            )
            try:
                return self._extract_json(retry)
            except (json.JSONDecodeError, ValueError) as exc:
                raise LLMCallError(
                    "malformed_json",
                    "The model did not return JSON after two attempts.",
                    detail=retry[:500],
                ) from exc

    # -- preflight ---------------------------------------------------------

    def preflight(self) -> LLMCallError | None:
        """One cheap call to find configuration errors before the run starts.

        Ten minutes of pipeline that degrades every stage for the same reason is
        the worst way to learn that a key is wrong.
        """

        if self.use_mock or self.provider == "mock":
            return None
        try:
            self.chat(
                "You are a configuration probe. Reply with the single word OK.",
                "Reply with OK.",
                stage="preflight",
            )
        except LLMCallError as error:
            return error
        return None

    # -- budget and recording ---------------------------------------------

    def _check_budget(self, stage: str, system_prompt: str, user_prompt: str) -> None:
        if self.tracker is None:
            return
        if self.max_calls is not None and self.tracker.successful_calls >= self.max_calls:
            raise LLMCallError(
                "budget",
                f"This run's cap of {self.max_calls} model call(s) has been reached.",
            )
        if self.max_cost_usd is not None:
            spent = self.tracker.estimated_cost_usd
            if spent is not None and spent >= self.max_cost_usd:
                raise LLMCallError(
                    "budget",
                    f"This run's estimated spend (${spent:.4f}) has reached its cap of "
                    f"${self.max_cost_usd:.4f}.",
                )

    def _record(
        self,
        stage: str,
        system_prompt: str,
        user_prompt: str,
        response: str,
        duration: float,
        attempt: int,
    ) -> None:
        if self.tracker is None:
            return
        self.tracker.record(
            self.provider,
            self.model_name,
            system_prompt,
            user_prompt,
            response,
            duration,
            stage=stage,
            attempt=attempt,
            price_per_1k_prompt_tokens=self.price_per_1k_prompt_tokens,
            price_per_1k_response_tokens=self.price_per_1k_response_tokens,
        )

    def _record_failure(
        self,
        stage: str,
        system_prompt: str,
        user_prompt: str,
        duration: float,
        attempt: int,
        error: LLMCallError,
    ) -> None:
        if self.tracker is None:
            return
        self.tracker.record_failure(
            self.provider,
            self.model_name,
            system_prompt,
            user_prompt,
            duration,
            stage=stage,
            attempt=attempt,
            error_kind=error.kind,
            error_detail=error.summary(),
            permanent=error.is_permanent,
        )

    def _mock_chat(self, system_prompt: str, user_prompt: str) -> str:
        text = f"{system_prompt}\n{user_prompt}".lower()
        if "novelty" in text:
            return "Mock novelty mode: no external API configured. Suggested queries are generated; no citations are fabricated."
        if "review" in text:
            return "Mock review: promising but preliminary. Verify modeling assumptions, baselines, and conclusions."
        if "debug" in text:
            return "Mock debug: inspect traceback and apply the smallest local patch."
        return "Mock IE/OR assistant response. Human verification required."


    def _post_json(self, url: str, payload: dict, headers: dict, what: str) -> dict:
        """POST and classify every way it can fail.

        The OpenAI path used to let `HTTPError` escape unread, so a 401 surfaced
        as `HTTP Error 401: Unauthorized` and the body -- which says *which*
        thing is wrong -- was thrown away.
        """

        req = request.Request(
            url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST"
        )
        try:
            with request.urlopen(req, timeout=self.timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            body = ""
            try:
                body = exc.read().decode("utf-8", errors="replace")
            except Exception:  # pragma: no cover - the body is best-effort
                body = ""
            retry_after = exc.headers.get("Retry-After") if exc.headers else None
            try:
                delay = float(retry_after) if retry_after else None
            except (TypeError, ValueError):
                delay = None
            raise LLMCallError(
                classify_status(exc.code),
                f"{what} rejected the request: {_first_line(body) or exc.reason}",
                status=exc.code,
                detail=body,
                retry_after_seconds=delay,
            ) from exc
        except TimeoutError as exc:
            raise LLMCallError(
                "timeout", f"{what} did not respond within {self.timeout_seconds}s."
            ) from exc
        except URLError as exc:
            reason = getattr(exc, "reason", exc)
            if isinstance(reason, TimeoutError):
                raise LLMCallError(
                    "timeout", f"{what} did not respond within {self.timeout_seconds}s."
                ) from exc
            raise LLMCallError("network", f"Could not reach {what}: {reason}") from exc
        except json.JSONDecodeError as exc:
            raise LLMCallError("malformed_json", f"{what} returned a body that is not JSON.") from exc

    def _openai_compatible_chat(self, system_prompt: str, user_prompt: str) -> str:
        api_key = self.api_key or os.getenv("OPENAI_API_KEY")
        base_url = (self.base_url or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        if not api_key:
            raise LLMCallError(
                "auth", "OPENAI_API_KEY is required when use_mock_llm is false."
            )
        data = self._post_json(
            f"{base_url}/chat/completions",
            {
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": self.temperature,
            },
            {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            "The OpenAI-compatible API",
        )
        choices = data.get("choices") or []
        content = (choices[0].get("message", {}).get("content") if choices else "") or ""
        if not content.strip():
            raise LLMCallError(
                "empty", "The API returned no message content.", detail=json.dumps(data)[:1000]
            )
        return content

    def _gemini_chat(self, system_prompt: str, user_prompt: str) -> str:
        api_key = self.api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise LLMCallError(
                "auth", "GEMINI_API_KEY is required when llm_provider is gemini."
            )
        model = self.model_name if self.model_name and self.model_name != "mock-ie-or" else "gemini-2.5-flash"
        endpoint = (self.base_url or "https://generativelanguage.googleapis.com/v1beta").rstrip("/")
        data = self._post_json(
            f"{endpoint}/models/{model}:generateContent",
            {
                "systemInstruction": {"parts": [{"text": system_prompt}]},
                "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
                "generationConfig": {"temperature": self.temperature},
            },
            {"Content-Type": "application/json", "x-goog-api-key": api_key},
            "The Gemini API",
        )
        candidates = data.get("candidates", [])
        if not candidates:
            # A safety block also lands here, and it is not a transient fault.
            reason = data.get("promptFeedback", {}).get("blockReason", "")
            raise LLMCallError(
                "empty",
                f"Gemini returned no candidates{f' (blockReason: {reason})' if reason else ''}.",
                detail=json.dumps(data)[:1000],
            )
        parts = candidates[0].get("content", {}).get("parts", [])
        text = "".join(part.get("text", "") for part in parts).strip()
        if not text:
            raise LLMCallError("empty", "Gemini returned a candidate with no text.")
        return text

    def _ollama_chat(self, system_prompt: str, user_prompt: str) -> str:
        endpoint = (self.base_url or os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434").rstrip("/")
        model = self.model_name if self.model_name and self.model_name != "mock-ie-or" else os.getenv("OLLAMA_MODEL", "llama3.1:8b")
        data = self._post_json(
            f"{endpoint}/api/chat",
            {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "stream": False,
                "options": {"temperature": self.temperature},
            },
            {"Content-Type": "application/json"},
            f"Ollama at {endpoint}",
        )
        content = (data.get("message", {}) or {}).get("content", "")
        if not content.strip():
            raise LLMCallError(
                "empty", "Ollama returned no message content.", detail=json.dumps(data)[:1000]
            )
        return content.strip()

    @staticmethod
    def _extract_json(text: str) -> dict:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            cleaned = cleaned.removeprefix("json").strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            start = cleaned.find("{")
            end = cleaned.rfind("}")
            if start >= 0 and end > start:
                return json.loads(cleaned[start : end + 1])
            raise


def _first_line(text: str, limit: int = 300) -> str:
    """Providers return a JSON error body; its message is the useful part."""

    body = (text or "").strip()
    if not body:
        return ""
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        return " ".join(body.split())[:limit]
    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict) and error.get("message"):
            return str(error["message"])[:limit]
        if isinstance(error, str):
            return error[:limit]
        if payload.get("message"):
            return str(payload["message"])[:limit]
    return " ".join(body.split())[:limit]
