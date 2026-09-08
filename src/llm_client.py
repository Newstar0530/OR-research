from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from urllib import request
from urllib.error import HTTPError

from src.utils.llm_tracker import LLMInteractionTracker


@dataclass
class LLMClient:
    provider: str = "mock"
    model_name: str = "mock-ie-or"
    use_mock: bool = True
    base_url: str | None = None
    api_key: str | None = None
    tracker: LLMInteractionTracker | None = None

    def chat(self, system_prompt: str, user_prompt: str) -> str:
        start = time.perf_counter()
        if self.use_mock or self.provider == "mock":
            response = self._mock_chat(system_prompt, user_prompt)
        elif self.provider.lower() in {"ollama", "local-ollama"}:
            response = self._ollama_chat(system_prompt, user_prompt)
        elif self.provider.lower() in {"gemini", "google", "google-gemini"}:
            response = self._gemini_chat(system_prompt, user_prompt)
        else:
            response = self._openai_compatible_chat(system_prompt, user_prompt)
        if self.tracker is not None:
            self.tracker.record(
                self.provider,
                self.model_name,
                system_prompt,
                user_prompt,
                response,
                time.perf_counter() - start,
            )
        return response

    def chat_json(self, system_prompt: str, user_prompt: str) -> dict:
        raw = self.chat(
            system_prompt + "\nReturn only valid JSON. Do not wrap it in markdown.",
            user_prompt,
        )
        return self._extract_json(raw)

    def _mock_chat(self, system_prompt: str, user_prompt: str) -> str:
        text = f"{system_prompt}\n{user_prompt}".lower()
        if "novelty" in text:
            return "Mock novelty mode: no external API configured. Suggested queries are generated; no citations are fabricated."
        if "review" in text:
            return "Mock review: promising but preliminary. Verify modeling assumptions, baselines, and conclusions."
        if "debug" in text:
            return "Mock debug: inspect traceback and apply the smallest local patch."
        return "Mock IE/OR assistant response. Human verification required."

    def _openai_compatible_chat(self, system_prompt: str, user_prompt: str) -> str:
        api_key = self.api_key or os.getenv("OPENAI_API_KEY")
        base_url = (self.base_url or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required when use_mock_llm is false.")
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.2,
        }
        req = request.Request(
            f"{base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        with request.urlopen(req, timeout=60) as response:
            data = json.loads(response.read().decode("utf-8"))
        return data["choices"][0]["message"]["content"]

    def _gemini_chat(self, system_prompt: str, user_prompt: str) -> str:
        api_key = self.api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is required when llm_provider is gemini.")
        model = self.model_name if self.model_name and self.model_name != "mock-ie-or" else "gemini-2.5-flash"
        endpoint = (self.base_url or "https://generativelanguage.googleapis.com/v1beta").rstrip("/")
        payload = {
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
            "generationConfig": {"temperature": 0.2},
        }
        req = request.Request(
            f"{endpoint}/models/{model}:generateContent",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "x-goog-api-key": api_key},
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=90) as response:
                data = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Gemini API error {exc.code}: {body}") from exc
        candidates = data.get("candidates", [])
        if not candidates:
            raise RuntimeError(f"Gemini API returned no candidates: {data}")
        parts = candidates[0].get("content", {}).get("parts", [])
        return "".join(part.get("text", "") for part in parts).strip()

    def _ollama_chat(self, system_prompt: str, user_prompt: str) -> str:
        endpoint = (self.base_url or os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434").rstrip("/")
        model = self.model_name if self.model_name and self.model_name != "mock-ie-or" else os.getenv("OLLAMA_MODEL", "llama3.1:8b")
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "options": {"temperature": 0.2},
        }
        req = request.Request(
            f"{endpoint}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=120) as response:
                data = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Ollama API error {exc.code}: {body}") from exc
        except OSError as exc:
            raise RuntimeError(
                f"Could not connect to Ollama at {endpoint}. Start Ollama and check OLLAMA_BASE_URL."
            ) from exc
        message = data.get("message", {})
        content = message.get("content", "")
        if not content:
            raise RuntimeError(f"Ollama returned no message content: {data}")
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
