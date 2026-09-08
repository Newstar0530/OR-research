from src.llm_client import LLMClient
from src.utils.llm_tracker import LLMInteractionTracker


def test_extract_json_from_markdown_fence() -> None:
    parsed = LLMClient._extract_json('```json\n{"ideas": []}\n```')
    assert parsed == {"ideas": []}


def test_ollama_provider_uses_local_chat_endpoint(monkeypatch) -> None:
    captured = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return b'{"message": {"content": "local response"}}'

    def fake_urlopen(req, timeout):
        captured["url"] = req.full_url
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr("src.llm_client.request.urlopen", fake_urlopen)
    response = LLMClient(provider="ollama", model_name="llama3.1:8b", use_mock=False).chat("system", "user")
    assert response == "local response"
    assert captured["url"] == "http://localhost:11434/api/chat"


def test_llm_client_records_interaction() -> None:
    tracker = LLMInteractionTracker()
    response = LLMClient(use_mock=True, tracker=tracker).chat("system", "user")

    assert response
    assert len(tracker.interactions) == 1
