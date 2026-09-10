"""What happens when the model call does not work.

Three behaviours are under test, and each replaces something that used to make
the system look better than it was behaving.

A failed call is recorded. `tracker.record` ran only on success, so a run where
six of eight calls failed produced an artifact showing two calls.

A transient failure is retried. A single 429 used to kill a stage, which then
degraded to `not_generated` -- indistinguishable in the report from a stage that
had no model at all.

A permanent failure stops the run. A mistyped key used to degrade every stage in
turn and finish with a report saying no stage had used a language model.
"""

import io
import json
from pathlib import Path
from urllib.error import HTTPError, URLError

import pytest

from src.llm_client import LLMClient
from src.llm_errors import PERMANENT_KINDS, LLMCallError, classify_status
from src.utils.llm_tracker import LLMInteractionTracker


def _http_error(status: int, body: str = "", retry_after: str | None = None) -> HTTPError:
    """A real HTTPError, with a readable body -- which is the part that matters.

    The client is supposed to read the body and pull the provider's message out
    of it, so the fixture has to supply one.
    """

    headers = {"Retry-After": retry_after} if retry_after else {}
    return HTTPError(
        "https://api.example/v1/chat/completions",
        status,
        "err",
        headers,
        io.BytesIO(body.encode("utf-8")),
    )


class _Response:
    def __init__(self, payload: dict) -> None:
        self._body = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *args) -> None:
        return None


class FakeTransport:
    """Stands in for `urllib.request.urlopen`, one level below the client's own
    error handling, so the classification under test actually runs."""

    def __init__(self, *outcomes) -> None:
        self.outcomes = list(outcomes)
        self.calls = 0
        self.requests: list = []

    def __call__(self, req, timeout=None):
        outcome = self.outcomes[min(self.calls, len(self.outcomes) - 1)]
        self.calls += 1
        self.requests.append(req)
        if isinstance(outcome, BaseException):
            raise outcome
        return _Response(outcome)


def _reply(text: str) -> dict:
    return {"choices": [{"message": {"content": text}}]}


@pytest.fixture
def transport(monkeypatch):
    """Install a fake transport and hand back a factory for clients using it."""

    installed: dict = {}

    def build(*outcomes, **kwargs):
        fake = FakeTransport(*outcomes)
        monkeypatch.setattr("src.llm_client.request.urlopen", fake)
        tracker = LLMInteractionTracker()
        slept: list[float] = []
        client = LLMClient(
            provider="openai",
            model_name="test-model",
            use_mock=False,
            api_key="sk-test",
            tracker=tracker,
            backoff_seconds=kwargs.pop("backoff_seconds", 0.0),
            _sleep=slept.append,
            **kwargs,
        )
        installed.update({"client": client, "fake": fake, "tracker": tracker, "slept": slept})
        return client, fake, tracker, slept

    return build


# -- classification --------------------------------------------------------


@pytest.mark.parametrize(
    "status,kind",
    [(401, "auth"), (403, "auth"), (404, "not_found"), (429, "rate_limit"),
     (400, "bad_request"), (422, "bad_request"), (500, "server"), (503, "server")],
)
def test_a_status_is_classified_by_what_the_caller_should_do(status: int, kind: str) -> None:
    assert classify_status(status) == kind


def test_only_configuration_errors_are_permanent() -> None:
    """A rate limit clears on its own. A wrong model name never does."""

    assert PERMANENT_KINDS == {"auth", "not_found", "bad_request", "budget"}
    assert LLMCallError("auth", "x").is_permanent
    assert LLMCallError("budget", "x").is_permanent
    assert not LLMCallError("rate_limit", "x").is_permanent
    assert not LLMCallError("server", "x").is_permanent
    assert not LLMCallError("timeout", "x").is_permanent


def test_a_permanent_error_says_what_to_change() -> None:
    """"It failed" is useless; "check the key in your environment" is not."""

    assert "API key" in LLMCallError("auth", "x").remedy
    assert "model list" in LLMCallError("not_found", "x").remedy
    assert "context window" in LLMCallError("bad_request", "x").remedy
    assert LLMCallError("rate_limit", "x").remedy == ""


# -- retry -----------------------------------------------------------------


def test_a_rate_limit_is_retried_and_the_call_still_succeeds(transport) -> None:
    client, fake, tracker, _ = transport(_http_error(429), _http_error(429), _reply("done"))

    assert client.chat("sys", "user", stage="gap") == "done"
    assert fake.calls == 3
    assert tracker.successful_calls == 1
    assert len(tracker.failures) == 2, "the two failed attempts must be recorded, not hidden"
    assert tracker.degraded_stages() == [], "the stage succeeded in the end"


def test_giving_up_reports_how_many_attempts_were_spent(transport) -> None:
    client, fake, tracker, _ = transport(_http_error(503), max_attempts=3)

    with pytest.raises(LLMCallError) as caught:
        client.chat("sys", "user", stage="review")

    assert caught.value.kind == "server"
    assert "3 attempt(s)" in str(caught.value)
    assert fake.calls == 3
    assert tracker.degraded_stages() == ["review"]


def test_a_permanent_error_is_not_retried(transport) -> None:
    """Retrying a bad key three times only makes the failure slower."""

    client, fake, _, _ = transport(_http_error(401, json.dumps({"error": {"message": "Invalid API key"}})))

    with pytest.raises(LLMCallError) as caught:
        client.chat("sys", "user")

    assert fake.calls == 1
    assert caught.value.is_permanent
    assert "Invalid API key" in str(caught.value)


def test_the_provider_error_body_is_read_rather_than_discarded(transport) -> None:
    """The OpenAI path used to let HTTPError escape unread, losing the reason."""

    body = json.dumps({"error": {"message": "This model does not exist"}})
    client, _, _, _ = transport(_http_error(404, body))

    with pytest.raises(LLMCallError) as caught:
        client.chat("sys", "user")
    assert "This model does not exist" in str(caught.value)
    assert caught.value.status == 404


def test_a_retry_after_header_is_obeyed_instead_of_the_backoff(transport) -> None:
    client, _, _, slept = transport(
        _http_error(429, "", retry_after="7"), _reply("ok"), backoff_seconds=99.0
    )
    client.chat("sys", "user")
    assert slept == [7.0]


def test_backoff_grows_between_attempts(transport) -> None:
    client, _, _, slept = transport(
        _http_error(503), _http_error(503), _reply("ok"), backoff_seconds=1.0
    )
    client.chat("sys", "user")
    assert len(slept) == 2
    assert slept[1] > slept[0]


@pytest.mark.parametrize(
    "failure,kind",
    [
        (TimeoutError("slow"), "timeout"),
        (URLError("Name or service not known"), "network"),
    ],
)
def test_transport_failures_are_classified_too(transport, failure, kind: str) -> None:
    client, _, tracker, _ = transport(failure, _reply("ok"))
    assert client.chat("sys", "user") == "ok"
    assert tracker.failures[0].error_kind == kind


def test_a_two_hundred_with_no_content_is_a_failure_not_an_empty_answer(transport) -> None:
    client, _, tracker, _ = transport({"choices": []}, _reply("ok"))
    assert client.chat("sys", "user") == "ok"
    assert tracker.failures[0].error_kind == "empty"


# -- budget ----------------------------------------------------------------


def test_a_call_cap_stops_a_runaway_loop(transport) -> None:
    client, fake, _, _ = transport(_reply("ok"), max_calls=2)

    assert client.chat("s", "u") == "ok"
    assert client.chat("s", "u") == "ok"
    with pytest.raises(LLMCallError) as caught:
        client.chat("s", "u")

    assert caught.value.kind == "budget"
    assert caught.value.is_permanent, "a run should not keep trying past its own cap"
    assert fake.calls == 2


def test_a_cost_cap_uses_the_prices_the_caller_supplied(transport) -> None:
    client, _, tracker, _ = transport(
        _reply("x" * 4000),
        max_cost_usd=0.000001,
        price_per_1k_prompt_tokens=1.0,
        price_per_1k_response_tokens=1.0,
    )
    client.chat("s", "u")
    with pytest.raises(LLMCallError) as caught:
        client.chat("s", "u")
    assert caught.value.kind == "budget"
    assert tracker.estimated_cost_usd > 0


def test_without_prices_the_spend_is_unknown_rather_than_zero(transport) -> None:
    """A zero would read as free. It means nobody said what it costs."""

    client, _, tracker, _ = transport(_reply("ok"))
    client.chat("s", "u")

    assert tracker.estimated_cost_usd is None
    assert "unknown rather than zero" in tracker.headline()


# -- json ------------------------------------------------------------------


def test_a_non_json_reply_is_retried_once_with_a_blunter_instruction(transport) -> None:
    client, fake, _, _ = transport(_reply("Sure! Here is the answer."), _reply('{"gaps": ["a"]}'))

    assert client.chat_json("sys", "user") == {"gaps": ["a"]}
    assert fake.calls == 2


def test_the_second_json_attempt_says_the_first_was_not_json() -> None:
    prompts: list[str] = []

    class Recording(LLMClient):
        def chat(self, system_prompt: str, user_prompt: str, stage: str = "") -> str:
            prompts.append(system_prompt)
            return "not json" if len(prompts) == 1 else '{"ok": true}'

    client = Recording(provider="openai", use_mock=False)
    assert client.chat_json("sys", "user") == {"ok": True}
    assert "was not valid JSON" in prompts[1]


def test_json_that_never_arrives_is_reported_as_malformed(transport) -> None:
    client, _, _, _ = transport(_reply("still prose"))
    with pytest.raises(LLMCallError) as caught:
        client.chat_json("sys", "user")
    assert caught.value.kind == "malformed_json"
    assert "still prose" in caught.value.detail


def test_json_embedded_in_prose_is_still_extracted(transport) -> None:
    client, fake, _, _ = transport(_reply('Here you go: {"gaps": ["a"]} -- hope that helps'))
    assert client.chat_json("sys", "user") == {"gaps": ["a"]}
    assert fake.calls == 1


# -- preflight -------------------------------------------------------------


def test_preflight_finds_a_bad_key_before_the_pipeline_runs(transport) -> None:
    client, fake, _, _ = transport(_http_error(401, json.dumps({"error": {"message": "bad key"}})))
    error = client.preflight()

    assert error is not None and error.is_permanent
    assert fake.calls == 1


def test_preflight_passes_when_the_provider_answers(transport) -> None:
    client, _, _, _ = transport(_reply("OK"))
    assert client.preflight() is None


def test_the_mock_client_needs_no_preflight() -> None:
    assert LLMClient(use_mock=True).preflight() is None


# -- the record ------------------------------------------------------------


def test_the_artifact_reports_failures_it_used_to_omit(transport) -> None:
    client, _, tracker, _ = transport(_http_error(429), _reply("ok"))
    client.chat("sys", "user", stage="gap_synthesis")

    markdown = tracker.to_markdown()
    assert "2 ok, 1 failed" not in markdown  # one ok, one failed attempt
    assert "1 ok, 1 failed" in markdown
    assert "## Failures" in markdown
    assert "rate_limit" in markdown
    assert "stage `gap_synthesis`" in markdown


def test_a_permanent_failure_is_marked_as_configuration_not_weather(transport) -> None:
    client, _, tracker, _ = transport(_http_error(401, "bad key"))
    with pytest.raises(LLMCallError):
        client.chat("sys", "user", stage="idea_generation")

    assert tracker.failures[0].permanent is True
    assert "permanent -- configuration, not weather" in tracker.to_markdown()


def test_the_headline_names_the_stages_that_produced_nothing(transport) -> None:
    client, _, tracker, _ = transport(_http_error(503), max_attempts=2)
    with pytest.raises(LLMCallError):
        client.chat("s", "u", stage="model_critique")

    headline = tracker.headline()
    assert "server x2" in headline
    assert "model_critique" in headline


# -- the run stops rather than degrading eight times -----------------------


def _project(tmp_path: Path) -> Path:
    import shutil

    root = Path.cwd()
    project = tmp_path / "project"
    project.mkdir()
    for name in ("configs", "templates", "domain_profiles"):
        shutil.copytree(root / name, project / name)
    return project


def test_a_bad_key_stops_the_run_instead_of_degrading_every_stage(tmp_path: Path, monkeypatch) -> None:
    """Ten minutes of pipeline is the worst way to learn a key is wrong."""

    from src.config import load_config
    from src.orchestrator import ResearchOrchestrator

    project = _project(tmp_path)
    fake = FakeTransport(_http_error(401, json.dumps({"error": {"message": "Incorrect API key"}})))
    monkeypatch.setattr("src.llm_client.request.urlopen", fake)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-wrong")

    config = load_config(project / "configs" / "default.yaml")
    config.output_dir = str(project / "runs")
    config.use_mock_llm = False
    config.llm_provider = "openai"
    config.model_name = "gpt-4o-mini"

    with pytest.raises(LLMCallError) as caught:
        ResearchOrchestrator(config, project_root=project).run()

    assert caught.value.kind == "auth"
    assert "Incorrect API key" in str(caught.value)
    assert "run was stopped before doing any work" in str(caught.value)
    assert fake.calls == 1, "one probe call, not one per stage"

    preflight = next((project / "runs").glob("*/llm_preflight.md"))
    assert "permanent: True" in preflight.read_text(encoding="utf-8")


def test_preflight_can_be_switched_off_to_run_degraded_on_purpose(tmp_path: Path, monkeypatch) -> None:
    from src.config import load_config
    from src.orchestrator import ResearchOrchestrator

    project = _project(tmp_path)
    monkeypatch.setattr(
        "src.llm_client.request.urlopen", FakeTransport(_http_error(503, "unavailable"))
    )
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    config = load_config(project / "configs" / "default.yaml")
    config.output_dir = str(project / "runs")
    config.use_mock_llm = False
    config.llm_provider = "openai"
    config.llm_preflight = False
    config.llm_max_attempts = 1
    config.enable_autonomous_loop = False

    run_dir = ResearchOrchestrator(config, project_root=project).run()

    interactions = (run_dir / "llm_interactions.md").read_text(encoding="utf-8")
    assert "failed" in interactions
    assert "server" in interactions
    provenance = (run_dir / "artifact_provenance.md").read_text(encoding="utf-8")
    assert "gap_synthesis" in provenance


def test_a_mock_run_records_that_no_call_was_made(tmp_path: Path) -> None:
    import json as _json

    from src.config import load_config
    from src.orchestrator import ResearchOrchestrator

    project = _project(tmp_path)
    config = load_config(project / "configs" / "default.yaml")
    config.output_dir = str(project / "runs")
    run_dir = ResearchOrchestrator(config, project_root=project).run()

    entries = _json.loads((run_dir / "artifact_provenance.json").read_text(encoding="utf-8"))["entries"]
    llm_entry = next(entry for entry in entries if entry["stage"] == "llm_calls")
    # In mock mode every agent skips the model entirely, so the honest count is
    # zero rather than "0 failed out of 0".
    assert llm_entry["detail"] == "No model call was made in this run."
    assert not (run_dir / "llm_preflight.md").exists(), "mock mode needs no probe"
