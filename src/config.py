from __future__ import annotations

import ast
from pathlib import Path

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - exercised only in minimal runtimes
    yaml = None
from pydantic import BaseModel, Field


class AppConfig(BaseModel):
    project_name: str
    research_goal: str
    domain: str = "Industrial Engineering / Operations Research"
    research_domain_mode: str = "auto"
    project_plugin: str = "auto"
    template_path: str | None = None
    experiment_template: str | None = None
    output_dir: str = "runs"
    previous_run_dir: str | None = None
    max_ideas: int = Field(default=3, ge=1)
    max_experiments: int = Field(default=3, ge=1)
    max_debug_attempts: int = Field(default=2, ge=0)
    solver_timeout_seconds: int = Field(default=30, ge=1)
    llm_provider: str = "mock"
    model_name: str = "mock-ie-or"
    use_mock_llm: bool = True
    enable_novelty_check: bool = True
    enable_report_generation: bool = True
    literature_dir: str | None = None
    benchmark_dir: str | None = None
    background_file: str | None = None
    enable_semantic_scholar: bool = False
    enable_llm_agent_runtime: bool = True
    enable_autonomous_loop: bool = True
    max_research_iterations: int = Field(default=2, ge=1)
    max_candidate_branches: int = Field(default=3, ge=1)
    autonomous_patience: int = Field(default=2, ge=1)
    min_metric_improvement: float = Field(default=0.0, ge=0.0)
    primary_metric: str = "objective"
    objective_direction: str = Field(default="minimize", pattern="^(minimize|maximize)$")
    primary_metric_method: str | None = None
    baseline_method: str | None = None
    proposed_method: str | None = None
    code_editing_backend: str = Field(default="deterministic", pattern="^(deterministic|aider|auto|disabled)$")
    aider_command: str = "aider"
    aider_model: str | None = None
    aider_timeout_seconds: int = Field(default=120, ge=1)


def load_config(path: str | Path) -> AppConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as f:
        text = f.read()
    raw = yaml.safe_load(text) if yaml else _simple_yaml(text)
    return AppConfig(**raw)


def _simple_yaml(text: str) -> dict[str, object]:
    data: dict[str, object] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        value = value.strip()
        if value.lower() in {"true", "false"}:
            parsed: object = value.lower() == "true"
        elif value in {"", "null", "None"}:
            parsed = None
        else:
            try:
                parsed = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                parsed = value.strip('"').strip("'")
        data[key.strip()] = parsed
    return data
