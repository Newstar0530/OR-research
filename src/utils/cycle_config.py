from __future__ import annotations

import json
from pathlib import Path

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover
    yaml = None

from src.config import _simple_yaml


def autorun_safety_decision(run_dir: str | Path) -> tuple[bool, list[str]]:
    root = Path(run_dir)
    plan_path = root / "next_research_cycle_plan.json"
    if not plan_path.exists():
        return False, [f"Missing next_research_cycle_plan.json in {root}"]
    try:
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return False, [f"Could not parse next_research_cycle_plan.json: {exc}"]

    reasons: list[str] = []
    if not bool(plan.get("can_autorun_without_human", False)):
        reasons.append("Plan field can_autorun_without_human is false.")
    for action in plan.get("actions", []):
        if not isinstance(action, dict):
            continue
        priority = int(action.get("priority", 5))
        human_required = bool(action.get("human_required", False))
        if priority <= 1 and human_required:
            action_type = action.get("action_type", "unknown_action")
            issue = action.get("issue", "No issue text provided.")
            reasons.append(f"P{priority} human-required action `{action_type}`: {issue}")
    return not reasons, reasons


def build_next_cycle_config(run_dir: str | Path, output_path: str | Path | None = None) -> Path:
    root = Path(run_dir)
    if not root.exists():
        raise FileNotFoundError(f"Run directory not found: {root}")
    base_path = root / "config_used.yaml"
    patch_path = root / "next_config_patch.yaml"
    if not base_path.exists():
        raise FileNotFoundError(f"Missing config_used.yaml in {root}")
    if not patch_path.exists():
        raise FileNotFoundError(f"Missing next_config_patch.yaml in {root}")

    base = _load_yaml_mapping(base_path)
    patch = _load_yaml_mapping(patch_path)
    if patch:
        base.update(patch)
    base["project_name"] = _next_project_name(str(base.get("project_name", "research_cycle")))
    base["previous_run_dir"] = str(root)
    target = Path(output_path) if output_path else root / "next_cycle_config.yaml"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(_dump_yaml(base), encoding="utf-8")
    return target


def _load_yaml_mapping(path: Path) -> dict[str, object]:
    text = path.read_text(encoding="utf-8")
    if not text.strip() or text.lstrip().startswith("# No automatic config patch"):
        return {}
    data = yaml.safe_load(text) if yaml else _simple_yaml(text)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return dict(data)


def _dump_yaml(data: dict[str, object]) -> str:
    if yaml:
        return yaml.safe_dump(data, sort_keys=False, allow_unicode=True)
    lines = []
    for key, value in data.items():
        if isinstance(value, bool):
            rendered = "true" if value else "false"
        elif value is None:
            rendered = ""
        elif isinstance(value, str):
            rendered = f'"{value}"'
        else:
            rendered = str(value)
        lines.append(f"{key}: {rendered}")
    return "\n".join(lines) + "\n"


def _next_project_name(project_name: str) -> str:
    marker = "_cycle"
    if marker not in project_name:
        return f"{project_name}_cycle2"
    prefix, suffix = project_name.rsplit(marker, 1)
    if suffix.isdigit():
        return f"{prefix}{marker}{int(suffix) + 1}"
    return f"{project_name}_cycle2"
