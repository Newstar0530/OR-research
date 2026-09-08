from __future__ import annotations

from pathlib import Path

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover
    yaml = None


PROFILE_DIR = Path(__file__).resolve().parents[1] / "domain_profiles"


def list_profiles() -> list[str]:
    return sorted(p.stem for p in PROFILE_DIR.glob("*.yaml"))


def load_domain_profile(profile_name: str) -> dict:
    path = PROFILE_DIR / f"{profile_name}.yaml"
    if not path.exists():
        path = PROFILE_DIR / "optimization.yaml"
    text = path.read_text(encoding="utf-8")
    if yaml:
        return yaml.safe_load(text) or {}
    return {"name": profile_name, "raw": text}


def render_profile_markdown(profile: dict) -> str:
    lines = [f"# Domain Profile: {profile.get('name', 'unknown')}", ""]
    for key in [
        "description",
        "typical_models",
        "algorithm_families",
        "benchmarks",
        "metrics",
        "sensitivity_parameters",
        "literature_keywords",
        "human_checks",
    ]:
        value = profile.get(key)
        if not value:
            continue
        lines.append(f"## {key.replace('_', ' ').title()}")
        if isinstance(value, list):
            lines.extend(f"- {item}" for item in value)
        else:
            lines.append(str(value))
        lines.append("")
    return "\n".join(lines)

