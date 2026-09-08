from __future__ import annotations

import re
from pathlib import Path

from pydantic import BaseModel, Field


class ModelComponent(BaseModel):
    name: str
    description: str
    source: str = "heuristic_extraction"
    needs_human_verification: bool = True


class OptimizationModelIR(BaseModel):
    problem_name: str
    objective_sense: str = "unknown"
    sets: list[ModelComponent] = Field(default_factory=list)
    parameters: list[ModelComponent] = Field(default_factory=list)
    decision_variables: list[ModelComponent] = Field(default_factory=list)
    objective: str | None = None
    constraints: list[ModelComponent] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    extraction_warnings: list[str] = Field(default_factory=list)

    def to_markdown(self) -> str:
        lines = ["# Structured Optimization Model IR", "", f"- problem_name: {self.problem_name}", f"- objective_sense: {self.objective_sense}", ""]
        lines.append("## Sets")
        lines.extend(_component_lines(self.sets))
        lines.append("")
        lines.append("## Parameters")
        lines.extend(_component_lines(self.parameters))
        lines.append("")
        lines.append("## Decision Variables")
        lines.extend(_component_lines(self.decision_variables))
        lines.append("")
        lines.append("## Objective")
        lines.append(self.objective or "Needs human verification: objective was not extracted.")
        lines.append("")
        lines.append("## Constraints")
        lines.extend(_component_lines(self.constraints))
        lines.append("")
        lines.append("## Assumptions")
        lines.extend(f"- {item}" for item in self.assumptions or ["Needs human verification."])
        lines.append("")
        lines.append("## Limitations")
        lines.extend(f"- {item}" for item in self.limitations or ["This IR is heuristic and must be checked before solver export."])
        lines.append("")
        lines.append("## Extraction Warnings")
        lines.extend(f"- {item}" for item in self.extraction_warnings or ["None."])
        return "\n".join(lines) + "\n"


def build_model_ir(model_markdown: str, problem_name: str = "research_model") -> OptimizationModelIR:
    sections = _sections(model_markdown)
    ir = OptimizationModelIR(
        problem_name=problem_name,
        objective_sense=_objective_sense(model_markdown),
        sets=_extract_components(sections, ["sets and indices", "sets", "indices"], "set"),
        parameters=_extract_components(sections, ["parameters"], "parameter"),
        decision_variables=_extract_components(sections, ["decision variables", "variables"], "variable"),
        objective=_extract_objective(sections),
        constraints=_extract_components(sections, ["constraints"], "constraint"),
        assumptions=_extract_bullets(sections, ["assumptions"]),
        limitations=_extract_bullets(sections, ["limitations"]),
    )
    if not ir.decision_variables:
        ir.extraction_warnings.append("No decision variables were extracted.")
    if not ir.constraints:
        ir.extraction_warnings.append("No constraints were extracted.")
    if not ir.objective:
        ir.extraction_warnings.append("No objective was extracted.")
    return ir


def write_model_ir(model_markdown: str, output_dir: str | Path, problem_name: str = "research_model") -> OptimizationModelIR:
    ir = build_model_ir(model_markdown, problem_name)
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    (root / "model_ir.json").write_text(ir.model_dump_json(indent=2), encoding="utf-8")
    (root / "model_ir.md").write_text(ir.to_markdown(), encoding="utf-8")
    return ir


def _sections(markdown: str) -> dict[str, str]:
    matches = list(re.finditer(r"^#{1,3}\s+(.+?)\s*$", markdown, flags=re.MULTILINE))
    sections: dict[str, str] = {}
    for index, match in enumerate(matches):
        title = match.group(1).strip().lower()
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(markdown)
        sections[title] = markdown[start:end].strip()
    return sections


def _extract_components(sections: dict[str, str], names: list[str], fallback_prefix: str) -> list[ModelComponent]:
    text = _section_text(sections, names)
    components = []
    for idx, item in enumerate(_list_items(text), start=1):
        name_match = re.match(r"`?([A-Za-z][A-Za-z0-9_,{}\[\]()^ -]{0,40})`?\s*[:：-]\s*(.+)", item)
        if name_match:
            name = name_match.group(1).strip(" `")
            description = name_match.group(2).strip()
        else:
            name = f"{fallback_prefix}_{idx}"
            description = item
        components.append(ModelComponent(name=name, description=description))
    return components


def _extract_objective(sections: dict[str, str]) -> str | None:
    text = _section_text(sections, ["objective function", "objective"])
    if not text:
        return None
    items = _list_items(text)
    return items[0] if items else re.sub(r"\s+", " ", text).strip()


def _extract_bullets(sections: dict[str, str], names: list[str]) -> list[str]:
    return _list_items(_section_text(sections, names))


def _section_text(sections: dict[str, str], names: list[str]) -> str:
    for key, value in sections.items():
        if any(name in key for name in names):
            return value
    return ""


def _list_items(text: str) -> list[str]:
    items = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith(("-", "*")):
            items.append(stripped.lstrip("-* ").strip())
    if not items and text.strip():
        sentences = [item.strip() for item in re.split(r"(?<=[.;])\s+", text.strip()) if item.strip()]
        return sentences[:8]
    return items


def _objective_sense(text: str) -> str:
    lowered = text.lower()
    has_max = "maximize" in lowered or "maximise" in lowered or "max " in lowered
    has_min = "minimize" in lowered or "minimise" in lowered or "min " in lowered
    if has_max and has_min:
        return "unknown"
    if has_max:
        return "maximize"
    if has_min:
        return "minimize"
    return "unknown"


def _component_lines(components: list[ModelComponent]) -> list[str]:
    if not components:
        return ["- Needs human verification: none extracted."]
    return [f"- `{item.name}`: {item.description}" for item in components]
