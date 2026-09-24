"""Recover a formulation from prose, when no structured one was produced.

This used to be the only representation the project had: a regular-expression
pass over `model_draft.md` that stored every set, parameter, variable and
constraint as a name and a sentence. It was called an IR, but nothing in it
could be checked -- a constraint was a string, a parameter had no unit, and the
objective sense was guessed by searching the document for the word `minimize`.

`or_ir.OptimizationModel` is now the representation, and a model that answers
with fields fills it in directly. This module is what happens when that fails:
a model that replied in prose, a hand-written draft, or the scaffold. It
produces the same type so nothing downstream has to know which path it came
from -- but it marks `source="markdown_extraction"`, and the fields it cannot
recover stay empty rather than being invented. Most of them cannot be
recovered: prose rarely states a variable's domain, a parameter's unit, or the
requirement a constraint exists to satisfy.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from src.core.or_ir import (
    ConstraintSpec,
    ObjectiveSpec,
    OptimizationModel,
    ParameterSpec,
    SetSpec,
    VariableSpec,
)


#: `p_j: processing time of job j, in minutes` -- the one shape prose states a
#: unit in often enough to be worth reading.
_UNIT = re.compile(r",\s*(?:in|measured in)\s+([A-Za-z][A-Za-z0-9_/%·^\- ]{0,30})\s*$", re.IGNORECASE)
_INDEXED_BY = re.compile(r"indexed by\s+([A-Za-z][A-Za-z0-9_,\s]{0,20})", re.IGNORECASE)
_BINARY = re.compile(r"\bbinary\b|\bin\s*\{\s*0\s*,\s*1\s*\}|\b0-1\b", re.IGNORECASE)
_INTEGER = re.compile(r"\binteger\b", re.IGNORECASE)
_CONTINUOUS = re.compile(r"\bcontinuous\b|\breal\b|\bfractional\b", re.IGNORECASE)


def build_model_ir(model_markdown: str, problem_name: str = "research_model") -> OptimizationModel:
    """Scrape what the prose happens to state. Everything else stays empty."""

    sections = _sections(model_markdown)
    sets = [
        SetSpec(name=name, description=description, index=_index_of(description))
        for name, description in _items(sections, ["sets and indices", "sets", "indices"], "set")
    ]
    parameters = [
        ParameterSpec(
            name=name,
            description=description,
            unit=_unit_of(description),
            indexed_by=_indexed_by(description),
        )
        for name, description in _items(sections, ["parameters"], "parameter")
    ]
    variables = [
        VariableSpec(
            name=name,
            description=description,
            domain=_domain_of(description),
            indexed_by=_indexed_by(description),
        )
        for name, description in _items(sections, ["decision variables", "variables"], "variable")
    ]
    constraints = [
        ConstraintSpec(name=name, expression=_expression_of(description), indexed_by=_indexed_by(description))
        for name, description in _items(sections, ["constraints"], "constraint")
    ]

    model = OptimizationModel(
        problem_name=problem_name,
        sets=sets,
        parameters=parameters,
        decision_variables=variables,
        objective=_objective(sections, model_markdown),
        constraints=constraints,
        assumptions=_bullets(sections, ["assumptions"]),
        limitations=_bullets(sections, ["limitations"]),
        source="markdown_extraction",
    )
    if not variables:
        model.extraction_warnings.append("No decision variables could be read out of the prose.")
    if not constraints:
        model.extraction_warnings.append("No constraints could be read out of the prose.")
    if model.objective is None:
        model.extraction_warnings.append(
            "No objective direction could be read: the draft either states none or states both."
        )
    model.extraction_warnings.append(
        "These fields were recovered from a document, not declared. Units, variable domains, "
        "bounds and requirement links are absent unless the prose stated them in a shape this "
        "scraper recognises."
    )
    return model


def write_model_ir(model: OptimizationModel, output_dir: str | Path) -> OptimizationModel:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    payload = model.model_dump()
    payload["objective_sense"] = model.objective_sense
    payload["structural_report"] = model.structural_report().model_dump()
    (root / "model_ir.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (root / "model_ir.md").write_text(model.render_markdown(), encoding="utf-8")
    return model


def _objective(sections: dict[str, str], whole: str) -> ObjectiveSpec | None:
    text = _section_text(sections, ["objective function", "objective"]) or whole
    lowered = text.lower()
    has_max = "maximize" in lowered or "maximise" in lowered
    has_min = "minimize" in lowered or "minimise" in lowered
    if has_max == has_min:
        # Both or neither: the draft has not chosen, and guessing here is how
        # a model ends up optimising in the wrong direction silently.
        return None
    items = _list_items(text)
    expression = items[0] if items else re.sub(r"\s+", " ", text).strip()
    return ObjectiveSpec(
        sense="maximize" if has_max else "minimize",
        expression=re.sub(r"^\s*(minimi[sz]e|maximi[sz]e)\s*", "", expression, flags=re.IGNORECASE),
    )


def _sections(markdown: str) -> dict[str, str]:
    matches = list(re.finditer(r"^#{1,3}\s+(.+?)\s*$", markdown or "", flags=re.MULTILINE))
    sections: dict[str, str] = {}
    for index, match in enumerate(matches):
        title = match.group(1).strip().lower()
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(markdown or "")
        sections[title] = (markdown or "")[start:end].strip()
    return sections


def _items(sections: dict[str, str], names: list[str], fallback_prefix: str) -> list[tuple[str, str]]:
    text = _section_text(sections, names)
    found: list[tuple[str, str]] = []
    for idx, item in enumerate(_list_items(text), start=1):
        match = re.match(r"`?([A-Za-z][A-Za-z0-9_,{}\[\]()^ -]{0,40})`?\s*[:：-]\s*(.+)", item)
        if match:
            found.append((match.group(1).strip(" `"), match.group(2).strip()))
        else:
            found.append((f"{fallback_prefix}_{idx}", item))
    return found


def _bullets(sections: dict[str, str], names: list[str]) -> list[str]:
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


def _unit_of(description: str) -> str | None:
    match = _UNIT.search(description or "")
    return match.group(1).strip() if match else None


def _indexed_by(description: str) -> list[str]:
    match = _INDEXED_BY.search(description or "")
    if not match:
        return []
    return [part.strip() for part in match.group(1).split(",") if part.strip()][:4]


def _index_of(description: str) -> str | None:
    indices = _indexed_by(description)
    return indices[0] if indices else None


def _domain_of(description: str) -> str:
    body = description or ""
    if _BINARY.search(body):
        return "binary"
    if _INTEGER.search(body):
        return "integer"
    if _CONTINUOUS.search(body):
        return "continuous"
    return "unspecified"


def _expression_of(description: str) -> str:
    """Keep the part that looks like algebra, if any of it does."""

    body = (description or "").strip()
    return body if re.search(r"(<=|>=|=|≤|≥|sum|\\sum|∑)", body) else ""
