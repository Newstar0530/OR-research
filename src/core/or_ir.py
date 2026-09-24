"""The formulation as a typed object rather than a document.

For most of this project's life the formulation was `ModelDraft.markdown: str`.
Everything downstream either read prose or re-derived structure from it with
regular expressions, which meant the business semantics -- what a parameter is
measured in, which requirement a constraint exists to satisfy, whether a bound
is a modelling choice or a physical fact -- survived only as English inside a
paragraph. Nothing could check them, because there was nothing to check.

A typed representation moves the model out of the prose and into fields an
independent validator can read: units on parameters, a source requirement on
each constraint, an explicit domain and bounds on each variable, a stated sense
and a business meaning for each objective term. The document is still produced,
because a human has to read it -- but it is now *rendered from* this object
rather than being the thing itself. When the two could disagree, the object
wins, because it is the one the checks can see.

What this is not: an algebra. Expressions are held as strings and are not
parsed, so nothing here can tell that a constraint is wrong -- only that it is
absent, unlabelled, unbounded or undeclared. Catching a wrong expression needs
the solver path, which is what `MixedIntegerProgram` already does once a model
reaches it.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


ObjectiveSense = Literal["minimize", "maximize"]
VariableDomain = Literal["binary", "integer", "continuous", "unspecified"]
Enforcement = Literal["hard", "soft"]
#: How the object was built. `llm_structured` came back as fields; the others
#: were recovered from prose or never written, and are far weaker.
ModelSource = Literal["llm_structured", "markdown_extraction", "scaffold"]


class SetSpec(BaseModel):
    name: str
    description: str = ""
    #: The symbol used to index it, e.g. `i` for the set `I`.
    index: str | None = None


class ParameterSpec(BaseModel):
    name: str
    description: str = ""
    #: What the number is measured in. A parameter without a unit is the
    #: standard way a formulation adds minutes to hours and reports the sum as
    #: an optimum.
    unit: str | None = None
    #: The requirement or data source that fixes this value, so a reader can
    #: ask where a number came from rather than trusting it.
    source: str | None = None
    indexed_by: list[str] = Field(default_factory=list)


class VariableSpec(BaseModel):
    name: str
    description: str = ""
    domain: VariableDomain = "unspecified"
    #: Held as strings because a bound is often symbolic (`0`, `u_i`, `M`).
    lower_bound: str | None = None
    upper_bound: str | None = None
    indexed_by: list[str] = Field(default_factory=list)
    #: The period the decision applies to, when the model is time-indexed.
    temporal_scope: str | None = None


class ObjectiveTerm(BaseModel):
    expression: str
    #: What the term is for in the problem, not in the algebra. A term nobody
    #: can explain is a term nobody can weigh against another.
    business_meaning: str = ""


class ObjectiveSpec(BaseModel):
    sense: ObjectiveSense
    terms: list[ObjectiveTerm] = Field(default_factory=list)
    expression: str = ""

    def rendered(self) -> str:
        if self.expression:
            return self.expression
        return " + ".join(term.expression for term in self.terms)


class ConstraintSpec(BaseModel):
    name: str
    #: The algebra, as written. Not parsed.
    expression: str = ""
    #: Which stated requirement this exists to impose, by id. This is the link
    #: `requirement_coverage` checks; a constraint that satisfies nothing
    #: stated is either undocumented or unnecessary, and both are worth seeing.
    source_requirement: str | None = None
    enforcement: Enforcement = "hard"
    #: What happens when a soft constraint is violated. A soft constraint with
    #: no penalty is not a constraint.
    penalty: str | None = None
    indexed_by: list[str] = Field(default_factory=list)


class UncertaintySpec(BaseModel):
    """Where the model admits it does not know a number."""

    parameter: str
    distribution: str | None = None
    scenarios: str | None = None
    #: For distributionally robust formulations: the set the distribution is
    #: only known to lie in.
    ambiguity_set: str | None = None


class SolverRequirements(BaseModel):
    problem_class: str | None = None
    suggested_solvers: list[str] = Field(default_factory=list)
    time_limit_seconds: float | None = None
    mip_gap: float | None = None


class ValidationTest(BaseModel):
    """A check the formulation claims it should pass once implemented."""

    description: str
    kind: Literal["feasibility", "bound", "invariant", "unit"] = "invariant"


class StructuralReport(BaseModel):
    """What the object is missing, as distinct from what it gets wrong."""

    missing: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @property
    def is_implementable(self) -> bool:
        """True when there is enough here to build a model from.

        Deliberately separate from correctness. An implementable formulation
        can still be the wrong one; an unimplementable one cannot even be
        tried, which is the failure this catches.
        """

        return not self.missing

    def failures(self) -> list[str]:
        return list(self.missing)


class OptimizationModel(BaseModel):
    """A formulation, typed."""

    problem_name: str = "research_model"
    sets: list[SetSpec] = Field(default_factory=list)
    parameters: list[ParameterSpec] = Field(default_factory=list)
    decision_variables: list[VariableSpec] = Field(default_factory=list)
    objective: ObjectiveSpec | None = None
    constraints: list[ConstraintSpec] = Field(default_factory=list)
    uncertainty: list[UncertaintySpec] = Field(default_factory=list)
    solver_requirements: SolverRequirements = Field(default_factory=SolverRequirements)
    validation_tests: list[ValidationTest] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    source: ModelSource = "scaffold"
    extraction_warnings: list[str] = Field(default_factory=list)

    @property
    def objective_sense(self) -> str:
        return self.objective.sense if self.objective else "unknown"

    def structural_report(self) -> StructuralReport:
        report = StructuralReport()
        if not self.decision_variables:
            report.missing.append("The model declares no decision variables.")
        if not self.constraints:
            report.missing.append("The model declares no constraints.")
        elif not any(item.expression.strip() for item in self.constraints):
            report.missing.append(
                "Every constraint is named but none is written as an expression. A named "
                "family such as `capacity constraints` cannot be implemented or checked."
            )
        if self.objective is None:
            report.missing.append("The model declares no objective, so no direction is chosen.")
        elif not self.objective.rendered().strip():
            report.missing.append(
                "The objective has a direction but no expression, so there is nothing to optimise."
            )

        undeclared = [item.name for item in self.decision_variables if item.domain == "unspecified"]
        if undeclared:
            report.missing.append(
                "These variables have no declared domain (binary, integer or continuous): "
                + ", ".join(undeclared)
                + "."
            )
        unwritten = [item.name for item in self.constraints if not item.expression.strip()]
        if unwritten:
            report.warnings.append(
                "These constraints are named but not written as expressions: "
                + ", ".join(unwritten)
                + "."
            )

        unitless = [item.name for item in self.parameters if not (item.unit or "").strip()]
        if unitless:
            report.warnings.append(
                "These parameters carry no unit, so nothing can detect a scale or dimension "
                "error in them: " + ", ".join(unitless) + "."
            )
        unsourced = [item.name for item in self.constraints if not (item.source_requirement or "").strip()]
        if unsourced:
            report.warnings.append(
                "These constraints cite no requirement, so why they exist is not recorded: "
                + ", ".join(unsourced)
                + "."
            )
        unpenalised = [
            item.name
            for item in self.constraints
            if item.enforcement == "soft" and not (item.penalty or "").strip()
        ]
        if unpenalised:
            report.warnings.append(
                "These constraints are soft but state no penalty, which makes them free to "
                "violate: " + ", ".join(unpenalised) + "."
            )
        return report

    def render_markdown(self) -> str:
        """The document a human reads, generated from the fields above.

        Constraints are emitted with their `[R*]` requirement tag so the
        coverage check reads the same link the object records, rather than
        guessing it back out of the prose.
        """

        lines = [f"# Mathematical Model Draft: {self.problem_name}", ""]
        if self.source != "llm_structured":
            lines.extend([_provenance_banner(self.source), ""])

        lines.extend(["## Problem Definition", ""])
        lines.append(
            f"A {self.solver_requirements.problem_class or 'mathematical'} formulation with "
            f"{len(self.decision_variables)} declared variable group(s) and "
            f"{len(self.constraints)} constraint group(s)."
        )

        lines.extend(["", "## Sets and Indices", ""])
        lines.extend(_or_placeholder(f"- {_set_line(item)}" for item in self.sets))

        lines.extend(["", "## Parameters", ""])
        lines.extend(_or_placeholder(f"- {_parameter_line(item)}" for item in self.parameters))

        lines.extend(["", "## Decision Variables", ""])
        lines.extend(_or_placeholder(f"- {_variable_line(item)}" for item in self.decision_variables))

        lines.extend(["", "## Objective Function", ""])
        if self.objective:
            lines.append(f"{self.objective.sense} {self.objective.rendered()}")
            if self.objective.terms:
                lines.append("")
                for term in self.objective.terms:
                    meaning = f" -- {term.business_meaning}" if term.business_meaning else ""
                    lines.append(f"- `{term.expression}`{meaning}")
        else:
            lines.append("Needs human verification: no objective was written.")

        lines.extend(["", "## Constraints", ""])
        lines.extend(_or_placeholder(f"- {_constraint_line(item)}" for item in self.constraints))

        if self.uncertainty:
            lines.extend(["", "## Uncertainty", ""])
            lines.extend(f"- {_uncertainty_line(item)}" for item in self.uncertainty)

        lines.extend(["", "## Assumptions", ""])
        lines.extend(_or_placeholder(f"- {item}" for item in self.assumptions))

        if self.validation_tests:
            lines.extend(["", "## Validation Tests", ""])
            lines.extend(f"- ({item.kind}) {item.description}" for item in self.validation_tests)

        if self.solver_requirements.suggested_solvers or self.solver_requirements.problem_class:
            lines.extend(["", "## Solver Requirements", ""])
            if self.solver_requirements.problem_class:
                lines.append(f"- problem class: {self.solver_requirements.problem_class}")
            if self.solver_requirements.suggested_solvers:
                lines.append(f"- suggested solvers: {', '.join(self.solver_requirements.suggested_solvers)}")
            if self.solver_requirements.time_limit_seconds is not None:
                lines.append(f"- time limit: {self.solver_requirements.time_limit_seconds} s")
            if self.solver_requirements.mip_gap is not None:
                lines.append(f"- acceptable MIP gap: {self.solver_requirements.mip_gap}")

        lines.extend(["", "## Limitations", ""])
        limitations = list(self.limitations)
        report = self.structural_report()
        limitations.extend(report.warnings)
        lines.extend(_or_placeholder(f"- {item}" for item in limitations))

        if self.extraction_warnings:
            lines.extend(["", "## Extraction Warnings", ""])
            lines.extend(f"- {item}" for item in self.extraction_warnings)

        return "\n".join(lines) + "\n"


def _provenance_banner(source: ModelSource) -> str:
    if source == "markdown_extraction":
        return (
            "> **Recovered from prose.** No structured formulation was produced, so the fields\n"
            "> below were scraped out of a markdown draft with regular expressions. Units,\n"
            "> bounds, domains and requirement links are absent unless the prose happened to\n"
            "> state them in a recognised shape."
        )
    return (
        "> **NOT GENERATED: this problem's formulation.** Nothing formulated this problem.\n"
        "> What follows is a form to fill in, not a draft of your model."
    )


def _or_placeholder(lines) -> list[str]:
    rendered = list(lines)
    return rendered or ["- Needs human verification: none declared."]


def _set_line(item: SetSpec) -> str:
    index = f", indexed by {item.index}" if item.index else ""
    description = f": {item.description}" if item.description else ""
    return f"`{item.name}`{description}{index}"


def _parameter_line(item: ParameterSpec) -> str:
    parts = [f"`{item.name}`"]
    if item.description:
        parts.append(f": {item.description}")
    if item.indexed_by:
        parts.append(f", indexed by {', '.join(item.indexed_by)}")
    parts.append(f" [unit: {item.unit}]" if item.unit else " [unit: **not stated**]")
    if item.source:
        parts.append(f" (source: {item.source})")
    return "".join(parts)


def _variable_line(item: VariableSpec) -> str:
    parts = [f"`{item.name}`"]
    if item.description:
        parts.append(f": {item.description}")
    if item.indexed_by:
        parts.append(f", indexed by {', '.join(item.indexed_by)}")
    if item.domain == "binary":
        parts.append(" in {0, 1}")
    else:
        bounds = ""
        if item.lower_bound is not None or item.upper_bound is not None:
            low = item.lower_bound if item.lower_bound is not None else "-inf"
            high = item.upper_bound if item.upper_bound is not None else "+inf"
            bounds = f", {low} <= {item.name} <= {high}"
        parts.append(f", {item.domain}{bounds}")
    if item.temporal_scope:
        parts.append(f", over {item.temporal_scope}")
    return "".join(parts)


def _constraint_line(item: ConstraintSpec) -> str:
    tag = f"[{item.source_requirement}] " if item.source_requirement else ""
    body = item.expression.strip() or f"**not written as an expression** ({item.name})"
    quantifier = f" for all {', '.join(item.indexed_by)}" if item.indexed_by else ""
    enforcement = ""
    if item.enforcement == "soft":
        penalty = item.penalty or "**no penalty stated**"
        enforcement = f" (soft, penalty: {penalty})"
    return f"{tag}`{item.name}`: {body}{quantifier}{enforcement}"


def _uncertainty_line(item: UncertaintySpec) -> str:
    parts = [f"`{item.parameter}`"]
    if item.distribution:
        parts.append(f": distribution {item.distribution}")
    if item.scenarios:
        parts.append(f", scenarios {item.scenarios}")
    if item.ambiguity_set:
        parts.append(f", ambiguity set {item.ambiguity_set}")
    if len(parts) == 1:
        parts.append(": uncertain, with no distribution or scenario set stated")
    return "".join(parts)
