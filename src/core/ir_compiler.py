"""Turn a declared formulation into a model the solver layer can actually run.

This project has two halves that never met. One is a rigorous solver path:
`MixedIntegerProgram` with verified solutions, exhaustive cross-checks, derived
Big-M constants and oracle-compared transformation chains. The other is the
research pipeline, which produced a formulation and then ran a template that
invented its own numbers. Nothing carried a model from the first half into the
second, so the rigour applied only to problems the pipeline had not written.

This is the bridge, and its most important behaviour is refusing to cross.
`OptimizationModel` holds expressions as text; turning text into coefficients
means parsing, and a parser that guesses would produce a model that solves
cleanly while encoding something else -- the exact failure the requirement
check and the typed IR exist to prevent, reintroduced one layer down. So the
grammar here is small and total: linear terms over declared variables with
resolvable numeric coefficients. Anything outside it is refused by name, with
the reason attached to the constraint that caused it.

A refusal is not a failure of the formulation. Most real models are indexed and
symbolic, and cannot compile without instance data the IR does not carry. The
report says which ones and why, which is the difference between "this did not
run" and "this cannot run until someone supplies the members of N".
"""

from __future__ import annotations

import re

from pydantic import BaseModel, Field

from src.core.mixed_integer_program import MipConstraint, MipVariable, MixedIntegerProgram
from src.core.or_ir import OptimizationModel


#: Markers of an aggregate or quantified expression. These are not unsupported
#: syntax so much as a different kind of object: `sum_j a_j x_j` denotes one
#: constraint per member of a set whose members the IR never states.
_AGGREGATE = re.compile(r"(\bsum\b|sum_|\\sum|∑|\bfor all\b|∀|\bforall\b|\bprod\b|∏)", re.IGNORECASE)
_STRICT = re.compile(r"(?<![<>=])<(?!=)|(?<![<>=])>(?!=)")
_COMPARISON = re.compile(r"(<=|>=|==|≤|≥|=)")
_TOKEN = re.compile(
    r"\s*(?:(?P<number>\d+\.?\d*(?:[eE][+-]?\d+)?)|(?P<name>[A-Za-z_][A-Za-z0-9_]*)|(?P<op>[-+*/()]))"
)


class CompileRefusal(Exception):
    """Raised when an expression leaves the grammar. Carries the reason."""


class Refusal(BaseModel):
    """One thing that could not be compiled, and why."""

    where: str
    expression: str = ""
    reason: str


class CompileReport(BaseModel):
    compiled: bool = False
    mip: MixedIntegerProgram | None = None
    refusals: list[Refusal] = Field(default_factory=list)
    variable_order: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

    def markdown(self) -> str:
        lines = ["# Model Compilation", ""]
        if self.compiled and self.mip is not None:
            lines.extend(
                [
                    "The formulation compiled into an executable model.",
                    "",
                    f"- variables: {self.mip.n_vars} ({self.mip.n_binaries} binary)",
                    f"- constraints: {self.mip.n_constraints}",
                    f"- sense: {self.mip.sense}",
                    f"- variable order: {', '.join(self.variable_order)}",
                    "",
                ]
            )
        else:
            lines.extend(
                [
                    "The formulation did not compile into an executable model. This is a statement",
                    "about what the declaration carries, not a judgement of the model: an indexed or",
                    "symbolic formulation is normal and cannot be compiled without instance data.",
                    "",
                ]
            )
        if self.refusals:
            lines.extend(["## What could not be compiled", ""])
            for item in self.refusals:
                lines.append(f"- **{item.where}**: {item.reason}")
                if item.expression:
                    lines.append(f"  - `{item.expression}`")
            lines.append("")
        if self.notes:
            lines.extend(["## Notes", ""])
            lines.extend(f"- {item}" for item in self.notes)
            lines.append("")
        lines.extend(
            [
                "## What compiling does and does not establish",
                "",
                "- It establishes that the declared expressions are linear and that every symbol",
                "  resolves to a declared variable or a parameter with a value.",
                "- It does not establish that the model is the right one. A model can compile,",
                "  solve and verify while encoding a problem nobody asked for; that is what the",
                "  requirement coverage check is for.",
            ]
        )
        return "\n".join(lines) + "\n"


class LinearExpr:
    """`sum(coefficients) + constant`, with linearity enforced as it is built."""

    __slots__ = ("coefficients", "constant")

    def __init__(self, coefficients: dict[str, float] | None = None, constant: float = 0.0) -> None:
        self.coefficients = coefficients or {}
        self.constant = constant

    @property
    def is_constant(self) -> bool:
        return not self.coefficients

    def scaled(self, factor: float) -> LinearExpr:
        return LinearExpr(
            {name: value * factor for name, value in self.coefficients.items()},
            self.constant * factor,
        )

    def plus(self, other: LinearExpr, sign: float = 1.0) -> LinearExpr:
        merged = dict(self.coefficients)
        for name, value in other.coefficients.items():
            merged[name] = merged.get(name, 0.0) + value * sign
        return LinearExpr(merged, self.constant + other.constant * sign)


def compile_to_mip(model: OptimizationModel) -> CompileReport:
    """Compile what can be compiled exactly; refuse the rest by name."""

    report = CompileReport()

    indexed = [item.name for item in model.decision_variables if item.indexed_by]
    if indexed:
        sets = sorted({name for item in model.decision_variables for name in item.indexed_by})
        report.refusals.append(
            Refusal(
                where="model",
                reason=(
                    "The model is indexed: "
                    + ", ".join(indexed)
                    + " range over "
                    + ", ".join(f"`{name}`" for name in sets)
                    + ". Compiling needs the members of those sets and the values of the "
                    "parameters over them, which a formulation declares but does not carry. "
                    "Supply an instance, or write the model out in ground form."
                ),
            )
        )
        return report
    if not model.decision_variables:
        report.refusals.append(Refusal(where="model", reason="The model declares no decision variables."))
        return report
    if model.objective is None:
        report.refusals.append(Refusal(where="objective", reason="The model declares no objective."))
        return report

    constants = {
        item.name: float(item.value)
        for item in model.parameters
        if item.value is not None
    }
    variables: list[MipVariable] = []
    for spec in model.decision_variables:
        if spec.domain == "unspecified":
            report.refusals.append(
                Refusal(where=f"variable `{spec.name}`", reason="No domain is declared.")
            )
            continue
        if spec.domain == "integer":
            report.refusals.append(
                Refusal(
                    where=f"variable `{spec.name}`",
                    reason=(
                        "Declared integer. The executable form carries binary and continuous "
                        "variables only, so a general integer variable would have to be "
                        "re-encoded -- silently doing that would change the model."
                    ),
                )
            )
            continue
        try:
            lower = _resolve_bound(spec.lower_bound, constants, default=0.0)
            upper = _resolve_bound(spec.upper_bound, constants, default=None)
        except CompileRefusal as exc:
            report.refusals.append(Refusal(where=f"variable `{spec.name}`", reason=str(exc)))
            continue
        variables.append(
            MipVariable(
                name=spec.name,
                lower=lower,
                upper=upper,
                is_binary=spec.domain == "binary",
            )
        )

    if report.refusals:
        return report

    order = [item.name for item in variables]
    known = set(order)

    try:
        objective_expr = _parse(model.objective.rendered(), known, constants)
    except CompileRefusal as exc:
        report.refusals.append(
            Refusal(where="objective", expression=model.objective.rendered(), reason=str(exc))
        )
        return report

    constraints: list[MipConstraint] = []
    for spec in model.constraints:
        if spec.enforcement == "soft":
            report.refusals.append(
                Refusal(
                    where=f"constraint `{spec.name}`",
                    expression=spec.expression,
                    reason=(
                        "Declared soft. A soft constraint belongs in the objective with its "
                        "penalty, and moving it there is a modelling decision this compiler "
                        "will not make on its own."
                    ),
                )
            )
            continue
        try:
            constraints.append(_parse_constraint(spec.name, spec.expression, order, constants))
        except CompileRefusal as exc:
            report.refusals.append(
                Refusal(where=f"constraint `{spec.name}`", expression=spec.expression, reason=str(exc))
            )

    if report.refusals:
        return report

    report.mip = MixedIntegerProgram(
        name=model.problem_name,
        sense=model.objective.sense,
        variables=variables,
        objective=[objective_expr.coefficients.get(name, 0.0) for name in order],
        objective_offset=objective_expr.constant,
        constraints=constraints,
        metadata={
            "compiled_from": "OptimizationModel",
            "model_source": model.source,
            "requirement_links": {
                spec.name: spec.source_requirement
                for spec in model.constraints
                if spec.source_requirement
            },
        },
    )
    report.compiled = True
    report.variable_order = order
    unresolved = [item.name for item in model.parameters if item.value is None]
    if unresolved:
        report.notes.append(
            "These declared parameters carry no value and were not needed by any expression: "
            + ", ".join(unresolved)
            + "."
        )
    return report


def _parse_constraint(
    name: str, expression: str, order: list[str], constants: dict[str, float]
) -> MipConstraint:
    """`order` is the variable order of the model, not a set: the coefficient
    vector is positional, and sorting here instead would scramble it."""
    text = (expression or "").strip()
    if not text:
        raise CompileRefusal("It is named but never written as an expression.")
    _reject_aggregates(text)
    if _STRICT.search(text):
        raise CompileRefusal(
            "Strict inequality. A mixed-integer program has no strict inequalities; write "
            "`<=` or `>=`, or state the margin explicitly."
        )
    parts = _COMPARISON.split(text)
    if len(parts) != 3:
        found = len(parts) // 2
        raise CompileRefusal(
            f"Expected exactly one comparison operator, found {found}. A double inequality must "
            "be written as two constraints."
            if found
            else "No comparison operator. This is an expression, not a constraint."
        )
    left, operator, right = parts
    known = set(order)
    lhs = _parse(left, known, constants)
    rhs = _parse(right, known, constants)
    combined = lhs.plus(rhs, sign=-1.0)
    if combined.is_constant:
        raise CompileRefusal("No variable appears, so this constrains nothing.")
    sense = {"≤": "<=", "≥": ">=", "=": "==", "==": "=="}.get(operator, operator)
    return MipConstraint(
        name=name,
        coefficients=[combined.coefficients.get(item, 0.0) for item in order],
        sense=sense,  # type: ignore[arg-type]
        rhs=-combined.constant,
    )


def _resolve_bound(
    raw: str | None, constants: dict[str, float], default: float | None
) -> float | None:
    if raw is None or not str(raw).strip():
        return default
    text = str(raw).strip()
    if text.lower() in {"inf", "+inf", "infinity", "none", "null"}:
        return None
    if text.lower() in {"-inf", "-infinity"}:
        raise CompileRefusal(
            "A lower bound of minus infinity cannot be represented; the executable form takes a "
            "finite lower bound."
        )
    try:
        return float(text)
    except ValueError:
        pass
    if text in constants:
        return constants[text]
    raise CompileRefusal(
        f"The bound `{text}` is a symbol with no value. Declare it as a parameter with a value, "
        "or write a number."
    )


def _reject_aggregates(text: str) -> None:
    match = _AGGREGATE.search(text)
    if match:
        raise CompileRefusal(
            f"Contains `{match.group(0)}`, which quantifies over a set. Compiling it needs that "
            "set's members and the parameter values over them, which the formulation declares "
            "but does not carry."
        )


def _parse(text: str, known: set[str], constants: dict[str, float]) -> LinearExpr:
    _reject_aggregates(text)
    tokens = _tokenize(text)
    parser = _Parser(tokens, known, constants)
    expression = parser.expression()
    parser.expect_end()
    return expression


def _tokenize(text: str) -> list[tuple[str, str]]:
    tokens: list[tuple[str, str]] = []
    position = 0
    body = (text or "").strip()
    while position < len(body):
        match = _TOKEN.match(body, position)
        if not match:
            raise CompileRefusal(f"Cannot read `{body[position:position + 12].strip()}`.")
        position = match.end()
        for kind in ("number", "name", "op"):
            value = match.group(kind)
            if value is not None:
                tokens.append((kind, value))
                break
    return tokens


class _Parser:
    """Recursive descent over a linear grammar, with linearity enforced."""

    def __init__(self, tokens: list[tuple[str, str]], known: set[str], constants: dict[str, float]) -> None:
        self.tokens = tokens
        self.position = 0
        self.known = known
        self.constants = constants

    def peek(self) -> tuple[str, str] | None:
        return self.tokens[self.position] if self.position < len(self.tokens) else None

    def take(self) -> tuple[str, str]:
        token = self.peek()
        if token is None:
            raise CompileRefusal("The expression ends before it is complete.")
        self.position += 1
        return token

    def expect_end(self) -> None:
        if self.peek() is not None:
            raise CompileRefusal(f"Unexpected `{self.peek()[1]}` after the expression ends.")

    def expression(self) -> LinearExpr:
        result = self.term()
        while True:
            token = self.peek()
            if token is None or token[0] != "op" or token[1] not in "+-":
                return result
            self.take()
            result = result.plus(self.term(), sign=1.0 if token[1] == "+" else -1.0)

    def term(self) -> LinearExpr:
        result = self.factor()
        while True:
            token = self.peek()
            if token is None or token[0] != "op" or token[1] not in "*/":
                return result
            self.take()
            other = self.factor()
            if token[1] == "*":
                if result.is_constant:
                    result = other.scaled(result.constant)
                elif other.is_constant:
                    result = result.scaled(other.constant)
                else:
                    raise CompileRefusal(
                        "Two variables are multiplied together, so the expression is not linear."
                    )
            else:
                if not other.is_constant:
                    raise CompileRefusal(
                        "Division by a variable, so the expression is not linear."
                    )
                if other.constant == 0.0:
                    raise CompileRefusal("Division by zero.")
                result = result.scaled(1.0 / other.constant)

    def factor(self) -> LinearExpr:
        token = self.peek()
        if token is not None and token[0] == "op" and token[1] in "+-":
            self.take()
            inner = self.factor()
            return inner.scaled(-1.0) if token[1] == "-" else inner
        return self.atom()

    def atom(self) -> LinearExpr:
        kind, value = self.take()
        if kind == "number":
            return LinearExpr(constant=float(value))
        if kind == "name":
            if value in self.known:
                return LinearExpr({value: 1.0})
            if value in self.constants:
                return LinearExpr(constant=self.constants[value])
            raise CompileRefusal(
                f"`{value}` is neither a declared decision variable nor a parameter with a value."
            )
        if value == "(":
            inner = self.expression()
            closing = self.peek()
            if closing is None or closing[1] != ")":
                raise CompileRefusal("An opening bracket is never closed.")
            self.take()
            return inner
        raise CompileRefusal(f"Unexpected `{value}`.")
