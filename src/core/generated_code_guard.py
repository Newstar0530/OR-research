"""What has to be true of a script before it is allowed to replace a working one.

Letting a language model rewrite a whole experiment is the only way this system
repairs a failure it has no rule for, and it is also the only way it can quietly
destroy its own guarantees. A rewrite that deletes the solve loop still
compiles. One that writes an empty `results.csv` and prints the summary line
still satisfies the experiment contract. One that reaches for the network still
runs -- and then the run is neither reproducible nor sandboxed.

So a rewrite is not accepted because a model produced it. It is accepted only
after passing the checks below, and a rejection is specific enough to hand back
to the model as the next instruction:

* it parses as Python;
* it still writes `results.csv` and still prints the `SUMMARY_JSON:` line, so
  the contract that makes a node measurable survives the edit;
* it opens no sockets and shells out to nothing -- checked on the parse tree,
  because a string search for `import socket` is fooled by anything that does
  not spell it that way;
* it writes nowhere but its own workspace;
* it is not a gutted stub. A "fix" that deletes the experiment passes every
  other check on this list, which is exactly why this one exists.

None of this makes a rewrite correct. It makes a rewrite *bounded*: the ways it
can be wrong are ways a human reading `results.csv` can still notice.
"""

from __future__ import annotations

import ast
import re
from pydantic import BaseModel, Field


#: Modules whose presence means the script can leave the sandbox.
FORBIDDEN_MODULES: frozenset[str] = frozenset(
    {
        "socket", "ssl", "urllib", "urllib2", "urllib3", "requests", "httpx", "aiohttp",
        "http", "ftplib", "smtplib", "poplib", "imaplib", "telnetlib", "xmlrpc",
        "subprocess", "multiprocessing", "ctypes", "pty", "shutil",
    }
)

#: Callables that execute text or reach the shell.
FORBIDDEN_CALLS: frozenset[str] = frozenset(
    {"eval", "exec", "compile", "__import__", "system", "popen", "spawn", "fork", "execv"}
)

#: The two outputs that make a node measurable at all.
REQUIRED_MARKERS: tuple[tuple[str, str], ...] = (
    ("results.csv", "the results table every downstream stage reads"),
    ("SUMMARY_JSON", "the stdout line the experiment contract checks for"),
)

_ABSOLUTE_PATH = re.compile(r"^(?:/|[A-Za-z]:[\\/])")
_FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)


class GuardVerdict(BaseModel):
    """Why a candidate script was accepted or refused."""

    ok: bool
    violations: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

    def to_feedback(self) -> str:
        """The rejection, phrased so it can be handed straight back to the model."""

        if self.ok:
            return ""
        lines = ["Your previous attempt was rejected. Fix all of these:"]
        lines.extend(f"- {item}" for item in self.violations)
        return "\n".join(lines)


def extract_code(response: str) -> str:
    """Pull the script out of a model response, fenced or not."""

    text = (response or "").strip()
    if not text:
        return ""
    fences = _FENCE.findall(text)
    if fences:
        # The longest block, because a model often shows a one-line snippet
        # before the full file.
        return max(fences, key=len).strip()
    return text


def _forbidden_imports(tree: ast.AST) -> list[str]:
    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root in FORBIDDEN_MODULES:
                    found.append(root)
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root in FORBIDDEN_MODULES:
                found.append(root)
    return sorted(set(found))


def _forbidden_calls(tree: ast.AST) -> list[str]:
    found: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = ""
        if isinstance(func, ast.Name):
            name = func.id
        elif isinstance(func, ast.Attribute):
            name = func.attr
        if name in FORBIDDEN_CALLS:
            found.append(name)
    return sorted(set(found))


def _escaping_paths(tree: ast.AST) -> list[str]:
    """String literals that point outside the node's own workspace."""

    found: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
            continue
        value = node.value
        if len(value) > 260 or "\n" in value:
            continue
        if _ABSOLUTE_PATH.match(value) or value.startswith("..") or "/../" in value or "\\..\\" in value:
            found.append(value)
    return sorted(set(found))


def _substantial_lines(code: str) -> int:
    return sum(
        1
        for line in code.splitlines()
        if line.strip() and not line.strip().startswith("#")
    )


def validate_generated_experiment(
    code: str,
    original: str | None = None,
    min_retained_fraction: float = 0.5,
    require_contract: bool = True,
) -> GuardVerdict:
    """Check a candidate experiment script. Never raises; reports instead."""

    violations: list[str] = []
    notes: list[str] = []

    if not (code or "").strip():
        return GuardVerdict(ok=False, violations=["The response contained no code."])

    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        line = f" at line {exc.lineno}" if exc.lineno else ""
        return GuardVerdict(
            ok=False,
            violations=[f"The script does not parse as Python{line}: {exc.msg}."],
        )
    except ValueError as exc:  # null bytes and similar
        return GuardVerdict(ok=False, violations=[f"The script could not be parsed: {exc}."])

    if require_contract:
        for marker, why in REQUIRED_MARKERS:
            if marker not in code:
                violations.append(
                    f"`{marker}` no longer appears in the script, so {why} would be lost. "
                    "Keep it."
                )

    forbidden_modules = _forbidden_imports(tree)
    if forbidden_modules:
        violations.append(
            "Remove the import of "
            + ", ".join(f"`{name}`" for name in forbidden_modules)
            + ". The experiment must run offline and inside this workspace."
        )

    forbidden_calls = _forbidden_calls(tree)
    if forbidden_calls:
        violations.append(
            "Remove the call to "
            + ", ".join(f"`{name}`" for name in forbidden_calls)
            + ". The experiment may not execute generated text or shell out."
        )

    escaping = _escaping_paths(tree)
    if escaping:
        violations.append(
            "These paths leave the node workspace: "
            + ", ".join(f"`{item}`" for item in escaping[:5])
            + ". Write outputs relative to the script's own directory."
        )

    if original:
        before, after = _substantial_lines(original), _substantial_lines(code)
        if before and after < before * min_retained_fraction:
            violations.append(
                f"The rewrite keeps {after} of {before} code lines "
                f"({after / before:.0%}). A repair changes the smallest thing that fixes the "
                "error; it does not delete the experiment."
            )
        elif before:
            notes.append(f"Rewrite kept {after} of {before} code lines ({after / before:.0%}).")

    return GuardVerdict(ok=not violations, violations=violations, notes=notes)


#: A line that changes nothing about what the script does.
_IGNORABLE = re.compile(r'^\s*(#|"""|\'\'\'|$)')


def behavioural_lines(code: str) -> list[str]:
    """The code with comments, docstring markers and blank lines dropped.

    Used to answer one question the mutation layer could not previously ask:
    did this variant actually change anything? The autonomous header the refiner
    inserts is a docstring, so a variant that differs only by its header differs
    only in how it describes itself.
    """

    return [
        " ".join(line.split())
        for line in (code or "").splitlines()
        if not _IGNORABLE.match(line)
    ]


def validate_mutation(
    code: str,
    parent_code: str,
    min_retained_fraction: float = 0.5,
) -> GuardVerdict:
    """Check a *variant* of an existing experiment.

    Everything `validate_generated_experiment` requires, plus the requirement
    that distinguishes a mutation from a copy: it has to differ from its parent
    somewhere that matters.

    That check is not pedantry. The old mutation layer applied exact string
    replacements, so on any template that did not happen to contain
    `PROPOSED_EFFECT_LOW = 0.5` every replacement was a no-op and the node
    became a byte-identical copy of its parent -- which the search then counted
    as an explored branch, scored, and ranked. Nothing noticed.
    """

    verdict = validate_generated_experiment(
        code, parent_code, min_retained_fraction=min_retained_fraction
    )
    if not verdict.ok:
        return verdict
    if behavioural_lines(code) == behavioural_lines(parent_code):
        return GuardVerdict(
            ok=False,
            violations=[
                "The variant is identical to its parent once comments and the header are "
                "ignored, so it would re-measure the parent rather than explore anything. "
                "Change something the script actually does."
            ],
            notes=verdict.notes,
        )
    return verdict
