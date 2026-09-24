"""Check a formulation against the requirements it is supposed to encode.

The verification this project already has proves that one model agrees with
another: a QUBO reproduces the MI 0-1 problem it was encoded from, a KKT
reformulation reproduces the bilevel problem it replaced, a returned solution
satisfies the constraints it was checked against. None of that can notice that
the chain began from the wrong problem. A solver returns an optimum for the
model it was handed, and a model that omits a requirement is optimal over a
feasible region that is too large -- which reads as a *better* objective, not
as an error. This is the one failure mode the numeric layer is blind to by
construction.

So this compares the draft against a written list of requirements and reports
which of them no constraint appears to impose. It does not prove completeness:
a requirement nobody wrote down cannot be looked for, and a lexical match is
evidence that words recur, not that an expression is correct. What it replaces
is the previous answer -- that automated checking could say nothing at all --
with a list of the specific requirements a reader should go and look for.
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field


RequirementKind = Literal["constraint", "objective"]
ObjectiveSense = Literal["minimize", "maximize", "unspecified"]
MatchMethod = Literal["declared", "llm", "lexical", "none"]

#: A constraint citing the requirement it imposes, as `[R2]` or `[R1, R2]`.
#: Asking the author to declare the link is stronger than inferring it: natural
#: language and algebra share almost no vocabulary, so `sum_s x_ns <= 5` and
#: "no nurse works more than five shifts" cannot be matched by their words, and
#: a model asked to judge its own draft is the blind spot this exists to close.
#: A declared link can still be false, but it is reviewable, which an absent
#: one is not.
_LABEL = re.compile(r"\[\s*(R\d+(?:\s*,\s*R\d+)*)\s*\]", re.IGNORECASE)

#: Framed as checking rather than authoring, so that a model asked to find
#: holes is not simultaneously being asked to defend the draft. Sharing one
#: client with `ModelingAgent` still means one model's blind spots apply to
#: both halves; a genuinely independent checker needs a second model.
COVERAGE_SYSTEM_PROMPT = (
    "You check whether a mathematical formulation encodes a list of stated requirements. "
    "You did not write the formulation and you do not improve it. For each requirement you "
    "decide exactly one thing: does some constraint in this draft actually impose it? A "
    "requirement that is only mentioned in prose, named as a family such as `capacity "
    "constraints`, or left implied by context is NOT encoded. Quote the constraint you relied "
    "on. Reporting a requirement as encoded when it is not is the costly error here."
)

_OBJECTIVE_LINE = re.compile(r"^\s*(minimi[sz]e|maximi[sz]e)\b", re.IGNORECASE | re.MULTILINE)

_STOPWORDS = frozenset(
    {
        "the", "and", "for", "with", "that", "this", "each", "any", "all", "not", "must",
        "should", "shall", "can", "cannot", "may", "are", "was", "were", "has", "have",
        "had", "its", "from", "into", "than", "then", "when", "where", "which", "while",
        "such", "per", "via", "also", "more", "most", "less", "least", "one", "two",
        "every", "some", "both", "over", "under", "between", "within", "across",
    }
)

#: A requirement counts as encoded when this share of its distinctive words
#: recur in one line. Tuned to be lenient: a false "missing" wastes a reviewer's
#: attention, while a false "covered" restores the blind spot this module exists
#: to close, so the LLM path is preferred wherever a model is configured.
_LEXICAL_THRESHOLD = 0.6
_MIN_MATCHED_TOKENS = 2


class Requirement(BaseModel):
    """One condition the formulation is supposed to impose."""

    id: str
    text: str
    kind: RequirementKind = "constraint"
    #: Where the requirement was stated, so a reader can go back and check that
    #: it is a real requirement rather than something the extractor invented.
    source: str = "research_goal"


class RequirementSet(BaseModel):
    """What the formulation will be held to."""

    requirements: list[Requirement] = Field(default_factory=list)
    required_objective_sense: ObjectiveSense = "unspecified"
    #: An empty list means "nothing was extracted", which is not the same as
    #: "nothing is required". Provenance keeps the two apart.
    extracted_by: Literal["llm", "derived", "not_generated"] = "not_generated"
    notes: list[str] = Field(default_factory=list)

    @property
    def constraint_requirements(self) -> list[Requirement]:
        return [item for item in self.requirements if item.kind == "constraint"]

    def texts(self) -> list[str]:
        return [item.text for item in self.requirements]


class RequirementMatch(BaseModel):
    requirement_id: str
    requirement_text: str
    covered: bool
    #: The draft line the decision rested on, so a disputed verdict can be read.
    evidence: str = ""
    method: MatchMethod = "none"


class CoverageVerdict(BaseModel):
    """Which requirements the draft appears to encode, and which it does not."""

    matches: list[RequirementMatch] = Field(default_factory=list)
    required_objective_sense: ObjectiveSense = "unspecified"
    observed_objective_sense: str = "unknown"
    #: False when there was no requirement list to check against. Distinguishes
    #: "the draft passed" from "nothing was checked".
    checked: bool = False
    searched_section: str = ""
    notes: list[str] = Field(default_factory=list)

    @property
    def missing(self) -> list[RequirementMatch]:
        return [item for item in self.matches if not item.covered]

    @property
    def encoded(self) -> list[RequirementMatch]:
        return [item for item in self.matches if item.covered]

    @property
    def coverage_ratio(self) -> float:
        if not self.matches:
            return 0.0
        return len(self.encoded) / len(self.matches)

    @property
    def missing_constraint_rate(self) -> float:
        if not self.matches:
            return 0.0
        return len(self.missing) / len(self.matches)

    @property
    def objective_sense_matches(self) -> bool:
        """An unspecified requirement cannot be contradicted."""

        if self.required_objective_sense == "unspecified":
            return True
        return self.observed_objective_sense == self.required_objective_sense

    @property
    def is_complete(self) -> bool:
        return self.checked and not self.missing and self.objective_sense_matches

    def blocks_acceptance(self) -> bool:
        """Whether this verdict is grounds for rejecting the draft.

        A check that could not run is not a failure. Only a check that ran and
        found something rejects, so a missing requirement list never silently
        stops the pipeline.
        """

        return self.checked and not self.is_complete

    def failures(self) -> list[str]:
        """Why the draft was rejected, phrased to be handed back to a writer."""

        reasons: list[str] = []
        if self.missing:
            reasons.append(
                "These stated requirements are not imposed by any constraint in the draft. "
                "Write each one as an expression, or state explicitly why it does not apply:\n"
                + "\n".join(f"  - {item.requirement_text}" for item in self.missing)
            )
        if not self.objective_sense_matches:
            reasons.append(
                f"The objective direction is `{self.observed_objective_sense}`, but the problem "
                f"calls for `{self.required_objective_sense}`. Minimising and maximising the same "
                "expression are different problems."
            )
        return reasons


def observed_objective_sense(draft: str) -> str:
    """The direction the draft actually declares.

    Reads the line that *begins* with the verb rather than searching the whole
    document for the word, because a draft discussing both directions in prose
    still declares only one.
    """

    match = _OBJECTIVE_LINE.search(draft or "")
    if not match:
        return "unknown"
    verb = match.group(1).lower()
    return "maximize" if verb.startswith("maxim") else "minimize"


class RequirementChecker:
    """Decides which requirements a draft encodes.

    Prefers a language model, because deciding whether an inequality imposes a
    stated condition is a semantic question. Falls back to lexical overlap so
    that the check still runs -- and still reports how it was run -- when no
    model is configured.
    """

    def __init__(self, llm=None) -> None:
        self.llm = llm

    def check(self, draft: str, requirements: RequirementSet) -> CoverageVerdict:
        if not requirements.requirements:
            return CoverageVerdict(
                checked=False,
                required_objective_sense=requirements.required_objective_sense,
                observed_objective_sense=observed_objective_sense(draft),
                notes=["No requirement list was available, so coverage was not checked."],
            )

        section_name, lines = _search_lines(draft)
        verdict = CoverageVerdict(
            checked=True,
            required_objective_sense=requirements.required_objective_sense,
            observed_objective_sense=observed_objective_sense(draft),
            searched_section=section_name,
        )
        if section_name == "whole document":
            verdict.notes.append(
                "The draft has no `## Constraints` section, so the whole document was searched. "
                "A requirement matched only in prose is not necessarily imposed."
            )

        matches = _declared_matches(requirements, lines)
        if matches is None:
            matches = self._llm_matches(draft, requirements)
        if matches is None:
            matches = [_lexical_match(item, lines) for item in requirements.requirements]
            verdict.notes.append(
                "No constraint cited a requirement and no model was available, so coverage was "
                "decided by word overlap. Algebra and prose share little vocabulary, so this "
                "method reports far more requirements missing than really are. Treat it as a "
                "prompt to look, not as a verdict."
            )
        verdict.matches = matches
        return verdict

    def _llm_matches(self, draft: str, requirements: RequirementSet) -> list[RequirementMatch] | None:
        if self.llm is None or getattr(self.llm, "use_mock", True):
            return None
        listing = "\n".join(f"{item.id}: {item.text}" for item in requirements.requirements)
        prompt = (
            "Formulation under review:\n\n"
            f"{draft[:8000]}\n\n"
            "Requirements it is supposed to encode:\n"
            f"{listing}\n\n"
            'Return JSON: {"results": [{"id": "<requirement id>", "encoded": true|false, '
            '"evidence": "<the constraint you relied on, verbatim, or why nothing encodes it>"}]}. '
            "Include every requirement id exactly once."
        )
        try:
            payload = self.llm.chat_json(COVERAGE_SYSTEM_PROMPT, prompt, stage="requirement_coverage")
        except Exception:
            # A checker that fails closed would block every run whose model call
            # flaked; the lexical path still produces a verdict, labelled as such.
            return None
        decisions = {}
        for entry in payload.get("results", []) or []:
            if isinstance(entry, dict) and entry.get("id"):
                decisions[str(entry["id"])] = entry
        if not decisions:
            return None
        matches = []
        for item in requirements.requirements:
            entry = decisions.get(item.id)
            if entry is None:
                # Silence is not assent: a requirement the checker skipped is
                # reported as unconfirmed rather than as encoded.
                matches.append(
                    RequirementMatch(
                        requirement_id=item.id,
                        requirement_text=item.text,
                        covered=False,
                        evidence="The checker did not return a verdict for this requirement.",
                        method="llm",
                    )
                )
                continue
            matches.append(
                RequirementMatch(
                    requirement_id=item.id,
                    requirement_text=item.text,
                    covered=bool(entry.get("encoded")),
                    evidence=str(entry.get("evidence", ""))[:400],
                    method="llm",
                )
            )
        return matches


def _search_lines(draft: str) -> tuple[str, list[str]]:
    """The lines a constraint could plausibly be written on.

    Restricted to the constraints section when there is one: a requirement
    named in the problem description is described, not imposed, and counting
    that as coverage would rebuild the blind spot.
    """

    body = draft or ""
    matches = list(re.finditer(r"^#{1,3}\s+(.+?)\s*$", body, flags=re.MULTILINE))
    for index, match in enumerate(matches):
        if "constraint" not in match.group(1).strip().lower():
            continue
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(body)
        section = body[start:end]
        return "## Constraints", [line for line in section.splitlines() if line.strip()]
    return "whole document", [line for line in body.splitlines() if line.strip() and not line.startswith("#")]


def _declared_matches(
    requirements: RequirementSet, lines: list[str]
) -> list[RequirementMatch] | None:
    """Read the requirement each constraint says it imposes.

    Returns None when the draft cites nothing, so an untagged draft falls
    through to a method that can still say something rather than being
    reported as covering nothing.
    """

    cited: dict[str, str] = {}
    for line in lines:
        for label in _LABEL.findall(line):
            for token in label.split(","):
                cited.setdefault(token.strip().upper(), line.strip())
    if not cited:
        return None
    return [
        RequirementMatch(
            requirement_id=item.id,
            requirement_text=item.text,
            covered=item.id.upper() in cited,
            evidence=cited.get(
                item.id.upper(), "No constraint in the draft cites this requirement."
            ),
            method="declared",
        )
        for item in requirements.requirements
    ]


def _tokens(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9_]+", (text or "").lower())
    return {word for word in words if len(word) >= 3 and word not in _STOPWORDS}


def _lexical_match(requirement: Requirement, lines: list[str]) -> RequirementMatch:
    wanted = _tokens(requirement.text)
    if not wanted:
        return RequirementMatch(
            requirement_id=requirement.id,
            requirement_text=requirement.text,
            covered=False,
            evidence="The requirement has no distinctive words to match on.",
            method="lexical",
        )
    best_line = ""
    best_score = 0.0
    best_hits = 0
    for line in lines:
        hits = wanted & _tokens(line)
        score = len(hits) / len(wanted)
        if score > best_score:
            best_line, best_score, best_hits = line.strip(), score, len(hits)
    needed_hits = min(_MIN_MATCHED_TOKENS, len(wanted))
    covered = best_score >= _LEXICAL_THRESHOLD and best_hits >= needed_hits
    return RequirementMatch(
        requirement_id=requirement.id,
        requirement_text=requirement.text,
        covered=covered,
        evidence=best_line if covered else "No constraint line shares enough of this requirement's wording.",
        method="lexical",
    )


def coverage_markdown(verdict: CoverageVerdict) -> str:
    """The verdict as a document, for the run folder."""

    lines = ["# Requirement Coverage", ""]
    if not verdict.checked:
        lines.extend(
            [
                "**Not checked.** No requirement list was available for this run, so nothing was",
                "compared against the formulation. This is not a pass.",
                "",
            ]
        )
        lines.extend(f"- {note}" for note in verdict.notes)
        return "\n".join(lines) + "\n"

    lines.extend(
        [
            f"- Requirements checked: {len(verdict.matches)}",
            f"- Encoded: {len(verdict.encoded)}",
            f"- Not encoded: {len(verdict.missing)}",
            f"- Missing-constraint rate: {verdict.missing_constraint_rate:.2f}",
            f"- Objective direction required: {verdict.required_objective_sense}",
            f"- Objective direction declared: {verdict.observed_objective_sense}",
            f"- Searched: {verdict.searched_section}",
            "",
        ]
    )
    if verdict.missing:
        lines.append("## Requirements no constraint imposes")
        lines.append("")
        for item in verdict.missing:
            lines.append(f"- **{item.requirement_text}**")
            lines.append(f"  - {item.evidence}")
        lines.append("")
    if verdict.encoded:
        lines.append("## Requirements the draft appears to impose")
        lines.append("")
        for item in verdict.encoded:
            lines.append(f"- {item.requirement_text}")
            if item.evidence:
                lines.append(f"  - evidence ({item.method}): `{item.evidence}`")
        lines.append("")
    if not verdict.objective_sense_matches:
        lines.extend(
            [
                "## Objective direction",
                "",
                f"The draft declares `{verdict.observed_objective_sense}` where the problem calls "
                f"for `{verdict.required_objective_sense}`.",
                "",
            ]
        )
    lines.append("## What this check does not establish")
    lines.append("")
    lines.append(
        "- That the requirement list itself is complete. A condition nobody wrote down was not looked for."
    )
    lines.append(
        "- That an encoded requirement is encoded *correctly*. This finds absence, not error."
    )
    lines.extend(f"- {note}" for note in verdict.notes)
    return "\n".join(lines) + "\n"
