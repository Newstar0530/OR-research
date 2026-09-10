"""One inspection of a model draft, used both to judge it and to accept it.

`CriticAgent` reports what is wrong with `model_draft.md`. `ModelingAgent`, once
it can ask a language model to write one, needs to decide whether what came back
is a formulation or another template. Those are the same question, so they are
the same code: the draft a model produces is accepted only if it would survive
the critique that is about to be written about it.

Keeping them separate would let the two drift until the modeling stage accepted
a draft the critic then called a scaffold -- which is the state this project
was already in, with the difference that nothing was checking at all.

Nothing here judges whether the model is *right*. It checks whether the
document is specific enough to be judged: whether the scaffold's own symbols
are gone, whether the sections exist, whether a single inequality or summation
was written down. A formulation nobody can implement cannot be falsified, and
an unfalsifiable model is not a research contribution.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, Field


#: Text that appears only in the untouched scaffold. Finding it means the
#: section was never specialised to this problem.
PLACEHOLDER_MARKERS: tuple[tuple[str, str], ...] = (
    ("F(x, y, z; c, d, theta)", "the objective is still the generic symbol `F(x, y, z; c, d, theta)`"),
    ("I: set of decision entities", "the sets are still the generic `I / T / S / A`"),
    ("c_i: cost, weight, processing time, or penalty", "the parameters are still the generic `c_i / d_t / u_i / theta`"),
    ("x_i: primary decision variable", "the decision variables are still the generic `x_i / y_t / z_s`"),
    ("resource/capacity constraints", "the constraints are still a list of constraint *families*, not constraints"),
    ("minimize or maximize", "the objective direction has not been chosen"),
)

#: Sections a usable formulation must have at all.
REQUIRED_SECTIONS: tuple[str, ...] = (
    "Sets and Indices",
    "Parameters",
    "Decision Variables",
    "Objective Function",
    "Constraints",
    "Assumptions",
)

#: Evidence that some mathematics was actually written, rather than described.
MATH_HINT = re.compile(
    r"(\\sum|\\sum_|sum_|sum\s*_|<=|>=|≤|≥|\\forall|for all|∀|s\.t\.|subject to|∈|\\in\b)",
    re.IGNORECASE,
)
_DIRECTION = re.compile(r"^\s*(minimi[sz]e|maximi[sz]e)\b", re.IGNORECASE | re.MULTILINE)


class DraftInspection(BaseModel):
    """What a model draft does and does not contain."""

    #: Scaffold elements still present, described in the reader's terms.
    placeholders: list[str] = Field(default_factory=list)
    missing_sections: list[str] = Field(default_factory=list)
    has_math: bool = False
    declares_direction: bool = False
    unresolved_markers: int = 0
    is_empty: bool = True
    checked_elements: int = len(PLACEHOLDER_MARKERS)

    @property
    def is_specific(self) -> bool:
        """True when the draft is about a problem rather than about modelling.

        Deliberately strict: every scaffold marker gone, every section present,
        real mathematics somewhere, and a chosen objective direction. A draft
        that fails any of these cannot be implemented as written.
        """

        return bool(
            not self.is_empty
            and not self.placeholders
            and not self.missing_sections
            and self.has_math
            and self.declares_direction
        )

    def failures(self) -> list[str]:
        """Why the draft is not yet specific, phrased to be handed to a writer."""

        if self.is_empty:
            return ["The document is empty."]
        reasons: list[str] = []
        if self.missing_sections:
            reasons.append(
                "Add these missing sections: " + ", ".join(f"## {name}" for name in self.missing_sections) + "."
            )
        if self.placeholders:
            reasons.append(
                "Replace the scaffold placeholders with this problem's own notation: "
                + "; ".join(self.placeholders)
                + "."
            )
        if not self.has_math:
            reasons.append(
                "Write the objective and at least the binding constraints as actual expressions. "
                "No summation, inequality or quantifier appears anywhere in the draft."
            )
        if not self.declares_direction:
            reasons.append(
                "State the objective direction: a line beginning `minimize` or `maximize`, not "
                "`minimize or maximize`."
            )
        return reasons


def inspect_model_draft(text: str) -> DraftInspection:
    """Check a draft against the scaffold it must no longer resemble."""

    body = text or ""
    return DraftInspection(
        placeholders=[reason for marker, reason in PLACEHOLDER_MARKERS if marker in body],
        missing_sections=[name for name in REQUIRED_SECTIONS if f"## {name}" not in body],
        has_math=bool(MATH_HINT.search(body)),
        # `minimize or maximize` is the scaffold's non-choice, so a line that
        # merely contains both words does not count as declaring one.
        declares_direction=bool(_DIRECTION.search(body)) and "minimize or maximize" not in body,
        unresolved_markers=body.count("Needs human verification"),
        is_empty=not body.strip(),
    )
