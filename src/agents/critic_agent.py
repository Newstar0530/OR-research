"""Criticise the model draft that was actually produced.

This agent used to ignore `model_markdown` and return a fixed list whose first
item read "The formulation is still a scaffold". That happened to be true --
`ModelingAgent` emits a scaffold -- but it was true by coincidence, not by
inspection. It would have said the same thing about a finished, correct model.

A critique that cannot be wrong is not a critique. So this now reads the draft
and reports what it finds: which sections still hold the generic placeholders,
which are missing entirely, and which are populated. That is a real check, it
needs no language model, and it fails loudly the day the modeling stage starts
producing something specific -- which is exactly when a stale constant would
have become dangerous.
"""

from __future__ import annotations

from src.core.artifact_provenance import ContentSource
from src.core.model_draft_inspection import PLACEHOLDER_MARKERS, inspect_model_draft
from src.core.requirement_coverage import CoverageVerdict
from src.llm_client import LLMClient
from src.schemas import ModelCritique


class CriticAgent:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def run(
        self, model_markdown: str, coverage: CoverageVerdict | None = None
    ) -> tuple[ModelCritique, ContentSource]:
        text = model_markdown or ""
        # The same inspection the modeling stage uses to decide whether a draft
        # is worth keeping, so the two can never disagree about what a scaffold is.
        inspection = inspect_model_draft(text)
        placeholders = inspection.placeholders
        missing = inspection.missing_sections
        unresolved = inspection.unresolved_markers

        critical: list[str] = []
        if inspection.is_empty:
            critical.append("There is no model draft to criticise: the document is empty.")
        if missing:
            critical.append(
                "The draft has no section for: " + ", ".join(missing) + "."
            )
        if placeholders:
            critical.append(
                f"{len(placeholders)} of {inspection.checked_elements} checked elements are still the "
                "untouched scaffold, so this is a template rather than a formulation of this "
                "problem: " + "; ".join(placeholders) + "."
            )
        if not inspection.has_math and not inspection.is_empty:
            critical.append(
                "No summation, inequality or quantifier appears anywhere in the draft. A "
                "formulation without a single written constraint cannot be checked, implemented "
                "or falsified."
            )
        if not inspection.declares_direction and not inspection.is_empty and not placeholders:
            critical.append(
                "The objective direction is not stated. Minimising and maximising the same "
                "expression are different problems."
            )
        if coverage is not None and coverage.checked:
            if coverage.missing:
                critical.append(
                    f"{len(coverage.missing)} of {len(coverage.matches)} stated requirements are "
                    "not imposed by any constraint in the draft. A model missing a requirement is "
                    "optimal over too large a feasible region, which reads as a better objective "
                    "rather than as an error: "
                    + "; ".join(item.requirement_text for item in coverage.missing)
                    + "."
                )
            if not coverage.objective_sense_matches:
                critical.append(
                    f"The draft declares a `{coverage.observed_objective_sense}` objective where the "
                    f"problem calls for `{coverage.required_objective_sense}`."
                )
        if not critical:
            critical.append(
                "No scaffold placeholder was detected. Every element below still needs a human "
                "to verify it is the right model for the problem -- this check confirms the draft "
                "is specific, not that it is correct."
            )

        minor: list[str] = []
        if unresolved:
            minor.append(f"{unresolved} `Needs human verification` marker(s) are still unresolved.")
        if placeholders and len(placeholders) < inspection.checked_elements:
            populated = inspection.checked_elements - len(placeholders)
            minor.append(f"{populated} element(s) have been specialised; the rest have not.")
        if not minor:
            minor.append("No minor structural issue was detected by the automated check.")

        critique = ModelCritique(
            critical_issues=critical,
            minor_issues=minor,
            missing_constraints=self._missing_constraints(text, placeholders, coverage),
            questionable_assumptions=self._assumptions(text),
            suggested_revisions=self._revisions(placeholders, missing),
            human_verification_checklist=[
                "Check every index, set and parameter against the real system.",
                "Confirm the objective direction matches the research question.",
                "Confirm the constraints are sufficient for feasibility, not merely plausible.",
                "Confirm the baselines are fair and strong enough that beating them means something.",
                "State what result would falsify the hypothesis.",
            ],
        )
        # `derived`: computed from the draft by the rules above, not by a model
        # and not from a constant.
        return critique, "derived"

    @staticmethod
    def _missing_constraints(
        text: str, placeholders: list[str], coverage: CoverageVerdict | None = None
    ) -> list[str]:
        if not text.strip():
            return ["Everything: there is no draft."]
        if coverage is not None and coverage.checked:
            if coverage.missing:
                return [
                    f"{item.requirement_text} -- {item.evidence}" for item in coverage.missing
                ]
            return [
                f"None of the {len(coverage.matches)} stated requirements is unimposed. This is "
                "completeness against the requirement list, not completeness in general: a "
                "condition nobody wrote down was not looked for."
            ]
        if placeholders:
            return [
                "None can be identified, because the constraint section lists families "
                "(`resource/capacity`, `flow/balance`, ...) rather than constraints. Nothing here "
                "can be checked for completeness until they are written down."
            ]
        return [
            "No requirement list was available for this run, so the constraint set was compared "
            "against nothing. Completeness is unchecked, not confirmed."
        ]

    @staticmethod
    def _assumptions(text: str) -> list[str]:
        found = []
        if "synthetic data" in text.lower():
            found.append(
                "The draft states it may use synthetic data. Synthetic instances can be made "
                "arbitrarily favourable to the proposed method; the generator needs its own review."
            )
        if "provisional" in text.lower() or "scaffold" in text.lower():
            found.append("The draft describes its own notation as provisional.")
        return found or [
            "No assumption was flagged by the automated check. That is not the same as there "
            "being none."
        ]

    @staticmethod
    def _revisions(placeholders: list[str], missing: list[str]) -> list[str]:
        revisions: list[str] = []
        if missing:
            revisions.append("Add the missing sections: " + ", ".join(missing) + ".")
        if placeholders:
            revisions.append(
                "Replace the generic notation with this problem's own sets, parameters and "
                "variables before the draft is used for anything."
            )
            revisions.append(
                "Write the objective and at least the binding constraints explicitly, so they "
                "can be implemented and checked."
            )
        revisions.append("Define benchmark generation, data sources and the comparison rules.")
        revisions.append("Declare what would falsify the hypothesis.")
        return revisions
