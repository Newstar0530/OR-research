"""An automated review, or an honest statement that none happened.

This agent used to ignore `final_report` entirely and return a fixed verdict:
`overall_score=6`, `novelty_score=4`, `recommendation="borderline"`, every run,
whatever the report said. Those constants did not stay in the review file --
`overall_score` was written into the cross-run memory record and read by the
readiness assessment, so a number that had never looked at anything was being
carried forward as a measurement.

Now a score exists only when a model produced it. Without one, the review
reports what can actually be checked about the report text -- which sections it
has, how long it is -- says plainly that no review was performed, and leaves
every score absent rather than defaulting.
"""

from __future__ import annotations

from src.core.artifact_provenance import ContentSource
from src.llm_client import LLMClient
from src.llm_errors import LLMCallError
from src.schemas import AutomatedReview


REVIEW_SYSTEM_PROMPT = (
    "You are a demanding but fair reviewer for an Operations Research venue. You judge only "
    "what the report actually demonstrates. An improvement that is not statistically supported "
    "is not a result, and a reproducible pipeline is not by itself a contribution."
)

#: Sections a finished OR report is expected to carry. Their absence is a fact
#: about the report, checkable without any model.
EXPECTED_SECTIONS: tuple[tuple[str, str], ...] = (
    ("result", "results"),
    ("method", "methodology"),
    ("limitation", "limitations"),
    ("baseline", "baseline comparison"),
    ("statistic", "statistical evidence"),
)


class ReviewerAgent:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def run(self, final_report: str) -> tuple[AutomatedReview, ContentSource]:
        text = final_report or ""
        failure = ""
        if not self.llm.use_mock:
            review, failure = self._llm_review(text)
            if review is not None:
                return review, "llm"
        return self._unreviewed(text, failure), "not_generated"

    def _llm_review(self, text: str) -> tuple[AutomatedReview | None, str]:
        try:
            payload = self.llm.chat_json(
                REVIEW_SYSTEM_PROMPT,
                (
                    "Review this research report. Return JSON with keys: summary, strengths, "
                    "weaknesses, questions_for_authors, soundness_score, novelty_score, "
                    "technical_quality_score, reproducibility_score, presentation_score, "
                    "overall_score, confidence_score (each an integer 1-10), recommendation "
                    "(one of accept, minor_revision, major_revision, reject, borderline), and "
                    "required_revision_checklist.\n\n"
                    f"Report:\n{text[:12000]}"
                ),
            )
            return AutomatedReview(review_performed=True, **payload), ""
        except LLMCallError as error:
            if error.is_permanent:
                raise
            # A failed review is not a review. Falling through to the unreviewed
            # path is the honest outcome -- but the reason must travel with it,
            # or the report says "no model was configured" when one was.
            return None, error.summary()
        except Exception as exc:
            return None, f"the review response could not be used: {exc}"

    def _unreviewed(self, text: str, failure: str = "") -> AutomatedReview:
        """Report what is checkable about the text, and score nothing."""

        lowered = text.lower()
        missing = [label for token, label in EXPECTED_SECTIONS if token not in lowered]
        present = [label for token, label in EXPECTED_SECTIONS if token in lowered]
        weaknesses = [
            f"No automated review was performed: {failure}. The scores below are absent rather "
            "than defaulted."
            if failure
            else "No automated review was performed: this run used the mock LLM, so nothing read "
            "the report. The scores below are absent rather than defaulted.",
        ]
        if missing:
            weaknesses.append(
                "The report text does not mention: " + ", ".join(missing) + "."
            )
        if len(text.strip()) < 400:
            weaknesses.append(
                f"The report is {len(text.strip())} characters long, which is too short to "
                "support a research claim."
            )
        return AutomatedReview(
            summary=(
                "NOT REVIEWED. "
                + (
                    f"The review call failed ({failure}), so no judgement was formed about the "
                    "report."
                    if failure
                    else "No language model was configured for this run, so no judgement was "
                    "formed about the report."
                )
                + " What follows is a structural check of the text only. Absence of a score here"
                " means no review happened -- it does not mean the work scored badly, and it does"
                " not mean it scored well."
            ),
            strengths=[
                f"Sections detected in the report text: {', '.join(present)}." if present
                else "No expected section headings were detected in the report text.",
            ],
            weaknesses=weaknesses,
            questions_for_authors=[
                "Which verified papers establish the baseline landscape for this problem?",
                "Does the statistical evidence license the comparative claims the report makes?",
                "Are the baselines strong enough that beating them means something?",
            ],
            recommendation="borderline",
            review_performed=False,
            required_revision_checklist=[
                "Configure a real LLM (`use_mock_llm: false`) or have a human review the report.",
                "Run a real literature review before making any novelty claim.",
                "Confirm every comparative claim against `statistical_evidence.md`.",
            ],
        )
