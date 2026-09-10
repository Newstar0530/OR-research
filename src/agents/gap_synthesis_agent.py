"""What the local literature actually supports, and where the gap really is.

This file used to end with three fixed sections -- candidate research gaps,
innovation points, testable hypotheses -- written about supply-chain chaos,
bullwhip ratios and multi-echelon replenishment. They were emitted verbatim for
every research goal. A study of bilevel-to-QUBO transformation received a
"research gap" about chaos precursor indicators, with its own goal pasted at
the top to make it look bespoke.

That is the most damaging thing a research assistant can do. A missing section
sends you to look for one; a fabricated one sends you to write a thesis around
it.

So the fixed sections are gone. What remains is what this stage can honestly
produce: an account of what the indexed literature actually contains, scored
against terms taken from this study, and -- when a model is available -- gaps
it was actually asked for. When no model is available, the section says so and
names what a real answer would take. A keyword count over filenames cannot
find a research gap, and pretending otherwise is the defect this file exists to
avoid repeating.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from src.core.artifact_provenance import ContentSource, declared_gap
from src.llm_client import LLMClient
from src.llm_errors import LLMCallError


GAP_SYSTEM_PROMPT = (
    "You are a skeptical Operations Research researcher identifying genuine research gaps. "
    "You never claim a gap exists on the basis of a filename or a keyword count. When the "
    "evidence provided is too thin to support a gap, you say so plainly instead of inventing one."
)


class GapSynthesisAgent:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    @staticmethod
    def _relevance_terms(research_goal: str, domain_profile: dict) -> list[str]:
        """Terms that make a local document relevant *to this study*."""

        terms = {str(domain_profile.get("name", "")).lower()}
        for key in ("keywords", "metrics", "algorithm_families", "sensitivity_parameters"):
            for item in domain_profile.get(key, []) or []:
                terms.add(str(item).lower())
        for word in re.findall(r"[A-Za-z][A-Za-z0-9_-]{3,}", research_goal.lower()):
            terms.add(word)
        return sorted(term for term in terms if len(term) >= 4)

    @staticmethod
    def _load_papers(literature_index: Path | None, terms: list[str]) -> list[dict]:
        if not literature_index or not literature_index.exists():
            return []
        try:
            df = pd.read_csv(literature_index)
        except (pd.errors.EmptyDataError, pd.errors.ParserError):
            # An unreadable or empty index means "no local literature", not
            # "abandon the run". This used to raise and kill the pipeline.
            return []
        if df.empty or "keyword_hits" not in df.columns:
            return []
        df["score"] = df["keyword_hits"].fillna("").apply(
            lambda hits: sum(term in str(hits).lower() for term in terms)
        )
        return df.sort_values(["score", "filename"], ascending=[False, True]).head(12).to_dict("records")

    def run(
        self,
        research_goal: str,
        domain_profile: dict,
        literature_index: Path | None = None,
    ) -> tuple[str, ContentSource, list[str]]:
        """Returns (markdown, content_source, inputs_used).

        The caller records the source in the provenance ledger, so a run can be
        asked how much of its own report was actually about it.
        """

        profile_name = domain_profile.get("name", "unknown")
        terms = self._relevance_terms(research_goal, domain_profile)
        papers = self._load_papers(literature_index, terms)
        relevant = [paper for paper in papers if int(paper.get("score", 0)) > 0]

        lines = [
            "# Research Gap Analysis",
            "",
            "## Research Goal",
            research_goal,
            "",
            "## Selected Domain",
            profile_name,
            "",
            "## What The Local Literature Index Actually Contains",
            "",
        ]
        inputs_used = ["research_goal", "domain_profile"]
        if papers:
            inputs_used.append("literature_index")
            lines.append(
                f"{len(papers)} indexed document(s), scored against {len(terms)} term(s) taken from"
                " this study's goal and domain profile. The score counts keyword overlap only."
            )
            lines.append("")
            lines.append("| document | matched terms | relevance score |")
            lines.append("| --- | --- | --- |")
            for paper in papers:
                hits = str(paper.get("keyword_hits") or "").strip() or "none"
                lines.append(f"| `{paper.get('filename')}` | {hits} | {paper.get('score', 0)} |")
            lines.append("")
            if not relevant:
                lines.append(
                    "**No indexed document shares a single term with this study.** Either the"
                    " literature folder holds work on a different topic, or the extractor did not"
                    " read enough of each file to tell -- it reads the first three pages and takes"
                    " the filename as the title."
                )
                lines.append("")
        else:
            lines.append(
                "No local literature was indexed for this run, so there is no evidence here about"
                " what has already been done."
            )
            lines.append("")

        gap_section, source = self._gap_section(research_goal, profile_name, relevant)
        lines.append(gap_section)
        lines.extend(
            [
                "",
                "## Required Human Verification",
                "- A keyword overlap is not a claim about content. Confirm any document listed"
                " above is really about this problem before citing it.",
                "- The index stores a filename-derived title, a page count and the first three"
                " pages. It holds no authors, no year and no abstract, so it cannot establish"
                " what a paper concluded.",
            ]
        )
        return "\n".join(lines) + "\n", source, inputs_used

    def _gap_section(
        self, research_goal: str, profile_name: str, relevant: list[dict]
    ) -> tuple[str, ContentSource]:
        """Ask a model for gaps, or declare that none were produced."""

        header = "## Candidate Research Gaps\n\n"
        if not self.llm.use_mock:
            evidence = "\n".join(
                f"- {paper.get('filename')}: {str(paper.get('snippet') or '')[:300]}"
                for paper in relevant[:8]
            ) or "No indexed document matched this study's terms."
            try:
                payload = self.llm.chat_json(
                    GAP_SYSTEM_PROMPT,
                    (
                        "Identify research gaps for this study. Return JSON with keys `gaps`, "
                        "`innovation_points` and `hypotheses`, each a list of strings. If the "
                        "evidence below is too thin to support a gap, return empty lists and put "
                        "your reason in a `refusal` key -- an empty answer is the correct answer "
                        "when the evidence is absent.\n\n"
                        f"Research goal: {research_goal}\n"
                        f"Domain: {profile_name}\n"
                        f"Indexed evidence:\n{evidence}"
                    ),
                )
                gaps = [str(item) for item in payload.get("gaps", []) if str(item).strip()]
                if gaps:
                    return header + self._render_llm_gaps(payload, gaps), "llm"
                refusal = str(payload.get("refusal") or "").strip()
                return (
                    header
                    + declared_gap(
                        "Candidate research gaps",
                        "The model was asked and declined to name a gap from the available evidence."
                        + (f" It said: {refusal}" if refusal else ""),
                        [
                            "literature with enough extracted content to compare against",
                            "or a researcher who has read the indexed papers",
                        ],
                    ),
                    "llm",
                )
            except Exception as exc:
                # A permanent failure -- a bad key, a wrong model name -- will
                # fail identically at every later stage. Degrading here would
                # hide the cause once per stage instead of naming it once.
                if isinstance(exc, LLMCallError) and exc.is_permanent:
                    raise
                reason = exc.summary() if isinstance(exc, LLMCallError) else str(exc)
                return (
                    header
                    + declared_gap(
                        "Candidate research gaps",
                        f"The language model call failed ({reason}), so no gap was produced.",
                        ["a working LLM configuration", "or a researcher's own reading"],
                    ),
                    "not_generated",
                )

        return (
            header
            + declared_gap(
                "Candidate research gaps, innovation points and testable hypotheses",
                "This run used the mock LLM (`use_mock_llm: true`), so nothing reasoned about the"
                " literature. A research gap cannot be derived from keyword counts over filenames,"
                " and this stage will not manufacture one.",
                [
                    "a configured language model (set `use_mock_llm: false` and an API key), or",
                    "a researcher who has read the indexed papers and can say what they leave open",
                ],
                "Until then, treat this section as empty rather than as a negative result: nothing"
                " here says a gap does not exist.",
            ),
            "not_generated",
        )

    @staticmethod
    def _render_llm_gaps(payload: dict, gaps: list[str]) -> str:
        lines = [f"- {item}" for item in gaps]
        for title, key in (("Candidate Innovation Points", "innovation_points"), ("Testable Hypotheses", "hypotheses")):
            items = [str(item) for item in payload.get(key, []) if str(item).strip()]
            lines.append("")
            lines.append(f"## {title}")
            lines.append("")
            lines.extend(f"- {item}" for item in items or ["None were proposed."])
        lines.append("")
        lines.append(
            "These were written by a language model from the evidence listed above. They are"
            " hypotheses to check, not established gaps."
        )
        return "\n".join(lines)
