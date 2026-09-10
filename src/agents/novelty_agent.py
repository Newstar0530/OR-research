"""Search queries built from this idea, and no claim of having searched.

The queries used to be four fixed strings, two of which named bilevel
optimization, KKT, MPEC, Big-M, QUBO and QAOA -- emitted for every study,
including ones about inventory or scheduling. A researcher handed those queries
would reasonably assume the system had decided they were relevant.

Now they come from the idea's own words. The rest of this agent's honesty was
already right and is kept: it does not claim to have searched anything, and it
does not present the local file list as related work.
"""

from __future__ import annotations

import re

from src.core.artifact_provenance import ContentSource
from src.llm_client import LLMClient
from src.schemas import NoveltyReport, ResearchIdea


_STOPWORDS = frozenset(
    "a an and are as at be by for from has have in into is it its of on or that the their "
    "to was were with using use used study research problem method methods approach based "
    "generic inspectable reproducible workflow explicit".split()
)


def _phrases(idea: ResearchIdea) -> list[str]:
    """Content words from the idea, longest-first, for query building."""

    text = " ".join(
        [idea.title, idea.proposed_model_type, idea.proposed_algorithm_type, idea.core_hypothesis]
    ).lower()
    words = [w for w in re.findall(r"[a-z][a-z0-9-]{2,}", text) if w not in _STOPWORDS]
    seen: list[str] = []
    for word in words:
        if word not in seen:
            seen.append(word)
    return seen


class NoveltyAgent:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def run(
        self,
        idea: ResearchIdea,
        api_configured: bool = False,
        local_literature: list[dict] | None = None,
    ) -> tuple[NoveltyReport, ContentSource]:
        words = _phrases(idea)
        queries = [
            f'"{idea.proposed_model_type}" "{idea.proposed_algorithm_type}"',
            f'"{idea.proposed_model_type}" benchmark comparison',
            " ".join(words[:6]) if words else idea.title,
            (" ".join(words[:4]) + " survey") if words else f"{idea.title} survey",
        ]
        queries = [query for index, query in enumerate(queries) if query.strip() and query not in queries[:index]]

        works = [] if not api_configured else ["External API integration placeholder - verify source metadata before citation."]
        if local_literature:
            works.extend(
                f"Local file: {item.get('filename')} | matched terms: {item.get('keyword_hits')}"
                for item in local_literature[:12]
            )
        return (
            NoveltyReport(
                novelty_summary=(
                    "NOT ASSESSED. No external literature API was queried, so nothing here "
                    "establishes whether this idea is novel. The queries below were built from "
                    "the idea's own wording for a human to run; the files listed, if any, matched "
                    "on keywords only and are not known to be related work."
                ),
                search_queries=queries,
                potentially_related_works=works,
                similarity_risk="unknown",
                recommendation="revise",
                revision_suggestions=[
                    "Run the generated queries in Scopus, Web of Science or Google Scholar.",
                    "Do not claim novelty until a human has checked what already exists.",
                    "Until then, position the contribution as a reproducible demonstration.",
                ],
            ),
            "derived",
        )
