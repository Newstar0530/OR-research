from __future__ import annotations

from src.llm_client import LLMClient
from src.schemas import NoveltyReport, ResearchIdea


class NoveltyAgent:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def run(self, idea: ResearchIdea, api_configured: bool = False, local_literature: list[dict] | None = None) -> NoveltyReport:
        queries = [
            f'"{idea.proposed_algorithm_type}" "{idea.proposed_model_type}"',
            f'"{idea.title}" operations research heuristic benchmark',
            "bilevel optimization KKT MPEC Big-M QUBO QAOA",
            "quantum approximate optimization QUBO bilevel programming",
        ]
        works = [] if not api_configured else ["External API integration placeholder - verify source metadata before citation."]
        if local_literature:
            works.extend(
                f"Local file: {item.get('filename')} | hits: {item.get('keyword_hits')}"
                for item in local_literature[:12]
            )
        return NoveltyReport(
            novelty_summary=(
                "No external literature API is configured. The system used local PDF indexing when available "
                "and generated search queries, but it does not claim verified citations."
            ),
            search_queries=queries,
            potentially_related_works=works,
            similarity_risk="medium",
            recommendation="revise",
            revision_suggestions=[
                "Have a researcher search the generated queries in scholarly databases.",
                "Position the contribution as a reproducible template demonstration unless a verified research gap is found.",
            ],
        )
