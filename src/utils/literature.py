from __future__ import annotations

import re
import zipfile
from dataclasses import dataclass, fields
from pathlib import Path

import pandas as pd

from src.utils.bibliography import (
    BibliographicRecord,
    corpus_headline,
    extract_bibliography,
    read_pdf,
)

try:
    from pypdf import PdfReader
except Exception:  # pragma: no cover - optional runtime dependency
    PdfReader = None


KEYWORDS = {
    "optimization": ["optimization", "programming", "integer", "linear", "nonlinear", "robust"],
    "scheduling": ["scheduling", "job shop", "flow shop", "makespan", "tardiness", "dispatching"],
    "inventory": ["inventory", "newsvendor", "reorder", "safety stock", "stockout", "shortage"],
    "network_analysis": ["network", "graph", "routing", "flow", "reliability", "resilience", "centrality"],
    "stochastic_models": ["stochastic", "uncertainty", "simulation", "queue", "queueing", "monte carlo"],
    "decision_analysis": ["decision", "multi-criteria", "mcdm", "ahp", "preference", "utility"],
    "heuristics_metaheuristics": ["heuristic", "metaheuristic", "local search", "tabu", "annealing", "genetic"],
    "ai_for_ie": ["machine learning", "reinforcement learning", "prediction", "neural", "industrial ai"],
    "modeling_terms": ["objective", "constraint", "decision variable", "feasible", "capacity"],
    "evaluation_terms": ["benchmark", "baseline", "sensitivity", "ablation", "statistical", "comparison"],
}


@dataclass
class LiteratureRecord:
    """One indexed document.

    Every metadata field is paired with a `*_source` column. A title read out
    of the document and a title taken from the file name are different kinds of
    thing, and collapsing them into one column is how a file name ends up in a
    bibliography.
    """

    path: str
    filename: str
    category: str
    pages: int | None
    title: str
    title_source: str
    authors: str
    authors_source: str
    year: str
    year_source: str
    doi: str
    abstract: str
    #: True when a human could check this record against the published paper:
    #: a DOI, or a title and year that were actually read from the document.
    verifiable: bool
    keyword_hits: str
    snippet: str
    #: Kept so older tooling that reads `title_guess` still works. It holds the
    #: same value as `title`, whatever that value's provenance.
    title_guess: str


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _extract_pdf_text(path: Path, max_pages: int = 6) -> tuple[int | None, str]:
    """Kept for callers that only want the text. See `read_pdf` for metadata."""

    pages, text, _, _ = read_pdf(path, max_pages=max_pages)
    return pages, text


#: Extensions the indexer understands. Anything else is listed as unreadable
#: rather than silently ignored: a corpus the user provided and the system
#: skipped without saying so is worse than one it refuses.
SUPPORTED_SUFFIXES = (".pdf", ".docx", ".txt", ".md")


def _extract_plain_text(path: Path) -> tuple[int | None, str]:
    try:
        return None, _clean_text(path.read_text(encoding="utf-8", errors="ignore"))
    except OSError:
        return None, ""


def _extract_docx_text(path: Path) -> tuple[int | None, str]:
    try:
        with zipfile.ZipFile(path) as zf:
            xml = zf.read("word/document.xml").decode("utf-8", errors="ignore")
        text = re.sub(r"<[^>]+>", " ", xml)
        return None, _clean_text(text)
    except Exception:
        return None, ""


def _keyword_hits(text: str) -> list[str]:
    lowered = text.lower()
    hits = []
    for label, words in KEYWORDS.items():
        if any(word.lower() in lowered for word in words):
            hits.append(label)
    return hits


def _category_from_hits(hits: list[str]) -> str:
    for category in [
        "optimization",
        "scheduling",
        "inventory",
        "network_analysis",
        "stochastic_models",
        "decision_analysis",
        "heuristics_metaheuristics",
        "ai_for_ie",
    ]:
        if category in hits:
            return category
    return "other"


def build_literature_index(literature_dir: str | Path, output_dir: str | Path) -> tuple[Path, Path]:
    root = Path(literature_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    records: list[LiteratureRecord] = []
    documents = (
        [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES]
        if root.exists()
        else []
    )
    bibliographies: list[BibliographicRecord] = []
    for path in sorted(documents):
        suffix = path.suffix.lower()
        first_lines: list[str] = []
        metadata: dict = {}
        if suffix == ".pdf":
            pages, text, first_lines, metadata = read_pdf(path)
        elif suffix == ".docx":
            pages, text = _extract_docx_text(path)
            first_lines = text.split(". ")[:25]
        else:
            pages, text = _extract_plain_text(path)
            try:
                first_lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()[:40]
            except OSError:
                first_lines = []

        bibliography = extract_bibliography(text, first_lines, metadata, path.stem)
        bibliographies.append(bibliography)
        combined = f"{bibliography.title.value} {text}"
        hits = _keyword_hits(combined)
        records.append(
            LiteratureRecord(
                path=str(path),
                filename=path.name,
                category=_category_from_hits(hits),
                pages=pages,
                title=bibliography.title.value,
                title_source=bibliography.title.source,
                authors=bibliography.authors.value,
                authors_source=bibliography.authors.source,
                year=bibliography.year.value,
                year_source=bibliography.year.source,
                doi=bibliography.doi.value,
                abstract=bibliography.abstract.value,
                verifiable=bibliography.is_verifiable,
                keyword_hits=", ".join(hits),
                snippet=text[:900],
                title_guess=bibliography.title.value,
            )
        )
    # An empty corpus must still produce a table with a header. A zero-column CSV
    # is unreadable by pandas, and used to abort the whole run downstream.
    columns = [field.name for field in fields(LiteratureRecord)]
    df = pd.DataFrame([record.__dict__ for record in records], columns=columns)
    index_path = output / "literature_index.csv"
    df.to_csv(index_path, index=False, encoding="utf-8-sig")
    review_path = output / "literature_review.md"
    review_path.write_text(render_literature_review(df, bibliographies), encoding="utf-8")
    return index_path, review_path


def render_literature_review(
    df: pd.DataFrame, bibliographies: list[BibliographicRecord] | None = None
) -> str:
    if df.empty:
        return (
            "# Local Literature Review\n\n"
            "No readable documents were found. Supported formats: "
            + ", ".join(SUPPORTED_SUFFIXES)
            + ".\n"
        )
    lines = ["# Local Literature Review", ""]
    if bibliographies:
        # The first thing a reader needs is how much of this corpus is actually
        # identified, rather than how many files were seen.
        lines.append(corpus_headline(bibliographies))
        lines.append("")
    lines.append("## Corpus Summary")
    lines.append(f"- files indexed: {len(df)}")
    for category, count in df["category"].value_counts().items():
        lines.append(f"- {category}: {count}")
    lines.append("")
    lines.append("## Most Relevant Local Documents")
    relevance_terms = list(KEYWORDS)
    df = df.copy()
    df["relevance_score"] = df["keyword_hits"].fillna("").apply(lambda text: sum(term in text for term in relevance_terms))
    relevant = df.sort_values(["relevance_score", "filename"], ascending=[False, True]).head(20)
    for row in relevant.itertuples(index=False):
        title = getattr(row, "title", "") or getattr(row, "title_guess", "")
        source = getattr(row, "title_source", "")
        suffix = " *(from the file name, not the document)*" if source == "filename" else ""
        lines.append(f"### {title}{suffix}")
        lines.append(f"- File: `{row.filename}`")
        authors = getattr(row, "authors", "")
        year = getattr(row, "year", "")
        doi = getattr(row, "doi", "")
        lines.append(f"- Authors: {authors or 'not found'}")
        lines.append(f"- Year: {year or 'not found'}")
        lines.append(f"- DOI: {doi or 'not found'}")
        lines.append(f"- Verifiable against the published record: {bool(getattr(row, 'verifiable', False))}")
        lines.append(f"- Category: {row.category}")
        lines.append(f"- Pages: {row.pages}")
        lines.append(f"- Keyword hits: {row.keyword_hits or 'none detected'}")
        abstract = getattr(row, "abstract", "")
        if abstract:
            lines.append(f"- Abstract: {str(abstract)[:600]}")
        elif row.snippet:
            lines.append(f"- No abstract was extracted. First text found: {row.snippet[:300]}")
        lines.append("")
    lines.append("## Research Positioning Notes")
    lines.append(
        "- A record marked `Verifiable: False` identifies a file, not a paper. Its title came "
        "from the file name, and searching for that string will find your download rather than "
        "the publication."
    )
    lines.append("- Extracted metadata is a heuristic read of page one. Check it before citing.")
    lines.append("- Use this index to form search queries and novelty risks, not as proof of novelty.")
    lines.append("- Human verification is required for every citation, mathematical claim, and contribution claim.")
    return "\n".join(lines) + "\n"

