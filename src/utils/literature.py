from __future__ import annotations

import re
import zipfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

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
    path: str
    filename: str
    category: str
    pages: int | None
    title_guess: str
    keyword_hits: str
    snippet: str


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _extract_pdf_text(path: Path, max_pages: int = 3) -> tuple[int | None, str]:
    if PdfReader is None:
        return None, ""
    try:
        reader = PdfReader(str(path))
        pages = len(reader.pages)
        chunks = [(page.extract_text() or "") for page in reader.pages[:max_pages]]
        return pages, _clean_text(" ".join(chunks))
    except Exception:
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
    documents = list(root.rglob("*.pdf")) + list(root.rglob("*.docx"))
    for path in sorted(documents):
        if path.suffix.lower() == ".pdf":
            pages, text = _extract_pdf_text(path)
        else:
            pages, text = _extract_docx_text(path)
        title_guess = path.stem.replace("_", " ")
        combined = f"{title_guess} {text}"
        hits = _keyword_hits(combined)
        records.append(
            LiteratureRecord(
                path=str(path),
                filename=path.name,
                category=_category_from_hits(hits),
                pages=pages,
                title_guess=title_guess,
                keyword_hits=", ".join(hits),
                snippet=text[:900],
            )
        )
    df = pd.DataFrame([record.__dict__ for record in records])
    index_path = output / "literature_index.csv"
    df.to_csv(index_path, index=False, encoding="utf-8-sig")
    review_path = output / "literature_review.md"
    review_path.write_text(render_literature_review(df), encoding="utf-8")
    return index_path, review_path


def render_literature_review(df: pd.DataFrame) -> str:
    if df.empty:
        return "# Local Literature Review\n\nNo PDF or DOCX files found.\n"
    lines = ["# Local Literature Review", ""]
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
        lines.append(f"### {row.title_guess}")
        lines.append(f"- File: `{row.filename}`")
        lines.append(f"- Category: {row.category}")
        lines.append(f"- Pages: {row.pages}")
        lines.append(f"- Keyword hits: {row.keyword_hits or 'none detected'}")
        if row.snippet:
            lines.append(f"- Snippet: {row.snippet[:300]}")
        lines.append("")
    lines.append("## Research Positioning Notes")
    lines.append("- Treat extracted PDF/DOCX text as a starting point, not verified citation metadata.")
    lines.append("- Use this index to form search queries and novelty risks, not as proof of novelty.")
    lines.append("- Human verification is required for every citation, mathematical claim, and contribution claim.")
    return "\n".join(lines) + "\n"

