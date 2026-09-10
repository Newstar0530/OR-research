"""Pull real bibliographic metadata out of a document, and say where each field came from.

The index this replaces stored one thing about each paper that resembled
metadata: `title_guess = filename`. A paper saved as
`1-s2.0-S0377221723004538-main.pdf` had that string as its title, no authors, no
year, no DOI and no abstract. Everything downstream inherited it -- the novelty
report listed it as a "potentially related work", `references.bib` emitted it as
a `@misc` title, and the gap analysis scored it on keyword overlap. None of that
could establish what the paper actually said, because nothing had read it.

Extraction from a PDF is guesswork, and the guesses differ enormously in
quality. A DOI matched by regex is near-certain and makes the whole record
checkable in one click. A title taken from the PDF's own metadata is usually
right. A title inferred from the layout of the first page is a heuristic. A
title taken from the filename is not extraction at all.

Collapsing those four into one `title` column is how a filename ends up in a
bibliography. So every field carries its own source, `is_known` is false for
anything that came from the filename or nowhere, and a record counts as
`verifiable` only when it has a DOI, or a title and a year that were actually
read out of the document.
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


#: Where a single field's value came from, best first.
FieldSource = Literal["doi_regex", "pdf_metadata", "first_page", "filename", "not_found"]

#: Sources that mean the document itself supplied the value.
READ_FROM_DOCUMENT: frozenset[str] = frozenset({"doi_regex", "pdf_metadata", "first_page"})

_DOI = re.compile(r"\b(10\.\d{4,9}/[-._;()/:A-Za-z0-9]+)", re.IGNORECASE)
_YEAR = re.compile(r"\b(19[5-9]\d|20[0-4]\d)\b")
_YEAR_CONTEXT = re.compile(
    r"(?:©|\(c\)|copyright|published|received|accepted|revised)[^.]{0,60}?\b(19[5-9]\d|20[0-4]\d)\b",
    re.IGNORECASE,
)
_ABSTRACT = re.compile(
    r"\babstract\b[:\s.\-]*(.{80,3000}?)(?=\b(?:keywords?|key words|index terms|1\.?\s+introduction|introduction)\b)",
    re.IGNORECASE | re.DOTALL,
)
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
#: Junk PDF producers put in /Title.
_BAD_METADATA_TITLE = re.compile(
    r"^(untitled|microsoft word|document\d*|pii:|doi:|\d+$|.*\.(doc|docx|pdf|tex)$)", re.IGNORECASE
)
#: Lines that belong to a journal header rather than to the paper.
_HEADER_NOISE = re.compile(
    r"(journal|volume|vol\.|issue|no\.|pp\.|issn|isbn|elsevier|springer|wiley|ieee|taylor & francis|"
    r"downloaded from|licen[cs]e|all rights reserved|www\.|http|doi:|©)",
    re.IGNORECASE,
)


class ExtractedField(BaseModel):
    """One metadata field and the honest story of where it came from."""

    value: str = ""
    source: FieldSource = "not_found"

    @property
    def is_known(self) -> bool:
        """True only when the document itself supplied this value."""

        return bool(self.value) and self.source in READ_FROM_DOCUMENT

    def render(self) -> str:
        if not self.value:
            return "not found"
        if self.source == "filename":
            return f"{self.value} (from the file name, not the document)"
        return self.value


class BibliographicRecord(BaseModel):
    title: ExtractedField = Field(default_factory=ExtractedField)
    authors: ExtractedField = Field(default_factory=ExtractedField)
    year: ExtractedField = Field(default_factory=ExtractedField)
    doi: ExtractedField = Field(default_factory=ExtractedField)
    abstract: ExtractedField = Field(default_factory=ExtractedField)

    @property
    def is_verifiable(self) -> bool:
        """Could a human check this record against the real paper?

        A DOI is enough on its own. Otherwise it takes a title and a year that
        were read from the document -- a filename-derived title identifies
        nothing, and searching for it finds the file, not the paper.
        """

        return self.doi.is_known or (self.title.is_known and self.year.is_known)

    def citation_key(self, fallback: str) -> str:
        first_author = self.authors.value.split(",")[0].split()[-1] if self.authors.value else ""
        parts = [part for part in (first_author.lower(), self.year.value) if part]
        if not parts:
            return re.sub(r"[^A-Za-z0-9]+", "_", fallback).strip("_").lower() or "reference"
        return re.sub(r"[^A-Za-z0-9]+", "", "".join(parts)).lower()

    def one_line(self) -> str:
        bits = [self.title.render()]
        if self.authors.value:
            bits.append(self.authors.value)
        if self.year.value:
            bits.append(self.year.value)
        if self.doi.value:
            bits.append(f"doi:{self.doi.value}")
        return " | ".join(bits)


def extract_doi(text: str) -> ExtractedField:
    match = _DOI.search(text or "")
    if not match:
        return ExtractedField()
    # Trailing punctuation is common when a DOI ends a sentence or a line.
    return ExtractedField(value=match.group(1).rstrip(".,;)"), source="doi_regex")


def extract_year(text: str) -> ExtractedField:
    """Prefer a year next to a publication word; fall back to the latest plausible one."""

    contextual = _YEAR_CONTEXT.search(text or "")
    if contextual:
        return ExtractedField(value=contextual.group(1), source="first_page")
    current = datetime.now().year
    years = [int(value) for value in _YEAR.findall(text or "")]
    plausible = [year for year in years if 1950 <= year <= current]
    if not plausible:
        return ExtractedField()
    return ExtractedField(value=str(max(plausible)), source="first_page")


def extract_abstract(text: str, limit: int = 1800) -> ExtractedField:
    match = _ABSTRACT.search(text or "")
    if not match:
        return ExtractedField()
    body = re.sub(r"\s+", " ", match.group(1)).strip()
    if len(body) < 80:
        return ExtractedField()
    return ExtractedField(value=body[:limit], source="first_page")


def _looks_like_title(line: str) -> bool:
    stripped = line.strip()
    if not 20 <= len(stripped) <= 250:
        return False
    if _HEADER_NOISE.search(stripped) or _EMAIL.search(stripped):
        return False
    letters = [c for c in stripped if c.isalpha()]
    if not letters:
        return False
    # A running head is often set in full capitals; a title rarely is.
    if sum(1 for c in letters if c.isupper()) / len(letters) > 0.7:
        return False
    if sum(1 for c in stripped if c.isdigit()) / len(stripped) > 0.2:
        return False
    return len(stripped.split()) >= 3


def _looks_like_authors(line: str) -> bool:
    stripped = line.strip()
    if not 4 <= len(stripped) <= 300 or _HEADER_NOISE.search(stripped):
        return False
    if _EMAIL.search(stripped):
        return False
    # Author lines are names separated by commas or "and", with initials and
    # affiliation markers, and essentially no sentence punctuation.
    if stripped.endswith(".") and " and " not in stripped and "," not in stripped:
        return False
    tokens = [token for token in re.split(r"[,\s]+", stripped) if token]
    if not 2 <= len(tokens) <= 30:
        return False
    capitalised = sum(1 for token in tokens if token[:1].isupper())
    return capitalised >= max(2, int(0.6 * len(tokens)))


def extract_title_and_authors(first_page_lines: list[str]) -> tuple[ExtractedField, ExtractedField]:
    """Read the title, then the author line beneath it, from the top of page one."""

    lines = [line for line in (first_page_lines or []) if line.strip()]
    stop = len(lines)
    for index, line in enumerate(lines):
        if re.match(r"^\s*abstract\b", line, re.IGNORECASE):
            stop = index
            break
    head = lines[: stop or len(lines)][:25]

    title_index = next((index for index, line in enumerate(head) if _looks_like_title(line)), None)
    if title_index is None:
        return ExtractedField(), ExtractedField()
    title = ExtractedField(value=head[title_index].strip(), source="first_page")

    for line in head[title_index + 1 : title_index + 6]:
        if _looks_like_authors(line):
            return title, ExtractedField(value=line.strip(), source="first_page")
    return title, ExtractedField()


def _metadata_title(raw_title: str | None) -> ExtractedField:
    value = (raw_title or "").strip()
    if len(value) < 10 or _BAD_METADATA_TITLE.match(value):
        return ExtractedField()
    return ExtractedField(value=value, source="pdf_metadata")


def _metadata_authors(raw_author: str | None) -> ExtractedField:
    value = (raw_author or "").strip()
    if len(value) < 4 or value.lower() in {"unknown", "author", "administrator", "user"}:
        return ExtractedField()
    return ExtractedField(value=value, source="pdf_metadata")


def extract_bibliography(
    text: str,
    first_page_lines: list[str] | None = None,
    pdf_metadata: dict | None = None,
    filename_stem: str = "",
) -> BibliographicRecord:
    """Best available metadata, each field labelled with how it was obtained.

    The order is by reliability, not by convenience: the PDF's own metadata
    beats a layout heuristic, and a layout heuristic beats the filename. The
    filename is used only so the record has something to display, and it is
    marked so nothing downstream mistakes it for a title.
    """

    metadata = pdf_metadata or {}
    layout_title, layout_authors = extract_title_and_authors(first_page_lines or [])

    title = _metadata_title(str(metadata.get("/Title", "") or ""))
    if not title.value:
        title = layout_title
    if not title.value and filename_stem:
        title = ExtractedField(value=filename_stem.replace("_", " ").strip(), source="filename")

    authors = _metadata_authors(str(metadata.get("/Author", "") or ""))
    if not authors.value:
        authors = layout_authors

    return BibliographicRecord(
        title=title,
        authors=authors,
        year=extract_year(text),
        doi=extract_doi(text),
        abstract=extract_abstract(text),
    )


def corpus_headline(records: list[BibliographicRecord]) -> str:
    """One sentence about how much of this corpus is actually identified."""

    total = len(records)
    if not total:
        return "No documents were indexed."
    with_doi = sum(1 for record in records if record.doi.is_known)
    verifiable = sum(1 for record in records if record.is_verifiable)
    filename_only = sum(1 for record in records if record.title.source == "filename")
    abstracts = sum(1 for record in records if record.abstract.is_known)
    return (
        f"{total} document(s) indexed. {with_doi} carry a DOI and {verifiable} are verifiable"
        f" against the published record; {filename_only} have no title but the file name, so they"
        f" identify a file rather than a paper. {abstracts} abstract(s) were extracted."
    )


def read_pdf(path: str | Path, max_pages: int = 6) -> tuple[int | None, str, list[str], dict]:
    """Page count, cleaned text, first-page lines, and the PDF's own metadata."""

    try:
        from pypdf import PdfReader
    except Exception:  # pragma: no cover - optional runtime dependency
        return None, "", [], {}
    try:
        reader = PdfReader(str(path))
        pages = len(reader.pages)
        first_page_text = reader.pages[0].extract_text() or "" if pages else ""
        chunks = [(page.extract_text() or "") for page in reader.pages[:max_pages]]
        metadata = dict(reader.metadata or {})
    except Exception:
        return None, "", [], {}
    text = re.sub(r"\s+", " ", " ".join(chunks)).strip()
    return pages, text, first_page_text.splitlines(), metadata


def literature_digest(rows: list[dict], limit: int = 8, abstract_chars: int = 700) -> str:
    """What the indexed corpus says, in a form an ideation prompt can use.

    The uploaded literature previously reached no planning agent at all: the
    idea generator's `background_text` was the static domain-profile YAML, so a
    user who supplied twenty papers got ideas that had seen none of them.

    Each entry says what was actually read. A paper whose title came from the
    file name is shown as such, because "here are eight papers" and "here are
    eight files I could not identify" should not read the same.
    """

    if not rows:
        return ""
    lines = ["Indexed literature for this study (extracted locally, not verified):"]
    for row in rows[:limit]:
        title = str(row.get("title") or row.get("title_guess") or row.get("filename", "")).strip()
        if str(row.get("title_source", "")) == "filename":
            title += " [file name only -- the document was not identified]"
        meta = ", ".join(
            str(row.get(key, "")).strip()
            for key in ("authors", "year", "doi")
            if str(row.get(key, "")).strip()
        )
        lines.append(f"- {title}" + (f" ({meta})" if meta else ""))
        abstract = str(row.get("abstract", "") or "").strip()
        if abstract:
            lines.append(f"  abstract: {abstract[:abstract_chars]}")
        else:
            snippet = str(row.get("snippet", "") or "").strip()
            if snippet:
                lines.append(f"  no abstract extracted; opening text: {snippet[:300]}")
    lines.append(
        "This is a local keyword index, not a literature review. Do not assume these are the"
        " relevant papers, and do not cite anything above without checking it."
    )
    return "\n".join(lines)
