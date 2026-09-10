from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
from pydantic import BaseModel, Field


class CitationRecord(BaseModel):
    key: str
    title: str
    filename: str
    path: str
    authors: str = ""
    year: str = ""
    doi: str = ""
    entry_type: str = "misc"
    needs_verification: bool = True
    #: False when the title came from the file name. Such an entry names a
    #: download, not a publication, and must not be cited as one.
    identifies_a_publication: bool = False


class CitationReport(BaseModel):
    records: list[CitationRecord] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @property
    def citable(self) -> list[CitationRecord]:
        return [record for record in self.records if record.identifies_a_publication]

    def to_markdown(self) -> str:
        citable = len(self.citable)
        lines = [
            "# Citation Report",
            "",
            f"- records: {len(self.records)}",
            f"- entries that identify a publication: {citable}",
            f"- entries that identify only a local file: {len(self.records) - citable}",
            "",
        ]
        if self.warnings:
            lines.append("## Warnings")
            lines.extend(f"- {item}" for item in self.warnings)
            lines.append("")
        lines.append("## Candidate References")
        for record in self.records or []:
            marker = "" if record.identifies_a_publication else "  **-- file name only, not a citation**"
            detail = ", ".join(part for part in (record.authors, record.year, record.doi) if part)
            lines.append(f"- `{record.key}`: {record.title}" + (f" ({detail})" if detail else "") + marker)
        lines.append("")
        lines.append("## Human Verification")
        lines.append(
            "- An entry marked `file name only` has no title, author or year read from the "
            "document. It identifies your download, not a paper, and searching for it will find "
            "the file rather than the publication."
        )
        lines.append(
            "- Entries with a DOI can be checked in one step. Do that before citing any of them: "
            "the metadata here is a heuristic read of page one, not a database record."
        )
        return "\n".join(lines) + "\n"


def build_citation_artifacts(literature_index_path: str | Path | None, output_dir: str | Path) -> CitationReport:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    if not literature_index_path or not Path(literature_index_path).exists():
        report = CitationReport(warnings=["No literature index available; references.bib contains no entries."])
        return _write(report, root)
    df = pd.read_csv(literature_index_path).fillna("")
    records = []
    used: set[str] = set()
    for row in df.to_dict("records"):
        title = str(row.get("title") or row.get("title_guess") or Path(str(row.get("filename", "reference"))).stem)
        authors = str(row.get("authors", ""))
        year = str(row.get("year", ""))
        doi = str(row.get("doi", ""))
        # A title lifted from the file name is not a citation, however well the
        # entry is formatted. `@article` is reserved for records the document
        # itself identified.
        from_document = str(row.get("title_source", "")) not in {"filename", "not_found", ""}
        identifies = bool(doi) or (from_document and bool(year))
        key = _unique_key(_preferred_key(authors, year, title), used)
        records.append(
            CitationRecord(
                key=key,
                title=title,
                filename=str(row.get("filename", "")),
                path=str(row.get("path", "")),
                authors=authors,
                year=year,
                doi=doi,
                entry_type="article" if identifies else "misc",
                identifies_a_publication=identifies,
            )
        )
    warnings = []
    unidentified = [record for record in records if not record.identifies_a_publication]
    if unidentified:
        warnings.append(
            f"{len(unidentified)} of {len(records)} entries have no metadata read from the "
            "document itself. They are written as `@misc` and must not be cited as publications."
        )
    report = CitationReport(records=records, warnings=warnings)
    return _write(report, root)


def _preferred_key(authors: str, year: str, title: str) -> str:
    """`doe2023` when the names are known, the title otherwise."""

    surname = authors.split(",")[0].split()[-1] if authors.strip() else ""
    if surname and year:
        return f"{surname}{year}"
    return title


def _write(report: CitationReport, output_dir: Path) -> CitationReport:
    (output_dir / "citation_report.json").write_text(report.model_dump_json(indent=2), encoding="utf-8")
    (output_dir / "citation_report.md").write_text(report.to_markdown(), encoding="utf-8")
    (output_dir / "references.bib").write_text(_bibtex(report.records), encoding="utf-8")
    return report


def _bibtex(records: list[CitationRecord]) -> str:
    """Write what was actually read, and say so in the note when little was.

    No field is invented. An entry with no author and no year simply has no
    author and no year -- a plausible-looking guess in a `.bib` file is the
    kind of error that survives all the way into a submitted paper.
    """

    entries = []
    for record in records:
        lines = [f"@{record.entry_type}{{{record.key},", f"  title = {{{_escape_bib(record.title)}}},"]
        if record.authors:
            lines.append(f"  author = {{{_escape_bib(record.authors)}}},")
        if record.year:
            lines.append(f"  year = {{{_escape_bib(record.year)}}},")
        if record.doi:
            lines.append(f"  doi = {{{_escape_bib(record.doi)}}},")
        note = (
            "Metadata extracted from the document; verify before citing."
            if record.identifies_a_publication
            else (
                "NOT A CITATION: no metadata was read from the document. The title above is the "
                f"file name `{record.filename}`. Replace this entry before use."
            )
        )
        lines.append(f"  note = {{{_escape_bib(note)}}},")
        lines.append(f"  howpublished = {{{_escape_bib(record.path)}}},")
        lines.append("}")
        entries.append("\n".join(lines))
    return "\n\n".join(entries) + ("\n" if entries else "")


def _unique_key(title: str, used: set[str]) -> str:
    base = re.sub(r"[^A-Za-z0-9]+", "_", title).strip("_").lower()[:40] or "reference"
    key = base
    suffix = 2
    while key in used:
        key = f"{base}_{suffix}"
        suffix += 1
    used.add(key)
    return key


def _escape_bib(text: str) -> str:
    return text.replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}")
