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
    entry_type: str = "misc"
    needs_verification: bool = True


class CitationReport(BaseModel):
    records: list[CitationRecord] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    def to_markdown(self) -> str:
        lines = ["# Citation Report", "", f"- records: {len(self.records)}", ""]
        if self.warnings:
            lines.append("## Warnings")
            lines.extend(f"- {item}" for item in self.warnings)
            lines.append("")
        lines.append("## Candidate References")
        lines.extend(f"- `{record.key}`: {record.title} ({record.filename})" for record in self.records or [])
        lines.append("")
        lines.append("## Human Verification")
        lines.append("- Generated BibTeX entries use filename/title guesses and must be corrected before publication.")
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
        title = str(row.get("title_guess") or Path(str(row.get("filename", "reference"))).stem)
        key = _unique_key(title, used)
        records.append(
            CitationRecord(
                key=key,
                title=title,
                filename=str(row.get("filename", "")),
                path=str(row.get("path", "")),
            )
        )
    report = CitationReport(records=records)
    return _write(report, root)


def _write(report: CitationReport, output_dir: Path) -> CitationReport:
    (output_dir / "citation_report.json").write_text(report.model_dump_json(indent=2), encoding="utf-8")
    (output_dir / "citation_report.md").write_text(report.to_markdown(), encoding="utf-8")
    (output_dir / "references.bib").write_text(_bibtex(report.records), encoding="utf-8")
    return report


def _bibtex(records: list[CitationRecord]) -> str:
    entries = []
    for record in records:
        entries.append(
            "\n".join(
                [
                    f"@misc{{{record.key},",
                    f"  title = {{{_escape_bib(record.title)}}},",
                    f"  note = {{{_escape_bib('Auto-generated from local file: ' + record.filename + '. Verify metadata.')}}},",
                    f"  howpublished = {{{_escape_bib(record.path)}}},",
                    "}",
                ]
            )
        )
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
