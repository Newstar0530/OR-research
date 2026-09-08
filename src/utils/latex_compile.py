from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from pydantic import BaseModel, Field


class LatexCompileReport(BaseModel):
    status: str
    tex_file: str
    pdf_file: str | None = None
    log_file: str | None = None
    notes: list[str] = Field(default_factory=list)

    def to_markdown(self) -> str:
        lines = ["# LaTeX Compile Report", "", f"- status: {self.status}", f"- tex_file: `{self.tex_file}`"]
        lines.append(f"- pdf_file: `{self.pdf_file or 'none'}`")
        lines.append(f"- log_file: `{self.log_file or 'none'}`")
        lines.append("")
        lines.append("## Notes")
        lines.extend(f"- {note}" for note in self.notes or ["None."])
        return "\n".join(lines) + "\n"


class PDFReviewStub(BaseModel):
    status: str
    pdf_available: bool
    checks: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    def to_markdown(self) -> str:
        lines = ["# PDF Review Stub", "", f"- status: {self.status}", f"- pdf_available: {self.pdf_available}", ""]
        lines.append("## Checks")
        lines.extend(f"- {item}" for item in self.checks or ["None."])
        lines.append("")
        lines.append("## Warnings")
        lines.extend(f"- {item}" for item in self.warnings or ["None."])
        lines.append("")
        lines.append("## Human Verification")
        lines.append("- This is a structural stub, not a visual or scientific paper review.")
        return "\n".join(lines) + "\n"


def compile_latex(run_dir: str | Path, tex_name: str = "final_paper.tex") -> LatexCompileReport:
    root = Path(run_dir)
    tex = root / tex_name
    pdflatex = shutil.which("pdflatex")
    if not tex.exists():
        report = LatexCompileReport(status="missing_tex", tex_file=str(tex), notes=["final_paper.tex was not found."])
        return _write_compile(report, root)
    if not pdflatex:
        report = LatexCompileReport(status="skipped_no_pdflatex", tex_file=str(tex), notes=["pdflatex was not found on PATH."])
        return _write_compile(report, root)
    proc = subprocess.run([pdflatex, "-interaction=nonstopmode", tex.name], cwd=str(root), capture_output=True, text=True, timeout=60, shell=False)
    log = root / "latex_compile_stdout.log"
    log.write_text(f"STDOUT\n{proc.stdout}\n\nSTDERR\n{proc.stderr}\n", encoding="utf-8")
    pdf = root / tex.with_suffix(".pdf").name
    status = "success" if proc.returncode == 0 and pdf.exists() else "failed"
    report = LatexCompileReport(status=status, tex_file=str(tex), pdf_file=str(pdf) if pdf.exists() else None, log_file=str(log), notes=[f"pdflatex returncode={proc.returncode}"])
    return _write_compile(report, root)


def review_pdf_stub(run_dir: str | Path) -> PDFReviewStub:
    root = Path(run_dir)
    pdf = root / "final_paper.pdf"
    tex = root / "final_paper.tex"
    checks = []
    warnings = []
    if pdf.exists():
        checks.append("PDF file exists.")
    else:
        warnings.append("PDF file was not generated.")
    if tex.exists():
        text = tex.read_text(encoding="utf-8", errors="ignore")
        for section in ["Research Report", "Statistical Evidence", "Literature Grounding", "Automated Review"]:
            if section in text:
                checks.append(f"Section present: {section}.")
            else:
                warnings.append(f"Section missing: {section}.")
    else:
        warnings.append("LaTeX source missing.")
    status = "review_stub_complete" if checks else "review_stub_no_evidence"
    report = PDFReviewStub(status=status, pdf_available=pdf.exists(), checks=checks, warnings=warnings)
    (root / "pdf_review_stub.json").write_text(json.dumps(report.model_dump(), indent=2), encoding="utf-8")
    (root / "pdf_review_stub.md").write_text(report.to_markdown(), encoding="utf-8")
    return report


def _write_compile(report: LatexCompileReport, root: Path) -> LatexCompileReport:
    (root / "latex_compile_report.json").write_text(json.dumps(report.model_dump(), indent=2), encoding="utf-8")
    (root / "latex_compile_report.md").write_text(report.to_markdown(), encoding="utf-8")
    return report
