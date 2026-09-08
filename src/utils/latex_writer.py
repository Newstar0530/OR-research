from __future__ import annotations

import re
from pathlib import Path


def write_latex_paper(run_dir: str | Path) -> Path:
    root = Path(run_dir)
    report = _read(root / "final_report.md")
    model = _read(root / "model_draft.md")
    stats = _read(root / "statistical_evidence.md")
    grounding = _read(root / "literature_grounding.md")
    citations = _read(root / "citation_report.md")
    review = _read(root / "automated_review.md")
    body = "\n\n".join(
        [
            "\\section{AI-Generated Draft Warning}\nThis manuscript is an AI-generated research draft for human inspection. Claims, citations, equations, and conclusions require verification.",
            "\\section{Research Report}\n" + _markdown_to_latex(report),
            "\\section{Mathematical Model Draft}\n" + _markdown_to_latex(model),
            "\\section{Statistical Evidence}\n" + _markdown_to_latex(stats),
            "\\section{Literature Grounding}\n" + _markdown_to_latex(grounding),
            "\\section{Candidate References}\n" + _markdown_to_latex(citations),
            "\\section{Automated Review}\n" + _markdown_to_latex(review),
        ]
    )
    tex = (
        "\\documentclass[11pt]{article}\n"
        "\\usepackage[margin=1in]{geometry}\n"
        "\\usepackage{booktabs}\n"
        "\\usepackage{hyperref}\n"
        "\\title{Human-in-the-Loop OR Research Draft}\n"
        "\\author{Research Automation Assistant}\n"
        "\\date{\\today}\n"
        "\\begin{document}\n"
        "\\maketitle\n"
        f"{body}\n"
        "\\end{document}\n"
    )
    path = root / "final_paper.tex"
    path.write_text(tex, encoding="utf-8")
    return path


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore") if path.exists() else "Not available."


def _markdown_to_latex(text: str) -> str:
    lines = []
    in_list = False
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            if in_list:
                lines.append("\\end{itemize}")
                in_list = False
            lines.append("")
            continue
        if line.startswith("#"):
            if in_list:
                lines.append("\\end{itemize}")
                in_list = False
            title = line.lstrip("#").strip()
            lines.append("\\subsection{" + _escape(title) + "}")
        elif line.startswith("- "):
            if not in_list:
                lines.append("\\begin{itemize}")
                in_list = True
            lines.append("\\item " + _escape(line[2:]))
        else:
            if in_list:
                lines.append("\\end{itemize}")
                in_list = False
            lines.append(_escape(line))
    if in_list:
        lines.append("\\end{itemize}")
    return "\n".join(lines)


def _escape(text: str) -> str:
    text = re.sub(r"`([^`]+)`", r"\1", text)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in text)
