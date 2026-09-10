"""A file name is not a title, and the index has to know the difference.

The index this replaces stored `title_guess = filename`. A paper downloaded as
`1-s2.0-S0377221723004538-main.pdf` had that string as its title everywhere
downstream -- in the novelty report, in `references.bib`, in the relevance
scoring. These tests pin the distinction that makes that impossible: every
field records where it came from, and a record only counts as verifiable when
the document itself supplied enough to find the paper.
"""

from pathlib import Path

import pandas as pd
import pytest

from src.utils.bibliography import (
    corpus_headline,
    extract_abstract,
    extract_bibliography,
    extract_doi,
    extract_title_and_authors,
    extract_year,
    literature_digest,
)
from src.utils.citation_manager import build_citation_artifacts
from src.utils.literature import build_literature_index


FIRST_PAGE = """International Journal of Production Research
Vol. 61, No. 4, pp. 1123-1145

A bilevel programming approach to network interdiction under uncertainty

Jane Q. Doe, Wei Chen, and A. B. Smith
Department of Industrial Engineering, Example University

Abstract
We study a bilevel network interdiction problem in which the leader allocates a limited
budget to disrupt arcs while the follower routes flow optimally. We derive a single-level
reformulation via KKT conditions with a derived Big-M bound, and show it is exact.
Keywords: bilevel programming, interdiction
1. Introduction
Network interdiction has been studied since ...
"""
FULL_TEXT = " ".join(FIRST_PAGE.split()) + " (c) 2023 Elsevier. https://doi.org/10.1016/j.ejor.2023.04.017"


# -- the individual extractors ---------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        ("see https://doi.org/10.1016/j.ejor.2023.04.017 for details", "10.1016/j.ejor.2023.04.017"),
        ("DOI: 10.1287/opre.2021.2145.", "10.1287/opre.2021.2145"),
        ("no identifier at all here", ""),
    ],
)
def test_the_doi_is_the_one_field_that_can_be_matched_exactly(text: str, expected: str) -> None:
    field = extract_doi(text)
    assert field.value == expected
    assert field.source == ("doi_regex" if expected else "not_found")


def test_a_publication_year_beats_any_other_number_on_the_page() -> None:
    """1123 and 1145 are page numbers; 2015 is a citation; 2023 is the year."""

    text = "Vol. 61, pp. 1123-1145. Following Smith (2015). (c) 2023 Elsevier."
    assert extract_year(text).value == "2023"
    assert extract_year("no year here").source == "not_found"


def test_an_implausible_year_is_not_accepted() -> None:
    assert extract_year("model 1492 with 3021 nodes").value == ""


def test_the_title_is_taken_from_the_paper_not_the_journal_header() -> None:
    title, authors = extract_title_and_authors(FIRST_PAGE.splitlines())
    assert title.value == "A bilevel programming approach to network interdiction under uncertainty"
    assert title.source == "first_page"
    assert authors.value == "Jane Q. Doe, Wei Chen, and A. B. Smith"


def test_nothing_is_invented_when_the_first_page_is_unreadable() -> None:
    title, authors = extract_title_and_authors(["", "   ", "12345"])
    assert title.value == "" and title.source == "not_found"
    assert authors.value == ""


def test_the_abstract_stops_at_the_keywords_line() -> None:
    field = extract_abstract(FULL_TEXT)
    assert field.value.startswith("We study a bilevel network interdiction problem")
    assert "Keywords" not in field.value
    assert "Introduction" not in field.value


# -- the record ------------------------------------------------------------


def test_a_real_paper_is_identified_from_the_document() -> None:
    record = extract_bibliography(FULL_TEXT, FIRST_PAGE.splitlines(), {}, "1-s2.0-S0377221723004538-main")

    assert record.title.source == "first_page"
    assert "bilevel programming approach" in record.title.value
    assert record.year.value == "2023"
    assert record.doi.value == "10.1016/j.ejor.2023.04.017"
    assert record.is_verifiable
    assert record.citation_key("fallback") == "doe2023"


def test_the_pdf_own_metadata_wins_over_the_layout_heuristic() -> None:
    record = extract_bibliography(
        FULL_TEXT, FIRST_PAGE.splitlines(), {"/Title": "The Publisher's Recorded Title"}, "x"
    )
    assert record.title.value == "The Publisher's Recorded Title"
    assert record.title.source == "pdf_metadata"


@pytest.mark.parametrize(
    "junk", ["Untitled", "Microsoft Word - draft3.doc", "document1", "12345", "paper.pdf"]
)
def test_junk_in_the_pdf_title_field_is_not_used(junk: str) -> None:
    """Producers routinely write the source file name into /Title."""

    record = extract_bibliography(FULL_TEXT, FIRST_PAGE.splitlines(), {"/Title": junk}, "x")
    assert record.title.value != junk
    assert record.title.source == "first_page"


def test_an_unreadable_document_falls_back_to_the_filename_and_says_so() -> None:
    """The whole point: this must never look like a title that was read."""

    record = extract_bibliography("", [], {}, "1-s2.0-S0377221723004538-main")

    assert record.title.value == "1-s2.0-S0377221723004538-main"
    assert record.title.source == "filename"
    assert record.title.is_known is False
    assert record.is_verifiable is False
    assert "from the file name, not the document" in record.title.render()


def test_a_title_without_a_year_is_not_verifiable() -> None:
    """Searching a bare title string is not the same as identifying a paper."""

    record = extract_bibliography(
        "A bilevel programming approach to network interdiction under uncertainty",
        FIRST_PAGE.splitlines()[3:5],
        {},
        "x",
    )
    assert record.title.is_known
    assert record.year.value == ""
    assert record.is_verifiable is False


def test_the_corpus_headline_counts_what_is_actually_identified() -> None:
    identified = extract_bibliography(FULL_TEXT, FIRST_PAGE.splitlines(), {}, "a")
    anonymous = extract_bibliography("", [], {}, "b")

    headline = corpus_headline([identified, anonymous])
    assert "2 document(s) indexed" in headline
    assert "1 carry a DOI" in headline
    assert "identify a file rather than a paper" in headline
    assert corpus_headline([]) == "No documents were indexed."


# -- the digest that reaches ideation --------------------------------------


def test_the_digest_marks_documents_that_were_never_identified() -> None:
    """"Here are eight papers" and "here are eight files" must not read alike."""

    digest = literature_digest(
        [
            {"title": "A real paper", "title_source": "first_page", "year": "2023",
             "authors": "Doe, J.", "doi": "10.1/x", "abstract": "We show that " + "x" * 100},
            {"title": "1-s2.0-S03772217", "title_source": "filename", "snippet": "opening words"},
        ]
    )
    assert "A real paper (Doe, J., 2023, 10.1/x)" in digest
    assert "[file name only -- the document was not identified]" in digest
    assert "no abstract extracted" in digest
    assert "not a literature review" in digest


def test_an_empty_corpus_produces_no_digest_rather_than_an_empty_heading() -> None:
    assert literature_digest([]) == ""


# -- end to end through the index and the bibliography ---------------------


def test_a_text_corpus_is_indexed_with_its_extracted_metadata(tmp_path: Path) -> None:
    corpus = tmp_path / "papers"
    corpus.mkdir()
    (corpus / "downloaded_file_xyz.txt").write_text(FIRST_PAGE + "\ndoi:10.1016/j.ejor.2023.04.017\n", encoding="utf-8")
    (corpus / "unreadable.txt").write_text("nothing useful", encoding="utf-8")

    index_path, review_path = build_literature_index(corpus, tmp_path / "out")
    frame = pd.read_csv(index_path).fillna("")

    identified = frame[frame["filename"] == "downloaded_file_xyz.txt"].iloc[0]
    assert identified["title_source"] == "first_page"
    assert "bilevel programming approach" in identified["title"]
    assert identified["doi"] == "10.1016/j.ejor.2023.04.017"
    assert bool(identified["verifiable"]) is True

    anonymous = frame[frame["filename"] == "unreadable.txt"].iloc[0]
    assert anonymous["title_source"] == "filename"
    assert bool(anonymous["verifiable"]) is False

    review = review_path.read_text(encoding="utf-8")
    assert "1 carry a DOI" in review
    assert "from the file name, not the document" in review


def test_the_bibtex_never_invents_a_field(tmp_path: Path) -> None:
    corpus = tmp_path / "papers"
    corpus.mkdir()
    (corpus / "good.txt").write_text(FIRST_PAGE + "\ndoi:10.1016/j.ejor.2023.04.017\n", encoding="utf-8")
    (corpus / "mystery.txt").write_text("nothing useful at all", encoding="utf-8")
    index_path, _ = build_literature_index(corpus, tmp_path / "out")

    report = build_citation_artifacts(index_path, tmp_path / "out")
    bib = (tmp_path / "out" / "references.bib").read_text(encoding="utf-8")

    assert "@article{doe2023" in bib
    assert "doi = {10.1016/j.ejor.2023.04.017}" in bib
    # The unidentified one gets no author and no year, because none were read.
    mystery = next(record for record in report.records if record.filename == "mystery.txt")
    assert mystery.entry_type == "misc"
    assert mystery.authors == "" and mystery.year == ""
    assert "NOT A CITATION" in bib
    assert len(report.citable) == 1
    assert any("must not be cited as publications" in warning for warning in report.warnings)


def test_a_real_pdf_is_read_for_its_metadata(tmp_path: Path) -> None:
    """The layout heuristics have to survive a genuine PDF text layer."""

    reportlab = pytest.importorskip("reportlab")
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas

    path = tmp_path / "S0377221723004538.pdf"
    pdf = canvas.Canvas(str(path), pagesize=A4)
    y = 800
    for line in FIRST_PAGE.splitlines():
        pdf.drawString(60, y, line)
        y -= 16
    pdf.drawString(60, y - 10, "(c) 2023 Elsevier. https://doi.org/10.1016/j.ejor.2023.04.017")
    pdf.save()

    index_path, _ = build_literature_index(tmp_path, tmp_path / "out")
    frame = pd.read_csv(index_path).fillna("")
    row = frame.iloc[0]

    assert row["pages"] == 1
    assert row["doi"] == "10.1016/j.ejor.2023.04.017"
    assert "bilevel programming approach" in row["title"]
    assert row["title_source"] == "first_page"
    assert bool(row["verifiable"]) is True


def test_the_uploaded_literature_actually_reaches_ideation(tmp_path: Path) -> None:
    """It never did. `background_text` was the static domain-profile YAML, so a
    user who supplied twenty papers got ideas that had seen none of them."""

    import json
    import shutil

    from src.config import load_config
    from src.orchestrator import ResearchOrchestrator

    root = Path.cwd()
    project = tmp_path / "project"
    project.mkdir()
    for name in ("configs", "templates", "domain_profiles"):
        shutil.copytree(root / name, project / name)
    corpus = project / "papers"
    corpus.mkdir()
    (corpus / "paper.txt").write_text(FIRST_PAGE + "\ndoi:10.1016/j.ejor.2023.04.017\n", encoding="utf-8")

    config = load_config(project / "configs" / "default.yaml")
    config.project_name = "with_literature"
    config.output_dir = str(project / "runs")
    config.literature_dir = str(corpus)
    config.research_goal = "Bilevel network interdiction with a derived Big-M reformulation"
    run_dir = ResearchOrchestrator(config, project_root=project).run()

    provenance = json.loads((run_dir / "artifact_provenance.json").read_text(encoding="utf-8"))
    ideation = next(entry for entry in provenance["entries"] if entry["stage"] == "idea_generation")
    assert "literature_index" in ideation["inputs_used"], (
        "the indexed literature must be an input to ideation, not just an artifact"
    )

    review = (run_dir / "literature_review.md").read_text(encoding="utf-8")
    assert "1 carry a DOI" in review
