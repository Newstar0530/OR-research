from pathlib import Path

import pandas as pd
import pytest

from src.agents.gap_synthesis_agent import GapSynthesisAgent
from src.llm_client import LLMClient
from src.utils.literature import SUPPORTED_SUFFIXES, build_literature_index


def test_an_empty_corpus_still_produces_a_readable_table(tmp_path: Path) -> None:
    """Regression: a zero-column CSV used to abort the whole run downstream."""

    index_path, review_path = build_literature_index(tmp_path, tmp_path / "out")
    assert index_path.exists() and review_path.exists()
    frame = pd.read_csv(index_path)
    assert len(frame) == 0
    # The columns downstream stages read by name. Pinning the exact full list
    # would break on every added field without protecting anything; pinning the
    # contract does the opposite.
    assert {"path", "filename", "category", "keyword_hits", "snippet", "title", "title_source"} <= set(
        frame.columns
    )
    assert list(frame.columns), "a zero-column CSV is what broke the run"
    assert "No readable documents were found" in review_path.read_text(encoding="utf-8")


def test_a_missing_folder_is_handled_like_an_empty_one(tmp_path: Path) -> None:
    index_path, _ = build_literature_index(tmp_path / "does_not_exist", tmp_path / "out")
    assert pd.read_csv(index_path).empty


def test_plain_text_documents_are_indexed(tmp_path: Path) -> None:
    corpus = tmp_path / "papers"
    corpus.mkdir()
    (corpus / "bilevel_kkt.txt").write_text(
        "Bilevel programming reformulation via KKT conditions and big-M linearization.",
        encoding="utf-8",
    )
    (corpus / "annealing_notes.md").write_text(
        "Quantum annealing for QUBO: penalty weight selection in constrained optimization.",
        encoding="utf-8",
    )
    index_path, _ = build_literature_index(corpus, tmp_path / "out")
    frame = pd.read_csv(index_path)
    assert len(frame) == 2
    assert set(frame["filename"]) == {"bilevel_kkt.txt", "annealing_notes.md"}
    assert frame["keyword_hits"].str.contains("optimization").any()
    assert ".txt" in SUPPORTED_SUFFIXES and ".md" in SUPPORTED_SUFFIXES


def test_unsupported_files_are_skipped_without_breaking_the_index(tmp_path: Path) -> None:
    corpus = tmp_path / "papers"
    corpus.mkdir()
    (corpus / "notes.rtf").write_text("ignored", encoding="utf-8")
    (corpus / "data.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    assert pd.read_csv(build_literature_index(corpus, tmp_path / "out")[0]).empty


@pytest.mark.parametrize("populate", [True, False])
def test_gap_synthesis_survives_any_index(tmp_path: Path, populate: bool) -> None:
    """The crash site: an empty index must mean "no literature", not "give up"."""

    corpus = tmp_path / "papers"
    corpus.mkdir()
    if populate:
        (corpus / "scheduling_paper.txt").write_text(
            "Single machine scheduling to minimise total tardiness with dispatching rules.",
            encoding="utf-8",
        )
    index_path, _ = build_literature_index(corpus, tmp_path / "out")
    text, source, inputs = GapSynthesisAgent(LLMClient(use_mock=True)).run(
        "Minimise total tardiness on a single machine",
        {"name": "scheduling", "metrics": ["tardiness"], "algorithm_families": ["dispatching"]},
        index_path,
    )
    assert text.startswith("# Research Gap Analysis")
    # Without a model there is no gap to state, and the stage says so rather
    # than emitting boilerplate that looks like one.
    assert source == "not_generated"
    assert "research_goal" in inputs


def test_relevance_terms_come_from_the_study_not_a_fixed_list() -> None:
    terms = GapSynthesisAgent._relevance_terms(
        "Bilevel programming transformed into a mixed binary program",
        {"name": "optimization", "metrics": ["gap_to_known_optimum"], "algorithm_families": ["kkt"]},
    )
    assert "bilevel" in terms and "programming" in terms
    assert "optimization" in terms and "gap_to_known_optimum" in terms
    # The old hard-coded topics have nothing to do with this study.
    assert "supply_chain" not in terms and "chaos" not in terms
