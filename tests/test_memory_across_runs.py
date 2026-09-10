"""Two runs of the same project, and the second one has to have read the first.

The point of memory is the read-back. A test that only checked `runs.jsonl` was
written would have passed against the old code, which wrote a memory line every
run and never opened it again.
"""

import json
import shutil
from pathlib import Path

import pytest

from src.config import load_config
from src.orchestrator import ResearchOrchestrator
from src.utils.research_memory import load_run_records


@pytest.fixture
def project(tmp_path: Path) -> Path:
    """An isolated project root, so the run history here is this test's alone."""

    root = Path.cwd()
    workspace = tmp_path / "project"
    workspace.mkdir()
    for name in ("configs", "templates"):
        shutil.copytree(root / name, workspace / name)
    return workspace


def _run(project: Path, name: str):
    config = load_config(project / "configs" / "default.yaml")
    config.project_name = name
    config.output_dir = str(project / "runs")
    return ResearchOrchestrator(config, project_root=project).run()


def test_the_first_run_has_no_history_and_says_so(project: Path) -> None:
    run_dir = _run(project, "first")

    findings = json.loads((run_dir / "prior_findings.json").read_text(encoding="utf-8"))
    assert findings["matched"] == []
    assert "No prior run" in (run_dir / "prior_findings.md").read_text(encoding="utf-8")


def test_the_first_run_records_what_it_measured(project: Path) -> None:
    _run(project, "first")

    records = load_run_records(project)
    assert len(records) == 1
    record = records[0]
    assert record.project_name == "first"
    assert record.primary_metric
    assert record.node_count >= 1
    assert record.evidence_strength, "the statistical verdict must be carried forward"
    assert record.timestamp


def test_the_second_run_reads_the_first_and_carries_it_as_evidence(project: Path) -> None:
    first_dir = _run(project, "first")
    second_dir = _run(project, "second")

    findings = json.loads((second_dir / "prior_findings.json").read_text(encoding="utf-8"))
    assert findings["total_runs_in_memory"] == 1
    assert len(findings["matched"]) == 1
    assert findings["matched"][0]["run_dir"] == str(first_dir)

    text = (second_dir / "prior_findings.md").read_text(encoding="utf-8")
    assert "first" in text
    assert "not a reason to skip one" in text, "prior evidence must not read as settled"


def test_a_run_never_matches_itself(project: Path) -> None:
    """Its own record is written after the brief is built, but exclusion is
    explicit rather than dependent on that ordering."""

    _run(project, "first")
    second_dir = _run(project, "second")

    findings = json.loads((second_dir / "prior_findings.json").read_text(encoding="utf-8"))
    assert str(second_dir) not in [item["run_dir"] for item in findings["matched"]]
