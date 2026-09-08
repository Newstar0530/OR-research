from pathlib import Path

from src.core.code_editing_backend import AiderRepairBackend, WorkspaceRepairRequest
from src.core.research_journal import ResearchJournal
from src.core.research_node import ResearchNode
from src.core.workspace_debugger import repair_workspace_after_failure
from src.utils.bfts_policy import build_bfts_policy_queue
from src.utils.latex_compile import compile_latex, review_pdf_stub
from src.utils.llm_tracker import LLMInteractionTracker
from src.utils.plot_aggregator import NodePlotRecord, aggregate_node_plots


def test_llm_interaction_tracker_records_token_estimates() -> None:
    tracker = LLMInteractionTracker()
    tracker.record("mock", "m", "system prompt", "user prompt", "response", 0.1)

    assert tracker.interactions[0].approximate_prompt_tokens > 0
    assert "calls: 1" in tracker.to_markdown()


def test_workspace_debugger_repairs_future_import_position(tmp_path: Path) -> None:
    script = tmp_path / "experiment.py"
    script.write_text(
        '"""header"""\n"""bad autonomous header"""\nfrom __future__ import annotations\nVALUE = 1\n',
        encoding="utf-8",
    )

    result = repair_workspace_after_failure(tmp_path, "from __future__ imports must occur at the beginning")

    assert result.applied
    text = script.read_text(encoding="utf-8")
    assert text.index("from __future__ import annotations") < text.index("bad autonomous header")


def test_aider_backend_skips_when_command_is_missing(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "experiment.py").write_text("VALUE = 1\n", encoding="utf-8")
    monkeypatch.setattr("src.core.code_editing_backend.shutil.which", lambda command: None)

    result = AiderRepairBackend().repair(WorkspaceRepairRequest(work_dir=tmp_path, stderr="NameError"))

    assert result.attempted is False
    assert result.backend == "aider"
    assert "not found" in result.reason


def test_aider_backend_invokes_single_workspace_file(tmp_path: Path, monkeypatch) -> None:
    script = tmp_path / "experiment.py"
    script.write_text("VALUE = 1\n", encoding="utf-8")
    calls = {}

    class Proc:
        returncode = 0
        stdout = "ok"
        stderr = ""

    def fake_run(cmd, cwd, capture_output, text, timeout, shell):
        calls["cmd"] = cmd
        calls["cwd"] = cwd
        calls["shell"] = shell
        return Proc()

    monkeypatch.setattr("src.core.code_editing_backend.shutil.which", lambda command: "aider.exe")
    monkeypatch.setattr("src.core.code_editing_backend.subprocess.run", fake_run)

    result = AiderRepairBackend(timeout_seconds=5).repair(
        WorkspaceRepairRequest(work_dir=tmp_path, stderr="NameError", attempt=2)
    )

    assert result.applied
    assert calls["cwd"] == str(tmp_path)
    assert calls["shell"] is False
    assert calls["cmd"][-1] == "experiment.py"
    assert (tmp_path / "aider_repair_prompt_attempt_2.md").exists()


def test_plot_aggregator_writes_report(tmp_path: Path) -> None:
    node_dir = tmp_path / "node"
    fig_dir = node_dir / "figures"
    fig_dir.mkdir(parents=True)
    fig = fig_dir / "plot.png"
    fig.write_bytes(b"fake")

    report = aggregate_node_plots(
        [NodePlotRecord(node_id="n1", work_dir=str(node_dir), plot_status="success", figures=[str(fig)])],
        tmp_path,
    )

    assert report.records[0].node_id == "n1"
    assert (tmp_path / "plot_aggregation.md").exists()
    assert list((tmp_path / "aggregate_figures").glob("*.png"))


def test_bfts_policy_queue_prioritizes_best_minimize_node(tmp_path: Path) -> None:
    journal = ResearchJournal()
    journal.append(ResearchNode(id="bad", iteration=0, branch_index=0, plan="a", status="success", metric_value=10.0))
    journal.append(ResearchNode(id="good", iteration=0, branch_index=1, plan="b", status="success", metric_value=5.0))

    report = build_bfts_policy_queue(journal, tmp_path, objective_direction="minimize")

    assert report.queue[0].node_id == "good"
    assert (tmp_path / "bfts_policy_queue.md").exists()


def test_latex_compile_and_pdf_review_stubs_without_pdflatex(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "final_paper.tex").write_text("\\documentclass{article}\\begin{document}x\\end{document}", encoding="utf-8")
    monkeypatch.setattr("src.utils.latex_compile.shutil.which", lambda name: None)

    compile_report = compile_latex(tmp_path)
    review = review_pdf_stub(tmp_path)

    assert compile_report.status == "skipped_no_pdflatex"
    assert review.pdf_available is False
    assert (tmp_path / "latex_compile_report.md").exists()
    assert (tmp_path / "pdf_review_stub.md").exists()
