from pathlib import Path

from src.config import load_config
from src.llm_client import LLMClient
from src.orchestrator import ResearchOrchestrator


def test_mock_llm_outputs_are_deterministic() -> None:
    llm = LLMClient(use_mock=True)
    assert llm.chat("novelty", "check") == llm.chat("novelty", "check")


def test_successful_generic_pipeline(tmp_path: Path) -> None:
    root = Path.cwd()
    config = load_config(root / "configs" / "default.yaml")
    config.output_dir = str(tmp_path)
    run_dir = ResearchOrchestrator(config, project_root=root).run()
    assert (run_dir / "results.csv").exists()
    assert (run_dir / "final_report.md").exists()
    assert (run_dir / "automated_review.md").exists()
    assert (run_dir / "research_protocol.md").exists()
    assert (run_dir / "domain_tool_spec.md").exists()
    assert (run_dir / "model_ir.json").exists()
    assert (run_dir / "statistical_evidence.md").exists()
    assert (run_dir / "final_paper.tex").exists()
    assert (run_dir / "literature_grounding.md").exists()
    assert (run_dir / "solver_tool_report.md").exists()
    assert (run_dir / "benchmark_manifest.md").exists()
    assert (run_dir / "model_export_manifest.json").exists()
    assert (run_dir / "citation_report.md").exists()
    assert (run_dir / "references.bib").exists()
    assert (run_dir / "research_tree.md").exists()
    assert (run_dir / "run_status.json").exists()
    assert (run_dir / "experiment_workspace" / "experiment.py").exists()
    assert (run_dir / "workspace_lineage.md").exists()
    assert (run_dir / "bfts_frontier.md").exists()
    assert (run_dir / "bfts_policy_queue.md").exists()
    assert (run_dir / "repair_trace.md").exists()
    assert (run_dir / "plot_aggregation.md").exists()
    assert (run_dir / "latex_compile_report.md").exists()
    assert (run_dir / "pdf_review_stub.md").exists()
    assert (run_dir / "llm_interactions.md").exists()
    assert (run_dir / "best_experiment.py").exists()
    assert (run_dir / "experiment_contract.md").exists()
    assert (run_dir / "llm_agent_trace.md").exists()
    assert (run_dir / "research_state.json").exists()
    assert list((run_dir / "figures").glob("*.png"))
