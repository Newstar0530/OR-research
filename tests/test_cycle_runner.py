from pathlib import Path

from src.utils.cycle_runner import run_bounded_cycles


def test_run_bounded_cycles_stops_for_human_gate(tmp_path: Path, monkeypatch) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("project_name: demo\nresearch_goal: test goal\n", encoding="utf-8")
    first_run = tmp_path / "run1"
    first_run.mkdir()
    (first_run / "config_used.yaml").write_text("project_name: demo\nresearch_goal: test goal\n", encoding="utf-8")
    (first_run / "next_config_patch.yaml").write_text("max_research_iterations: 3\n", encoding="utf-8")
    (first_run / "next_research_cycle_plan.json").write_text(
        '{"can_autorun_without_human": false, "actions": [{"priority": 1, "action_type": "verify_claims", "human_required": true}]}',
        encoding="utf-8",
    )

    class FakeOrchestrator:
        def __init__(self, config, project_root):
            pass

        def run(self):
            return first_run

    monkeypatch.setattr("src.utils.cycle_runner.ResearchOrchestrator", FakeOrchestrator)

    report = run_bounded_cycles(config_path, max_cycles=3, project_root=tmp_path, chain_output=tmp_path / "chain.json")

    assert len(report.records) == 2
    assert report.records[0].status == "completed"
    assert report.records[1].status == "stopped_for_human_review"
    assert (tmp_path / "chain.json").exists()
    assert (tmp_path / "chain.md").exists()
