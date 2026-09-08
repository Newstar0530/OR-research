from pathlib import Path
import json

from src.config import load_config
from src.main import main
from src.utils.cycle_config import autorun_safety_decision, build_next_cycle_config


def test_build_next_cycle_config_applies_patch(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "config_used.yaml").write_text(
        "\n".join(
            [
                "project_name: demo",
                "research_goal: test goal",
                "max_research_iterations: 2",
                "max_candidate_branches: 2",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "next_config_patch.yaml").write_text(
        "max_research_iterations: 3\nmax_candidate_branches: 4\n",
        encoding="utf-8",
    )

    output = build_next_cycle_config(run_dir)
    config = load_config(output)

    assert output == run_dir / "next_cycle_config.yaml"
    assert config.project_name == "demo_cycle2"
    assert config.max_research_iterations == 3
    assert config.max_candidate_branches == 4


def test_prepare_next_cli_writes_config(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "config_used.yaml").write_text(
        "project_name: demo_cycle2\nresearch_goal: test goal\nmax_research_iterations: 2\n",
        encoding="utf-8",
    )
    (run_dir / "next_config_patch.yaml").write_text("max_research_iterations: 5\n", encoding="utf-8")
    output = tmp_path / "prepared.yaml"

    exit_code = main(["prepare-next", "--run-dir", str(run_dir), "--output", str(output)])
    config = load_config(output)

    assert exit_code == 0
    assert config.project_name == "demo_cycle3"
    assert config.max_research_iterations == 5


def test_autorun_safety_decision_blocks_human_required_action(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "next_research_cycle_plan.json").write_text(
        json.dumps(
            {
                "can_autorun_without_human": False,
                "actions": [
                    {
                        "priority": 1,
                        "action_type": "verify_claims",
                        "issue": "Claim checker flagged unsupported claims.",
                        "human_required": True,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    safe, reasons = autorun_safety_decision(run_dir)

    assert not safe
    assert any("verify_claims" in reason for reason in reasons)


def test_prepare_next_autorun_safe_only_runs_when_safe(tmp_path: Path, monkeypatch) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "config_used.yaml").write_text(
        "project_name: demo\nresearch_goal: test goal\nmax_research_iterations: 2\n",
        encoding="utf-8",
    )
    (run_dir / "next_config_patch.yaml").write_text("max_research_iterations: 3\n", encoding="utf-8")
    (run_dir / "next_research_cycle_plan.json").write_text(
        json.dumps({"can_autorun_without_human": True, "actions": []}),
        encoding="utf-8",
    )
    calls = []

    class FakeOrchestrator:
        def __init__(self, config, project_root):
            calls.append((config, project_root))

        def run(self):
            return tmp_path / "new_run"

    monkeypatch.setattr("src.main.ResearchOrchestrator", FakeOrchestrator)

    exit_code = main(["prepare-next", "--run-dir", str(run_dir), "--autorun-safe-only"])

    assert exit_code == 0
    assert calls
    assert calls[0][0].project_name == "demo_cycle2"


def test_prepare_next_autorun_safe_only_skips_when_unsafe(tmp_path: Path, monkeypatch) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "config_used.yaml").write_text(
        "project_name: demo\nresearch_goal: test goal\nmax_research_iterations: 2\n",
        encoding="utf-8",
    )
    (run_dir / "next_config_patch.yaml").write_text("max_research_iterations: 3\n", encoding="utf-8")
    (run_dir / "next_research_cycle_plan.json").write_text(
        json.dumps(
            {
                "can_autorun_without_human": False,
                "actions": [{"priority": 1, "action_type": "verify_claims", "human_required": True}],
            }
        ),
        encoding="utf-8",
    )

    def fail_orchestrator(*args, **kwargs):
        raise AssertionError("orchestrator should not run")

    monkeypatch.setattr("src.main.ResearchOrchestrator", fail_orchestrator)

    exit_code = main(["prepare-next", "--run-dir", str(run_dir), "--autorun-safe-only"])

    assert exit_code == 0
    assert (run_dir / "next_cycle_config.yaml").exists()
