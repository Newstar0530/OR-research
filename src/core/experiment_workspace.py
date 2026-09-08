from __future__ import annotations

import json
import shutil
from pathlib import Path

from src.schemas import AlgorithmPlan, ResearchIdea


WORKSPACE_FILES = ["experiment.py", "plot.py", "prompt.json", "notes.md"]


class ExperimentWorkspaceManager:
    """Create copyable experiment workspaces for AI-Scientist-style search."""

    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir)

    def initialize_root(self, base_code: str, idea: ResearchIdea, algorithm: AlgorithmPlan) -> Path:
        workspace = self.run_dir / "experiment_workspace"
        workspace.mkdir(parents=True, exist_ok=True)
        (workspace / "experiment.py").write_text(base_code, encoding="utf-8")
        (workspace / "plot.py").write_text(_plot_script(), encoding="utf-8")
        (workspace / "prompt.json").write_text(
            json.dumps(
                {
                    "idea": idea.model_dump(),
                    "algorithm": algorithm.model_dump(),
                    "instructions": [
                        "Edit experiment.py only through safe patches or bounded refiner operations.",
                        "Preserve results.csv output and SUMMARY_JSON stdout contract.",
                        "Keep runs short and inspectable.",
                    ],
                },
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
        (workspace / "notes.md").write_text(
            "# Experiment Workspace\n\n"
            "This is the root editable workspace for the autonomous experiment search.\n"
            "Each research node copies this workspace, applies bounded edits, and runs locally.\n",
            encoding="utf-8",
        )
        return workspace

    def create_node_workspace(self, source_workspace: str | Path, node_dir: str | Path, plan: str) -> Path:
        source = Path(source_workspace)
        target = Path(node_dir)
        target.mkdir(parents=True, exist_ok=True)
        for item in source.iterdir():
            destination = target / item.name
            if item.is_dir():
                if destination.exists():
                    shutil.rmtree(destination)
                shutil.copytree(item, destination)
            elif item.is_file():
                shutil.copyfile(item, destination)
        notes = target / "notes.md"
        previous = notes.read_text(encoding="utf-8") if notes.exists() else "# Experiment Workspace\n"
        notes.write_text(previous + f"\n## Node Plan\n{plan}\n", encoding="utf-8")
        prompt_path = target / "prompt.json"
        if prompt_path.exists():
            payload = json.loads(prompt_path.read_text(encoding="utf-8"))
        else:
            payload = {}
        payload["node_plan"] = plan
        prompt_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        return target


def _plot_script() -> str:
    return '''"""Optional plot helper for an experiment workspace."""
from pathlib import Path

import pandas as pd


def main() -> None:
    path = Path("results.csv")
    if not path.exists():
        print("No results.csv found.")
        return
    df = pd.read_csv(path)
    print(df.head().to_string(index=False))


if __name__ == "__main__":
    main()
'''
