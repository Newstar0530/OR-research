from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field

from src.config import load_config
from src.orchestrator import ResearchOrchestrator
from src.utils.cycle_config import autorun_safety_decision, build_next_cycle_config


class CycleRecord(BaseModel):
    cycle_index: int
    source_config: str | None = None
    source_run_dir: str | None = None
    generated_config: str | None = None
    run_dir: str | None = None
    status: str
    stop_reasons: list[str] = Field(default_factory=list)


class CycleChainReport(BaseModel):
    max_cycles: int
    records: list[CycleRecord] = Field(default_factory=list)

    def to_markdown(self) -> str:
        lines = ["# Cycle Chain", "", f"- max_cycles: {self.max_cycles}", f"- records: {len(self.records)}", ""]
        for record in self.records:
            lines.append(f"## Cycle {record.cycle_index}")
            lines.append(f"- status: {record.status}")
            if record.source_config:
                lines.append(f"- source_config: `{record.source_config}`")
            if record.source_run_dir:
                lines.append(f"- source_run_dir: `{record.source_run_dir}`")
            if record.generated_config:
                lines.append(f"- generated_config: `{record.generated_config}`")
            if record.run_dir:
                lines.append(f"- run_dir: `{record.run_dir}`")
            if record.stop_reasons:
                lines.append("- stop_reasons:")
                lines.extend(f"  - {reason}" for reason in record.stop_reasons)
            lines.append("")
        return "\n".join(lines)


def run_bounded_cycles(
    initial_config: str | Path,
    max_cycles: int,
    project_root: str | Path,
    chain_output: str | Path | None = None,
) -> CycleChainReport:
    if max_cycles < 1:
        raise ValueError("max_cycles must be at least 1")
    root = Path(project_root)
    report = CycleChainReport(max_cycles=max_cycles)
    config_path = Path(initial_config)
    previous_run: Path | None = None

    for cycle_index in range(1, max_cycles + 1):
        if cycle_index == 1:
            config = load_config(config_path)
            run_dir = ResearchOrchestrator(config, project_root=root).run()
            previous_run = run_dir
            report.records.append(
                CycleRecord(
                    cycle_index=cycle_index,
                    source_config=str(config_path),
                    run_dir=str(run_dir),
                    status="completed",
                )
            )
            continue

        if previous_run is None:
            report.records.append(CycleRecord(cycle_index=cycle_index, status="stopped", stop_reasons=["No previous run found."]))
            break

        safe, reasons = autorun_safety_decision(previous_run)
        generated_config = build_next_cycle_config(previous_run)
        if not safe:
            report.records.append(
                CycleRecord(
                    cycle_index=cycle_index,
                    source_run_dir=str(previous_run),
                    generated_config=str(generated_config),
                    status="stopped_for_human_review",
                    stop_reasons=reasons,
                )
            )
            break
        config = load_config(generated_config)
        run_dir = ResearchOrchestrator(config, project_root=root).run()
        report.records.append(
            CycleRecord(
                cycle_index=cycle_index,
                source_run_dir=str(previous_run),
                generated_config=str(generated_config),
                run_dir=str(run_dir),
                status="completed",
            )
        )
        previous_run = run_dir

    if chain_output:
        output = Path(chain_output)
    elif previous_run:
        output = previous_run / "cycle_chain.json"
    else:
        output = root / "cycle_chain.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report.model_dump(), indent=2), encoding="utf-8")
    output.with_suffix(".md").write_text(report.to_markdown(), encoding="utf-8")
    return report
