from __future__ import annotations

import json
import shutil
from pathlib import Path

from pydantic import BaseModel, Field


class NodePlotRecord(BaseModel):
    node_id: str
    work_dir: str
    plot_status: str
    plot_log: str | None = None
    figures: list[str] = Field(default_factory=list)


class PlotAggregationReport(BaseModel):
    records: list[NodePlotRecord] = Field(default_factory=list)

    def to_markdown(self) -> str:
        lines = ["# Plot Aggregation", "", f"- nodes: {len(self.records)}", ""]
        for record in self.records:
            lines.append(f"## Node {record.node_id}")
            lines.append(f"- status: {record.plot_status}")
            lines.append(f"- work_dir: `{record.work_dir}`")
            if record.plot_log:
                lines.append(f"- plot_log: `{record.plot_log}`")
            if record.figures:
                lines.append("- figures:")
                lines.extend(f"  - `{figure}`" for figure in record.figures)
            lines.append("")
        return "\n".join(lines)


def aggregate_node_plots(records: list[NodePlotRecord], output_dir: str | Path) -> PlotAggregationReport:
    root = Path(output_dir)
    aggregate_dir = root / "aggregate_figures"
    aggregate_dir.mkdir(exist_ok=True)
    for record in records:
        work_dir = Path(record.work_dir)
        for figure in list(record.figures):
            src = Path(figure)
            if not src.is_absolute():
                src = work_dir / src
            if src.exists() and src.is_file():
                dst = aggregate_dir / f"{record.node_id}_{src.name}"
                shutil.copyfile(src, dst)
    report = PlotAggregationReport(records=records)
    (root / "plot_aggregation.json").write_text(json.dumps(report.model_dump(), indent=2), encoding="utf-8")
    (root / "plot_aggregation.md").write_text(report.to_markdown(), encoding="utf-8")
    return report
