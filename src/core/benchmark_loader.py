from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from pydantic import BaseModel, Field


class BenchmarkRecord(BaseModel):
    path: str
    filename: str
    file_type: str
    rows: int | None = None
    columns: list[str] = Field(default_factory=list)
    size_bytes: int
    status: str
    notes: str = ""


class BenchmarkManifest(BaseModel):
    benchmark_dir: str | None
    records: list[BenchmarkRecord] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    def to_markdown(self) -> str:
        lines = ["# Benchmark Manifest", "", f"- benchmark_dir: `{self.benchmark_dir or 'none'}`", f"- files: {len(self.records)}", ""]
        if self.warnings:
            lines.append("## Warnings")
            lines.extend(f"- {item}" for item in self.warnings)
            lines.append("")
        lines.append("## Records")
        if not self.records:
            lines.append("- No benchmark files indexed.")
        for record in self.records:
            lines.append(f"### {record.filename}")
            lines.append(f"- type: {record.file_type}")
            lines.append(f"- status: {record.status}")
            lines.append(f"- rows: {record.rows if record.rows is not None else 'unknown'}")
            lines.append(f"- columns: {', '.join(record.columns) if record.columns else 'unknown'}")
            lines.append(f"- size_bytes: {record.size_bytes}")
            if record.notes:
                lines.append(f"- notes: {record.notes}")
            lines.append("")
        lines.append("## Human Verification")
        lines.append("- Confirm benchmark provenance, license, preprocessing, and whether instances match the research question.")
        return "\n".join(lines) + "\n"


def build_benchmark_manifest(benchmark_dir: str | Path | None, output_dir: str | Path) -> BenchmarkManifest:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    if not benchmark_dir:
        manifest = BenchmarkManifest(benchmark_dir=None, warnings=["No benchmark_dir configured; experiments may use synthetic data."])
        return _write(manifest, root)
    bench_root = Path(benchmark_dir)
    if not bench_root.exists():
        manifest = BenchmarkManifest(benchmark_dir=str(bench_root), warnings=[f"Benchmark directory not found: {bench_root}"])
        return _write(manifest, root)
    records = []
    for path in sorted([p for p in bench_root.rglob("*") if p.is_file()]):
        if path.suffix.lower() not in {".csv", ".json", ".jsonl", ".txt", ".dat"}:
            continue
        records.append(_inspect_file(path))
    manifest = BenchmarkManifest(
        benchmark_dir=str(bench_root),
        records=records,
        warnings=[] if records else ["Benchmark directory exists but no supported benchmark files were found."],
    )
    return _write(manifest, root)


def _inspect_file(path: Path) -> BenchmarkRecord:
    file_type = path.suffix.lower().lstrip(".") or "unknown"
    size = path.stat().st_size
    try:
        if file_type == "csv":
            df = pd.read_csv(path, nrows=1000)
            return BenchmarkRecord(
                path=str(path),
                filename=path.name,
                file_type=file_type,
                rows=len(df),
                columns=[str(col) for col in df.columns],
                size_bytes=size,
                status="indexed",
                notes="CSV preview limited to 1000 rows.",
            )
        if file_type == "json":
            data = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
            rows = len(data) if isinstance(data, list) else 1
            columns = list(data[0].keys()) if isinstance(data, list) and data and isinstance(data[0], dict) else []
            return BenchmarkRecord(path=str(path), filename=path.name, file_type=file_type, rows=rows, columns=columns, size_bytes=size, status="indexed")
        preview = path.read_text(encoding="utf-8", errors="ignore")[:2000]
        rows = len([line for line in preview.splitlines() if line.strip()])
        return BenchmarkRecord(path=str(path), filename=path.name, file_type=file_type, rows=rows, size_bytes=size, status="previewed")
    except Exception as exc:
        return BenchmarkRecord(path=str(path), filename=path.name, file_type=file_type, size_bytes=size, status="failed", notes=str(exc))


def _write(manifest: BenchmarkManifest, output_dir: Path) -> BenchmarkManifest:
    (output_dir / "benchmark_manifest.json").write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    (output_dir / "benchmark_manifest.md").write_text(manifest.to_markdown(), encoding="utf-8")
    return manifest
