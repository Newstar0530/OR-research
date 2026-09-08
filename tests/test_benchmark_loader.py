from pathlib import Path

import pandas as pd

from src.core.benchmark_loader import build_benchmark_manifest


def test_benchmark_loader_indexes_csv(tmp_path: Path) -> None:
    bench = tmp_path / "bench"
    bench.mkdir()
    pd.DataFrame([{"a": 1, "b": 2}]).to_csv(bench / "instance.csv", index=False)

    manifest = build_benchmark_manifest(bench, tmp_path)

    assert manifest.records
    assert manifest.records[0].columns == ["a", "b"]
    assert (tmp_path / "benchmark_manifest.md").exists()
