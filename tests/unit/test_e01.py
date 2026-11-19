"""Offline regression of experiment output and scientific/fixture separation."""

import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest


def test_e01_fixture_regression(tmp_path: Path) -> None:
    """Recompute golden metrics from the fixed fixture, including model seeds."""
    output = tmp_path / "fixture.csv"
    subprocess.run(
        [sys.executable, "experiments/e01_baseline_classical_vs_transformer.py",
         "--fixture", "--output", str(output)],
        check=True, capture_output=True, text=True,
    )
    with output.open(encoding="utf-8", newline="") as handle:
        actual = list(csv.DictReader(handle))
    with Path("results/tables/e01_classical_fixture.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        expected = list(csv.DictReader(handle))
    assert len(actual) == len(expected) == 3
    for row, golden in zip(actual, expected, strict=True):
        for column, value in row.items():
            if column in ("precision", "recall", "f1", "roc_auc"):
                assert float(value) == pytest.approx(float(golden[column]), abs=1e-12)
            else:
                assert value == golden[column]
    metadata = json.loads(output.with_suffix(".metadata.json").read_text(encoding="utf-8"))
    assert metadata["fixture_only"] is True
    assert metadata["dataset_origin"] == "offline-csv-sample"
    assert metadata["seed"] == metadata["config"]["seed"]
    assert metadata["csv_sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    assert metadata["versions"]["scikit-learn"] == "1.6.1"
    partitions = [set(row["text_sha256"] for row in metadata["splits"][name])
                  for name in ("train", "val", "test")]
    assert not partitions[0] & partitions[1]
    assert not partitions[0] & partitions[2]
    assert not partitions[1] & partitions[2]
    assert all("text" not in row for rows in metadata["splits"].values() for row in rows)
    golden_metadata = json.loads(Path(
        "results/tables/e01_classical_fixture.metadata.json"
    ).read_text(encoding="utf-8"))
    assert metadata["models"] == golden_metadata["models"]


def test_fixture_cannot_overwrite_scientific_table(tmp_path: Path) -> None:
    """The fixture path fails before writing a misleading full-dataset filename."""
    output = tmp_path / "e01_classical.csv"
    process = subprocess.run(
        [sys.executable, "experiments/e01_baseline_classical_vs_transformer.py",
         "--fixture", "--output", str(output)],
        capture_output=True, text=True, check=False,
    )
    assert process.returncode != 0
    assert "CI fixture must not overwrite" in process.stderr
    assert not output.exists()
