"""Offline end-to-end quick experiments, report provenance and persisted measurement math."""

import csv
import hashlib
import json
import re
from collections.abc import Iterator
from importlib import import_module
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType

import numpy as np
import pytest


@pytest.fixture
def runner(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Import standalone experiment entry points from their native script path."""
    monkeypatch.syspath_prepend(str(Path("experiments").resolve()))
    return import_module("run_all")


@pytest.fixture
def artifacts() -> Iterator[Path]:
    """Contain even mocked raw transcripts under results/ throughout the test."""
    with TemporaryDirectory(prefix="pytest_m8_", dir="results") as directory:
        yield Path(directory)


def test_quick_experiments_report_and_measurements(runner: ModuleType, artifacts: Path) -> None:
    """Run real small classical fits offline; recalculate all latency/flip summaries from rows."""
    report = artifacts / "report.md"
    manifest_path = runner.run(
        Path("configs/default.yaml"), artifacts, quick=True, report_path=report
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["quick"] and manifest["fixture_only"]
    assert any(s["status"] == "synthetic_fixture_not_live" for s in manifest["stages"])
    for path, sha in manifest["artifact_sha256"].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == sha

    def rows(name: str) -> list[dict[str, str]]:
        """Read the output rather than copying expected efficacy values into tests."""
        with (artifacts / "tables" / name).open(encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))

    timings = rows("e05_latency_samples_quick.csv")
    for row in rows("e05_latency_quick.csv"):
        if row["model"] == "distilbert":
            assert row["status"] == "not_requested_optional_checkpoint" and row["mean_ms"] == ""
            continue
        samples = [float(r["duration_ms"]) for r in timings if r["model"] == row["model"]]
        assert float(row["mean_ms"]) == pytest.approx(np.mean(samples))
        assert float(row["median_ms"]) == pytest.approx(np.median(samples))
        assert float(row["p95_ms"]) == pytest.approx(np.percentile(samples, 95))
        assert 0 <= float(row["mean_ci_low_ms"]) <= float(row["mean_ci_high_ms"])
    details = rows("e06_search_quick.csv")
    for row in rows("e06_robustness_quick.csv"):
        if row["status"] != "evaluated":
            continue
        selected = [r for r in details if r["model"] == row["model"]]
        eligible = [r for r in selected if r["status"] != "already_benign"]
        found = [r for r in selected if r["status"] == "flipped"]
        assert int(row["n_eligible"]) == len(eligible)
        assert int(row["n_flipped"]) == len(found)
        assert float(row["flip_rate"]) == len(found) / len(eligible)
        assert all(int(r["queries"]) <= int(row["max_queries"]) for r in selected)
        assert (
            row["minimum_observed_budget"] == str(min(int(r["minimum_budget"]) for r in found))
            if found
            else row["minimum_observed_budget"] == ""
        )
    text = report.read_text(encoding="utf-8")
    assert len(re.findall(r"^## ", text, re.MULTILINE)) == 11
    report_module = import_module("report")
    assert all(item in text for item in report_module.LIMITATIONS)
    assert report_module.INTEGRATION in text
    assert "run_llm(findings)" in text and "ai_runs.output_json" in text
    assert "PIDEFENSE_SYSTEM_PRIVATE" not in text
    assert "MẪU CI/QUICK" in text and "Chưa có kết quả" in text
    data = json.loads((artifacts / "tables/report_data.json").read_text(encoding="utf-8"))
    for relative, entry in data["inputs"].items():
        assert (
            hashlib.sha256((report.parent / relative).read_bytes()).hexdigest() == entry["sha256"]
        )
    from PIL import Image

    for path in (artifacts / "figures").glob("*.png"):
        with Image.open(path) as figure:
            assert figure.info["dpi"][0] == pytest.approx(300, abs=0.1)
    # Tampering with a persisted measurement must not silently enter the report.
    summary = artifacts / "tables/e05_latency_quick.csv"
    summary.write_text(
        summary.read_text(encoding="utf-8").replace("evaluated", "modified", 1), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="recorded hash"):
        report_module.build_report(artifacts, report)


def test_quick_rejects_live_and_scientific_options(runner: ModuleType, artifacts: Path) -> None:
    """CI mode must never load a live endpoint or a mismatched full-dataset checkpoint."""
    with pytest.raises(ValueError, match="offline"):
        runner.run(Path("configs/default.yaml"), artifacts, quick=True, live_ollama=True)
    evaluation = import_module("m8_evaluation")
    with pytest.raises(ValueError, match="scientific checkpoint"):
        evaluation.run(Path("configs/default.yaml"), artifacts, quick=True, checkpoint=Path("x"))
    with pytest.raises(ValueError, match="positive"):
        evaluation.run(Path("configs/default.yaml"), artifacts, quick=True, n_bootstrap=0)


def test_roc_tie_math(runner: ModuleType) -> None:
    """Tied scores move ROC together; endpoints and trapezoidal AUC match a hand example."""
    plotting = import_module("plotting")
    rows = [
        {
            "model": "test",
            "partition": "test",
            "technique": "original",
            "score": str(score),
            "label": str(label),
        }
        for score, label in ((0.9, 1), (0.5, 0), (0.5, 1), (0.1, 0))
    ]
    points = plotting.roc_points(rows)
    assert [(p["fpr"], p["tpr"]) for p in points] == [(0, 0), (0, 0.5), (0.5, 1), (1, 1)]
    assert np.trapezoid([p["tpr"] for p in points], [p["fpr"] for p in points]) == 0.875
