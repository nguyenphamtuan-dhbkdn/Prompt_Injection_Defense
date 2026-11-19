"""Hand-computed rate/PR examples, threshold leakage prevention and E02/E03 regression."""

import csv
import hashlib
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from pidefense.attacks import generate_variant
from pidefense.data.schema import PromptSample
from pidefense.eval.bypass_rate import score_variant, summarize_bypass
from pidefense.eval.metrics import (
    bootstrap_metrics,
    bootstrap_rate,
    choose_threshold,
    false_positive_rate,
    pr_curve,
)
from pidefense.models.registry import Prediction


class ToyDetector:
    """Expose the common detector contract with score equal to parsed text."""

    def predict(self, text: str) -> Prediction:
        """Return the default label while tests verify arbitrary thresholds."""
        score = float(text)
        return ("injection" if score >= 0.5 else "benign", score)


def test_rates_and_pr_by_hand() -> None:
    """Check FP/TN, threshold equality and PR endpoints against exact fractions."""
    assert false_positive_rate([0.1, 0.5, 0.7, 0.2], 0.5) == 0.5
    assert false_positive_rate([0.0, 1.0], 0) == 1
    assert false_positive_rate([0.0, 1.0], 1) == 0.5
    points = pr_curve([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8])
    assert points == [
        (0.1, 0.5, 1.0),
        (0.35, 2 / 3, 1.0),
        (0.4, 0.5, 0.5),
        (0.8, 1.0, 0.5),
        (None, 1.0, 0.0),
    ]
    assert choose_threshold([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]) == 0.35
    assert choose_threshold([0, 1], [0.5, 0.5]) == 0.5


@pytest.mark.parametrize(
    "scores,threshold",
    [([], 0.5), ([float("nan")], 0.5), ([1.1], 0.5), ([0.2], -1), ([0.2], float("inf"))],
)
def test_invalid_rates(scores: list[float], threshold: float) -> None:
    """Undefined denominators and invalid scores fail without payload text."""
    with pytest.raises(ValueError):
        false_positive_rate(scores, threshold)


def test_bypass_skips_and_multiturn() -> None:
    """All turns must pass; equality blocks; skipped translation is never a bypass."""
    rng = np.random.default_rng(3)
    sample = PromptSample("some test input", 1, "test", "source:1")
    variant = generate_variant(sample, "multiturn", rng)
    variant = replace(variant, turns=("0.1", "0.8"))
    assert score_variant(ToyDetector(), variant) == 0.8
    skipped = replace(variant, status="skipped", skip_reason="unavailable")
    assert score_variant(ToyDetector(), skipped) is None
    with pytest.raises(ValueError):
        score_variant(ToyDetector(), replace(variant, label=0))
    with pytest.raises(ValueError):
        score_variant(ToyDetector(), replace(variant, turns=("nan",)))
    report = summarize_bypass("multiturn", [0.1, 0.5, None, 0.9], 0.5, rng, n_bootstrap=30)
    assert (report.n_attempted, report.n_evaluated, report.n_skipped) == (4, 3, 1)
    assert report.n_bypassed == 1
    assert report.bypass_rate == 1 / 3
    missing = summarize_bypass("translate", [None, None], 0.5, rng)
    assert missing.bypass_rate is missing.ci_low is missing.ci_high is None
    assert missing.n_evaluated == 0


@pytest.mark.statistical
def test_bootstrap_seeded_and_degenerate() -> None:
    """Percentile resampling is reproducible and admits degenerate empirical CIs."""
    first = bootstrap_rate([True, False, False, True], np.random.default_rng(7))
    assert first == bootstrap_rate([True, False, False, True], np.random.default_rng(7))
    assert first[0] <= 0.5 <= first[1]
    assert bootstrap_rate([False] * 5, np.random.default_rng(7)) == (0, 0)
    assert bootstrap_rate([True] * 5, np.random.default_rng(7)) == (1, 1)
    labels, scores = [0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]
    expected = {name: (1.0, 1.0) for name in ("precision", "recall", "f1")}
    assert (
        bootstrap_metrics(labels, scores, 0.5, np.random.default_rng(8), n_bootstrap=30) == expected
    )
    for events, count in (([], 2), ([True], 0)):
        with pytest.raises(ValueError):
            bootstrap_rate(events, np.random.default_rng(7), n_bootstrap=count)
    with pytest.raises(ValueError):
        bootstrap_metrics(labels, scores, 0.5, np.random.default_rng(8), n_bootstrap=0)


def test_m5_fixture_regression(tmp_path: Path) -> None:
    """Run offline with small bootstrap; recompute counts and golden scientific isolation."""
    subprocess.run(
        [
            sys.executable,
            "experiments/e03_benign_false_positive_rate.py",
            "--fixture",
            "--n-bootstrap",
            "30",
            "--output-dir",
            str(tmp_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    def read(name: str) -> list[dict[str, str]]:
        """Read a generated table using its explicit fixture-only name."""
        with (tmp_path / "tables" / name).open(encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))

    e02 = read("e02_bypass_rate_fixture.csv")
    e03 = read("e03_fpr_fixture.csv")
    predictions = read("m5_predictions_fixture.csv")
    thresholds = read("e03_thresholds_fixture.csv")
    for row in e02:
        relevant = [
            p
            for p in predictions
            if p["model"] == row["model"] and p["technique"] == row["technique"] and p["score"]
        ]
        assert int(row["n_evaluated"]) == len(relevant)
        if relevant:
            assert float(row["bypass_rate"]) == sum(
                float(p["score"]) < float(row["threshold"]) for p in relevant
            ) / len(relevant)
        else:
            assert row["bypass_rate"] == ""
    for row in e03:
        relevant = [
            p for p in predictions if p["model"] == row["model"] and p["partition"] == "benign"
        ]
        assert float(row["fpr"]) == sum(
            float(p["score"]) >= float(row["threshold"]) for p in relevant
        ) / len(relevant)
        if row["policy"] == "selected":
            selected = [
                t for t in thresholds if t["model"] == row["model"] and t["selected"] == "True"
            ]
            assert len(selected) == 1
            assert row["threshold"] == selected[0]["threshold"]
    for name in ("e02_bypass_rate", "e03_fpr"):
        meta = json.loads((tmp_path / "tables" / f"{name}_fixture.metadata.json").read_text())
        assert meta["fixture_only"] is True
        assert meta["seed"] == meta["config"]["seed"]
        assert not (tmp_path / "tables" / f"{name}.csv").exists()
        for path, digest in meta["artifact_sha256"].items():
            assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
    with Path("results/tables/e03_fpr_fixture.csv").open(encoding="utf-8", newline="") as handle:
        golden = list(csv.DictReader(handle))
    assert e03 == golden
    with Path("results/tables/e02_bypass_rate_fixture.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        assert e02 == list(csv.DictReader(handle))
    # PNG physical density must be 300 dpi; PDF is a vector figure saved at the same setting.
    from PIL import Image

    with Image.open(tmp_path / "figures/e03_precision_recall_fixture.png") as image:
        assert image.info["dpi"] == pytest.approx((300, 300), abs=0.02)
    assert (tmp_path / "figures/e03_precision_recall_fixture.pdf").read_bytes().startswith(b"%PDF")
