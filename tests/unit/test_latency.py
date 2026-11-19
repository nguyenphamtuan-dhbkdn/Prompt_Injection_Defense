"""Clock-controlled timing tests without optional dependencies."""

from unittest.mock import patch

import pytest

from pidefense.eval.latency import measure_latency
from pidefense.models.registry import Prediction
from pidefense.rng import SimRNG


class CountingDetector:
    """Count invocations to check warmup and measured repetitions."""

    def __init__(self) -> None:
        """Start with no predictions."""
        self.calls = 0

    def predict(self, text: str) -> Prediction:
        """Return a constant valid detection result."""
        self.calls += 1
        return "benign", 0.1


def test_latency_clock_and_warmup() -> None:
    """Warmup is untimed; each text is measured once per repetition."""
    detector = CountingDetector()
    with patch("pidefense.eval.latency.perf_counter_ns", side_effect=[0, 1_000_000, 2, 3_000_002]):
        report = measure_latency(detector, ["a", "b"], warmup=3, repeats=1)
    assert detector.calls == 5
    assert report.samples_ms == (1, 3)
    assert report.mean_ms == report.median_ms == 2
    assert report.p95_ms == pytest.approx(2.9)


@pytest.mark.parametrize(
    "texts,warmup,repeats",
    [
        ([], 0, 1),
        ([""], 0, 1),
        (["x"], -1, 1),
        (["x"], 0, 0),
        (["x"], True, 1),
        (["x"], 0, False),
    ],
)
def test_invalid_timing_inputs(texts: list[str], warmup: int, repeats: int) -> None:
    """Reject inputs that would produce empty or misleading timings."""
    with pytest.raises(ValueError):
        measure_latency(CountingDetector(), texts, warmup=warmup, repeats=repeats)


@pytest.mark.statistical
def test_latency_bootstrap_clusters_and_reproducibility() -> None:
    """Keep input clusters intact: complementary repeats yield a degenerate CI of their mean."""
    from pidefense.eval.latency import LatencyReport, bootstrap_latency

    report = LatencyReport((1, 1, 9, 9), 5, 5, 9, 0, 2)
    first = bootstrap_latency(report, SimRNG(4).generator, n_bootstrap=100)
    assert first.mean_ci_low_ms == first.mean_ci_high_ms == 5
    assert first.n_texts == 2
    assert first == bootstrap_latency(report, SimRNG(4).generator, n_bootstrap=100)
    report = LatencyReport((1, 3, 1, 3), 2, 2, 3, 0, 2)
    interval = bootstrap_latency(report, SimRNG(4).generator, n_bootstrap=100)
    assert interval.mean_ci_low_ms == 1 and interval.mean_ci_high_ms == 3


def test_latency_ci_validation() -> None:
    """Undefined timing arrays, confidence and resampling budgets are rejected."""
    from pidefense.eval.latency import LatencyReport, bootstrap_latency

    for samples, repeats in (((), 1), ((float("nan"),), 1), ((-1,), 1), ((1,), 2), ((1,), 0)):
        with pytest.raises(ValueError):
            bootstrap_latency(LatencyReport(samples, 1, 1, 1, 0, repeats), SimRNG(3).generator)
    report = LatencyReport((1,), 1, 1, 1, 0, 1)
    for count, confidence in ((0, 0.95), (1, 1), (1, float("inf"))):
        with pytest.raises(ValueError):
            bootstrap_latency(report, SimRNG(3).generator, n_bootstrap=count, confidence=confidence)
