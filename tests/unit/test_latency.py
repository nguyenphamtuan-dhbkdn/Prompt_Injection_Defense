"""Clock-controlled timing tests without optional dependencies."""

from unittest.mock import patch

import pytest

from pidefense.eval.latency import measure_latency
from pidefense.models.registry import Prediction


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


@pytest.mark.parametrize("texts,warmup,repeats", [
    ([], 0, 1), ([""], 0, 1), (["x"], -1, 1), (["x"], 0, 0),
    (["x"], True, 1), (["x"], 0, False),
])
def test_invalid_timing_inputs(texts: list[str], warmup: int, repeats: int) -> None:
    """Reject inputs that would produce empty or misleading timings."""
    with pytest.raises(ValueError):
        measure_latency(CountingDetector(), texts, warmup=warmup, repeats=repeats)
