"""Synchronous batch-one wall-clock inference timing, reusable by E5."""

from collections.abc import Sequence
from dataclasses import dataclass
from time import perf_counter_ns

import numpy as np

from pidefense.models.registry import Detector


@dataclass(frozen=True)
class LatencyReport:
    """Raw milliseconds and their arithmetic mean, median and 95th percentile."""

    samples_ms: tuple[float, ...]
    mean_ms: float
    median_ms: float
    p95_ms: float
    warmup: int
    repeats: int


def measure_latency(
    detector: Detector, texts: Sequence[str], *, warmup: int = 3, repeats: int = 3
) -> LatencyReport:
    """Time predict including tokenization, excluding load/train and warmup.

    Each repetition visits all texts in order, with batch size one. Detectors
    must return host scores (synchronizing accelerator work before returning).
    Quantiles use NumPy's default linear interpolation; no CI is claimed here.
    """
    if not texts or any(not isinstance(text, str) or not text.strip() for text in texts):
        raise ValueError("Latency requires non-empty texts")
    if type(warmup) is not int or warmup < 0 or type(repeats) is not int or repeats < 1:
        raise ValueError("warmup must be nonnegative and repeats must be positive integers")
    for index in range(warmup):
        detector.predict(texts[index % len(texts)])
    timings = []
    for _ in range(repeats):
        for text in texts:
            start = perf_counter_ns()
            detector.predict(text)
            timings.append((perf_counter_ns() - start) / 1_000_000)
    return LatencyReport(
        tuple(timings), float(np.mean(timings)), float(np.median(timings)),
        float(np.percentile(timings, 95)), warmup, repeats,
    )
