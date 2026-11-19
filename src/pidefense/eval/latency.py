"""Synchronous batch-one wall-clock inference timing, reusable by E5."""

from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite
from time import perf_counter_ns

import numpy as np
from numpy.random import Generator

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


@dataclass(frozen=True)
class LatencyInterval:
    """Percentile bootstrap CI for mean latency, clustering repeated timings by input."""

    mean_ci_low_ms: float
    mean_ci_high_ms: float
    confidence: float
    n_bootstrap: int
    n_texts: int


def bootstrap_latency(
    report: LatencyReport, rng: Generator, *, n_bootstrap: int = 2000, confidence: float = 0.95
) -> LatencyInterval:
    """Resample input clusters with replacement, retaining all repeats for each input.

    Sampling repeated calls independently would overstate effective sample size.
    This CI describes input-mix uncertainty conditional on observed timings, not
    training, hardware, operating-system scheduling or temporal autocorrelation.
    NumPy linear quantiles give the equal-tailed percentile interval.
    """
    if type(n_bootstrap) is not int or n_bootstrap < 1:
        raise ValueError("n_bootstrap must be a positive integer")
    if not isfinite(confidence) or not 0 < confidence < 1:
        raise ValueError("confidence must be finite and between zero and one")
    if (
        type(report.repeats) is not int
        or report.repeats < 1
        or not report.samples_ms
        or len(report.samples_ms) % report.repeats
        or any(not isfinite(t) or t < 0 for t in report.samples_ms)
    ):
        raise ValueError("Latency samples must be finite, nonnegative and match repeats")
    means = np.asarray(report.samples_ms).reshape(report.repeats, -1).mean(axis=0)
    estimates = [
        float(np.mean(means[rng.integers(0, len(means), size=len(means))]))
        for _ in range(n_bootstrap)
    ]
    alpha = (1 - confidence) / 2
    low, high = np.quantile(estimates, [alpha, 1 - alpha])
    return LatencyInterval(float(low), float(high), confidence, n_bootstrap, len(means))


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
        tuple(timings),
        float(np.mean(timings)),
        float(np.median(timings)),
        float(np.percentile(timings, 95)),
        warmup,
        repeats,
    )
