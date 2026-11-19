"""Binary detection metrics with injection as the positive class."""

from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite

import numpy as np
from numpy.random import Generator
from sklearn.metrics import (  # type: ignore[import-untyped]
    precision_recall_curve,
    roc_auc_score,
)


@dataclass(frozen=True)
class MetricsReport:
    """Precision TP/(TP+FP), recall TP/(TP+FN), F1 harmonic mean, and ROC-AUC."""

    precision: float
    recall: float
    f1: float
    roc_auc: float


def compute_classification_metrics(
    y_true: Sequence[int], y_score: Sequence[float], threshold: float = 0.5
) -> MetricsReport:
    """Threshold scores with >=; AUC uses ranking, including half credit for ties.

    Undefined precision/F1 become zero; both true classes are required for AUC.
    """
    if len(y_true) != len(y_score) or set(y_true) != {0, 1}:
        raise ValueError("Metrics require equal lengths and both binary true classes")
    if not isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be finite and in [0, 1]")
    if any(not isfinite(score) or not 0 <= score <= 1 for score in y_score):
        raise ValueError("Scores must be finite and in [0, 1]")
    predicted = [score >= threshold for score in y_score]
    tp = sum(label == 1 and pred for label, pred in zip(y_true, predicted, strict=True))
    fp = sum(label == 0 and pred for label, pred in zip(y_true, predicted, strict=True))
    fn = sum(label == 1 and not pred for label, pred in zip(y_true, predicted, strict=True))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn)
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    return MetricsReport(precision, recall, f1, float(roc_auc_score(y_true, y_score)))


def validate_scores(scores: Sequence[float], threshold: float) -> None:
    """Require finite injection scores and an inclusive threshold in [0, 1]."""
    if not isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be finite and in [0, 1]")
    if any(not isfinite(score) or not 0 <= score <= 1 for score in scores):
        raise ValueError("Scores must be finite and in [0, 1]")


def false_positive_rate(scores: Sequence[float], threshold: float) -> float:
    """Return FP/(FP+TN) on an explicitly benign-only score sequence."""
    validate_scores(scores, threshold)
    if not scores:
        raise ValueError("FPR requires a nonempty benign corpus")
    return sum(score >= threshold for score in scores) / len(scores)


def pr_curve(
    labels: Sequence[int],
    scores: Sequence[float],
) -> list[tuple[float | None, float, float]]:
    """Return (threshold, precision, recall); the terminal (1, 0) has no threshold."""
    compute_classification_metrics(labels, scores)
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    return [
        (float(t), float(p), float(r))
        for t, p, r in zip(thresholds, precision[:-1], recall[:-1], strict=True)
    ] + [(None, 1.0, 0.0)]


def choose_threshold(labels: Sequence[int], scores: Sequence[float]) -> float:
    """Maximize validation F1; break ties by recall, then the smaller threshold.

    Candidates are observed validation scores plus 0.5. No test/benign corpus
    information enters this choice; F1 gives equal weight to precision/recall.
    """
    compute_classification_metrics(labels, scores)
    candidates = sorted(set(scores) | {0.5})
    return max(
        candidates,
        key=lambda t: (
            compute_classification_metrics(labels, scores, t).f1,
            compute_classification_metrics(labels, scores, t).recall,
            -t,
        ),
    )


def bootstrap_rate(
    events: Sequence[bool],
    rng: Generator,
    *,
    n_bootstrap: int = 2000,
) -> tuple[float, float]:
    """Percentile 95% CI of mean(binary events), resampling rows with replacement.

    Empty denominators are undefined. All-zero/all-one samples yield degenerate
    intervals; this is conditional sampling uncertainty, not a safety guarantee.
    """
    if not events or n_bootstrap < 1:
        raise ValueError("Bootstrap requires events and positive n_bootstrap")
    values = np.asarray(events, dtype=float)
    means = [
        float(rng.choice(values, size=len(values), replace=True).mean()) for _ in range(n_bootstrap)
    ]
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def bootstrap_metrics(
    labels: Sequence[int],
    scores: Sequence[float],
    threshold: float,
    rng: Generator,
    *,
    n_bootstrap: int = 2000,
) -> dict[str, tuple[float, float]]:
    """Stratified percentile 95% CIs for precision/recall/F1 at a fixed threshold.

    Resample within each true class, holding observed prevalence fixed and
    avoiding undefined AUC samples. Threshold selection is not repeated.
    """
    compute_classification_metrics(labels, scores, threshold)
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be positive")
    groups = [[i for i, label in enumerate(labels) if label == cls] for cls in (0, 1)]
    values: dict[str, list[float]] = {name: [] for name in ("precision", "recall", "f1")}
    for _ in range(n_bootstrap):
        indices = [
            int(i) for group in groups for i in rng.choice(group, size=len(group), replace=True)
        ]
        report = compute_classification_metrics(
            [labels[i] for i in indices],
            [scores[i] for i in indices],
            threshold,
        )
        for name in values:
            values[name].append(getattr(report, name))
    return {
        name: (float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975)))
        for name, v in values.items()
    }
