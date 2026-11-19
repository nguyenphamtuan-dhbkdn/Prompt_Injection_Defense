"""Binary detection metrics with injection as the positive class."""

from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite

from sklearn.metrics import roc_auc_score  # type: ignore[import-untyped]


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
