"""Classifier bypass only: allowed attacks / evaluated attacks, by technique."""

from collections.abc import Sequence
from dataclasses import dataclass

from numpy.random import Generator

from pidefense.attacks import AttackVariant
from pidefense.eval.metrics import bootstrap_rate, validate_scores
from pidefense.models.registry import Detector


@dataclass(frozen=True)
class BypassReport:
    """Count one source variant, with multi-turn blocked if any turn is flagged."""

    technique: str
    n_attempted: int
    n_evaluated: int
    n_skipped: int
    n_bypassed: int
    bypass_rate: float | None
    ci_low: float | None
    ci_high: float | None


def score_variant(detector: Detector, variant: AttackVariant) -> float | None:
    """Use maximum per-turn injection score; skip unavailable transformations."""
    if variant.status == "skipped":
        return None
    if variant.label != 1 or not variant.turns:
        raise ValueError("Bypass evaluation requires injection variants with turns")
    scores = [detector.predict(turn)[1] for turn in variant.turns]
    validate_scores(scores, 0.5)
    return max(scores)


def summarize_bypass(
    technique: str,
    scores: Sequence[float | None],
    threshold: float,
    rng: Generator,
    *,
    n_bootstrap: int = 2000,
) -> BypassReport:
    """Count scores < threshold; exclude skips from denominator and bootstrap CI."""
    evaluated = [score for score in scores if score is not None]
    validate_scores(evaluated, threshold)
    events = [score < threshold for score in evaluated]
    low, high = bootstrap_rate(events, rng, n_bootstrap=n_bootstrap) if events else (None, None)
    return BypassReport(
        technique,
        len(scores),
        len(events),
        len(scores) - len(events),
        sum(events),
        sum(events) / len(events) if events else None,
        low,
        high,
    )
