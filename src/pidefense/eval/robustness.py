"""Bounded classifier prediction-flip search; it does not test chatbot compliance."""

from dataclasses import dataclass, field
from math import isfinite
from typing import Literal

from numpy.random import Generator

from pidefense.attacks.common import validate_payload
from pidefense.attacks.perturb import perturbations
from pidefense.models.registry import Detector


@dataclass(frozen=True)
class RobustnessReport:
    """Minimum observed word-edit cost within the tested family, never a global minimum."""

    status: Literal["flipped", "not_found", "already_benign"]
    baseline_score: float
    final_score: float
    minimum_budget: int | None
    queries: int
    method: str | None = None
    perturbed_text: str | None = field(default=None, repr=False)


def _score(detector: Detector, text: str) -> float:
    score = detector.predict(text)[1]
    if not isfinite(score) or not 0 <= score <= 1:
        raise ValueError("Detector score must be finite and in [0, 1]")
    return score


def adversarial_min_perturbation(
    detector: Detector,
    payload: str,
    rng: Generator,
    *,
    threshold: float = 0.5,
    max_budget: int = 8,
    max_queries: int = 128,
) -> RobustnessReport:
    """Search ascending budgets for score < threshold after an originally blocked input.

    Equality blocks. The baseline counts toward max_queries. Initially missed
    injections are reported separately, not counted as new adversarial flips.
    The first flip minimizes observed cost among queried candidates; truncation
    and the small lexical family preclude any claim of certified robustness.
    """
    validate_payload(payload)
    if not isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be finite and in [0, 1]")
    if type(max_queries) is not int or max_queries < 2:
        raise ValueError("max_queries must be an integer of at least two")
    if type(max_budget) is not int or max_budget < 1:
        raise ValueError("max_budget must be a positive integer")
    baseline = _score(detector, payload)
    if baseline < threshold:
        return RobustnessReport("already_benign", baseline, baseline, None, 1)
    queries = 1
    for candidate in perturbations(
        payload, rng, max_budget=max_budget, max_candidates=max_queries - 1
    ):
        score = _score(detector, candidate.text)
        queries += 1
        if score < threshold:
            return RobustnessReport(
                "flipped",
                baseline,
                score,
                candidate.budget,
                queries,
                candidate.method,
                candidate.text,
            )
    return RobustnessReport("not_found", baseline, baseline, None, queries)
