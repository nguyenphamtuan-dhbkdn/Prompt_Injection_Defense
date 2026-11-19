"""Bounded edit-cost search tests with controlled scores, not claimed efficacy numbers."""

from unittest.mock import Mock

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from pidefense.attacks.common import MAX_PAYLOAD_LENGTH
from pidefense.attacks.perturb import perturbations
from pidefense.eval.robustness import adversarial_min_perturbation
from pidefense.models.registry import Detector
from pidefense.rng import SimRNG


def test_minimum_observed_cost_and_query_cap() -> None:
    """A controlled single-word dependency flips at cost one, regardless of candidate order."""
    detector = Mock(spec=Detector)
    detector.predict.side_effect = lambda text: ("benign", 0.9 if "ignore" in text else 0.1)
    report = adversarial_min_perturbation(detector, "ignore instructions", SimRNG(3).generator)
    assert report.status == "flipped" and report.minimum_budget == 1
    assert report.final_score < 0.5 and report.baseline_score >= 0.5
    assert report.queries == detector.predict.call_count
    assert report.perturbed_text not in repr(report)
    detector.predict.side_effect = None
    detector.predict.return_value = ("injection", 0.9)
    report = adversarial_min_perturbation(
        detector, "ignore previous instructions", SimRNG(3).generator, max_queries=2
    )
    assert report.status == "not_found" and report.queries == 2
    assert report.minimum_budget is report.perturbed_text is None


def test_preamble_cost_initial_miss_and_equality() -> None:
    """Preamble insertion costs its word count; an initial miss is not an adversarial flip."""
    detector = Mock(spec=Detector)
    detector.predict.side_effect = lambda text: ("benign", 0.5 if text == "original" else 0.1)
    report = adversarial_min_perturbation(detector, "original", SimRNG(5).generator)
    assert report.status == "flipped" and report.minimum_budget == 2
    assert report.method == "preamble"
    detector.predict.side_effect = None
    detector.predict.return_value = ("injection", 0.1)
    report = adversarial_min_perturbation(detector, "original", SimRNG(5).generator)
    assert report.status == "already_benign" and report.queries == 1
    assert report.minimum_budget is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_budget": 0},
        {"max_queries": 1},
        {"max_queries": True},
        {"threshold": float("nan")},
        {"threshold": -1},
    ],
)
def test_invalid_search_parameters(kwargs: dict[str, float]) -> None:
    """Invalid budgets and thresholds fail before querying or logging source content."""
    with pytest.raises(ValueError):
        adversarial_min_perturbation(Mock(), "input", SimRNG(3).generator, **kwargs)


@pytest.mark.parametrize("payload", ["", "x" * (MAX_PAYLOAD_LENGTH + 1)], ids=["empty", "long"])
def test_invalid_search_payload(payload: str) -> None:
    """Reuse M4 length bounds and sanitized errors."""
    with pytest.raises(ValueError):
        adversarial_min_perturbation(Mock(), payload, SimRNG(3).generator)


def test_invalid_score_and_generator_parameters() -> None:
    """Invalid model scores are never interpreted as a successful perturbation."""
    detector = Mock(spec=Detector)
    detector.predict.return_value = ("benign", float("nan"))
    with pytest.raises(ValueError, match="score"):
        adversarial_min_perturbation(detector, "input", SimRNG(1).generator)
    for values in ({"max_budget": 0}, {"max_candidates": 0}):
        with pytest.raises(ValueError):
            list(perturbations("input", SimRNG(1).generator, **values))


@given(st.text(min_size=1, max_size=100))
@settings(max_examples=30, derandomize=True, database=None)
def test_perturbations_total_and_bounded(payload: str) -> None:
    """All bounded Unicode inputs yield distinct nonidentity candidates in cost order."""
    candidates = list(perturbations(payload, SimRNG(17).generator, max_budget=3, max_candidates=10))
    assert candidates == list(
        perturbations(payload, SimRNG(17).generator, max_budget=3, max_candidates=10)
    )
    assert len(candidates) <= 10
    assert all(c.text and c.text != payload and 1 <= c.budget <= 3 for c in candidates)
    assert len({c.text for c in candidates}) == len(candidates)
    assert [c.budget for c in candidates] == sorted(c.budget for c in candidates)
