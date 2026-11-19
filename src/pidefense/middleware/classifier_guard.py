"""Threshold injection-oriented scores without depending on the attack corpus."""

from dataclasses import dataclass
from math import isfinite
from typing import Literal

from pidefense.models.registry import Detector

MAX_TEXT_LENGTH = 100_000


def validate_text(text: str) -> None:
    """Reject blank or oversized input without disclosing its contents."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Input must be non-empty text")
    if len(text) > MAX_TEXT_LENGTH:
        raise ValueError("Input exceeds the text length limit")


@dataclass(frozen=True)
class GuardDecision:
    """Expose thresholded label, injection score and optional technique."""

    label: Literal["benign", "injection"]
    score: float
    technique_guess: str | None = None


class ClassifierGuard:
    """Adapt any Detector using score >= threshold, without guessing technique."""

    def __init__(self, detector: Detector, threshold: float = 0.5) -> None:
        """Keep a fitted detector and a finite probability threshold."""
        if not isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("Guard threshold must be finite and in [0, 1]")
        self.detector = detector
        self.threshold = threshold

    def check(self, text: str) -> GuardDecision:
        """Validate input and rethreshold score independently of detector label."""
        validate_text(text)
        _, score = self.detector.predict(text)
        if not isfinite(score) or not 0 <= score <= 1:
            raise ValueError("Detector score must be finite and in [0, 1]")
        return GuardDecision("injection" if score >= self.threshold else "benign", score)
