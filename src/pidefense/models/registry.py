"""Family-independent detection contract; score always refers to injection."""

from typing import Literal, Protocol, runtime_checkable

Prediction = tuple[Literal["benign", "injection"], float]


@runtime_checkable
class Detector(Protocol):
    """Return (label, injection score); larger scores mean stronger suspicion."""

    def predict(self, text: str) -> Prediction:
        """Classify one nonblank string, with an injection score in [0, 1]."""
        ...
