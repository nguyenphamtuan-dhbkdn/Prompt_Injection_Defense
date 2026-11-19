"""Validated prompt records and conservative text identity for leakage checks."""

from dataclasses import dataclass
from typing import Literal
from unicodedata import normalize

Label = Literal[0, 1]


def text_identity(text: str) -> str:
    """Normalize NFKC, case and whitespace for conservative duplicate detection."""
    return " ".join(normalize("NFKC", text).casefold().split())


@dataclass(frozen=True)
class PromptSample:
    """Keep original text and provenance; label 0 is benign and 1 is injection."""

    text: str
    label: Label
    source: str
    source_id: str
    technique: str | None = None

    def __post_init__(self) -> None:
        """Reject empty fields and nonbinary labels without disclosing payloads."""
        for name in ("text", "source", "source_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not text_identity(value):
                raise ValueError(f"{name} must be a non-empty string")
        if type(self.label) is not int or self.label not in (0, 1):
            raise ValueError("label must be integer 0 or 1")
        if self.technique is not None and (
            not isinstance(self.technique, str) or not self.technique.strip()
        ):
            raise ValueError("technique must be None or a non-empty string")
