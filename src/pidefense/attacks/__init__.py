# Authorized-testing-only: research evaluation of the project's own local demo.
"""Research-only generators with explicit provenance and graceful skip reporting."""

from dataclasses import dataclass, field
from typing import Literal

from numpy.random import Generator

from pidefense.attacks.direct import direct_injection
from pidefense.attacks.indirect import DEFAULT_CARRIER, embed_in_document
from pidefense.attacks.multiturn import instruction_splitting
from pidefense.attacks.obfuscate import (
    Translator,
    base64_wrap,
    leetspeak,
    translate_roundtrip,
    zero_width_insert,
)
from pidefense.config import TECHNIQUES, Technique
from pidefense.data.schema import Label, PromptSample


@dataclass(frozen=True)
class AttackVariant:
    """Keep source identity and ordered turns; hide raw text from the default repr.

    For multi-turn input, text is the exact concatenation of turns, not a simulated
    single-message attack. Consumers must dispatch turns in order for that case.
    """

    text: str = field(repr=False)
    turns: tuple[str, ...] = field(repr=False)
    source: str
    source_id: str
    label: Label
    technique: Technique
    status: Literal["generated", "skipped"]
    skip_reason: str | None = None


def generate_variant(
    sample: PromptSample,
    technique: Technique,
    rng: Generator,
    *,
    translator: Translator | None = None,
    via_lang: str = "vi",
    carrier_template: str = DEFAULT_CARRIER,
    n_turns: int = 2,
) -> AttackVariant:
    """Apply one technique to a labeled injection, keeping provenance and skip state."""
    direct_injection(sample)
    if technique not in TECHNIQUES:
        raise ValueError("unsupported attack technique")
    reason: str | None = None
    turns: tuple[str, ...]
    if technique == "base64":
        turns = (base64_wrap(sample.text),)
    elif technique == "leetspeak":
        turns = (leetspeak(sample.text, rng),)
    elif technique == "zero_width":
        turns = (zero_width_insert(sample.text, rng),)
    elif technique == "translate":
        turns = (translate_roundtrip(sample.text, via_lang, translator=translator),)
        if turns[0] == sample.text:
            reason = "translation_unavailable_failed_or_unchanged"
    elif technique == "indirect":
        turns = (embed_in_document(sample.text, carrier_template),)
    else:
        turns = tuple(instruction_splitting(sample.text, n_turns))
    return AttackVariant(
        text="".join(turns), turns=turns, source=sample.source, source_id=sample.source_id,
        label=sample.label, technique=technique,
        status="skipped" if reason else "generated", skip_reason=reason,
    )
