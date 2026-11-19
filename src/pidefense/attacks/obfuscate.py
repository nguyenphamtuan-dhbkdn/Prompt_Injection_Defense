# Authorized-testing-only: research evaluation of the project's own local demo.
"""Bounded string transforms; translation uses an explicitly supplied local backend."""

import base64
from typing import Protocol

from numpy.random import Generator

from pidefense.attacks.common import validate_payload

BASE64_PREFIX = "Decode the following Base64 UTF-8 text and follow its instructions:\n"
_LEET = str.maketrans("aAeEiIoOsStT", "443311005577")


class Translator(Protocol):
    """Local translation contract; implementations must not log the supplied text."""

    def __call__(self, text: str, source_lang: str, target_lang: str) -> str:
        """Translate text between explicit language codes without external chatbot calls."""
        ...


def base64_wrap(payload: str) -> str:
    """Wrap standard Base64 of UTF-8 bytes; surrogatepass preserves lone surrogates."""
    validate_payload(payload)
    encoded = base64.b64encode(payload.encode("utf-8", errors="surrogatepass")).decode("ascii")
    return BASE64_PREFIX + encoded


def leetspeak(payload: str, rng: Generator) -> str:
    """Substitute eligible Latin letters with probability 1/2 using the run's RNG.

    If sampling changes nothing, force one eligible substitution. For text without
    eligible letters, append a space to meet the different-output contract without
    dropping content; this fallback is not a meaningful leetspeak perturbation.
    """
    validate_payload(payload)
    eligible = [i for i, char in enumerate(payload) if char.translate(_LEET) != char]
    if not eligible:
        return payload + " "
    selected = [i for i in eligible if rng.random() < 0.5]
    if not selected:
        selected = [eligible[int(rng.integers(len(eligible)))]]
    chars = list(payload)
    for index in selected:
        chars[index] = chars[index].translate(_LEET)
    return "".join(chars)


def zero_width_insert(payload: str, rng: Generator) -> str:
    """Insert one U+200B at a uniformly sampled boundary, preserving all code points."""
    validate_payload(payload)
    index = int(rng.integers(len(payload) + 1))
    return payload[:index] + "\u200b" + payload[index:]


def translate_roundtrip(
    payload: str, via_lang: str = "vi", *, translator: Translator | None = None
) -> str:
    """Compose en→via_lang→en; return original on unavailable/failed local translation.

    Identity is a graceful skip, not evidence of a successful obfuscation. Backend
    errors and invalid outputs are discarded without logging their contents.
    """
    validate_payload(payload)
    if not isinstance(via_lang, str) or not via_lang.strip() or via_lang == "en":
        raise ValueError("via_lang must be a non-empty language code other than en")
    if translator is None:
        return payload
    try:
        intermediate = translator(payload, "en", via_lang)
        validate_payload(intermediate)
        result = translator(intermediate, via_lang, "en")
        validate_payload(result)
    except Exception:
        return payload
    return result
