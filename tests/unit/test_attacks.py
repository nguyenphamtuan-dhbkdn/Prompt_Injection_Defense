"""M4 contracts, Unicode properties, provenance and offline translation behavior."""

import base64
from collections.abc import Callable
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from pidefense.attacks import generate_variant
from pidefense.attacks.common import MAX_PAYLOAD_LENGTH
from pidefense.attacks.direct import direct_injection
from pidefense.attacks.indirect import embed_in_document
from pidefense.attacks.multiturn import instruction_splitting
from pidefense.attacks.obfuscate import (
    BASE64_PREFIX,
    base64_wrap,
    leetspeak,
    translate_roundtrip,
    zero_width_insert,
)
from pidefense.config import TECHNIQUES, Technique
from pidefense.data.schema import PromptSample
from pidefense.rng import SimRNG

# Include ASCII controls, all Unicode code points and lone surrogates explicitly.
PAYLOADS = st.one_of(
    st.text(alphabet=st.characters(max_codepoint=127), min_size=1, max_size=256),
    st.text(alphabet=st.integers(0, 0x10FFFF).map(chr), min_size=1, max_size=256),
)


@settings(database=None, derandomize=True)
@given(PAYLOADS)
def test_obfuscation_properties(payload: str) -> None:
    """All bounded strings survive; required transforms differ and Base64 is lossless."""
    rng = SimRNG(42).generator
    wrapped = base64_wrap(payload)
    leet = leetspeak(payload, rng)
    invisible = zero_width_insert(payload, rng)
    for output in (wrapped, leet, invisible):
        assert output and output != payload
    encoded = wrapped.removeprefix(BASE64_PREFIX)
    assert base64.b64decode(encoded, validate=True).decode("utf-8", "surrogatepass") == payload
    assert len(invisible) >= len(payload)
    # Removing exactly one inserted character recovers even preexisting U+200B.
    assert any(
        invisible[:i] + invisible[i + 1 :] == payload
        for i, char in enumerate(invisible) if char == "\u200b"
    )
    assert translate_roundtrip(payload) == payload
    assert translate_roundtrip(payload, translator=_identity_translation) == payload
    assert embed_in_document(payload).count(payload) >= 1


def _identity_translation(text: str, source_lang: str, target_lang: str) -> str:
    """Model an unchanged round-trip without dependencies or network access."""
    return text


@settings(database=None, derandomize=True)
@given(PAYLOADS.filter(lambda text: len(text) >= 3), st.sampled_from([2, 3]))
def test_splitting_is_lossless(payload: str, n_turns: int) -> None:
    """Ordered contiguous chunks reconstruct every code point including whitespace."""
    turns = instruction_splitting(payload, n_turns)
    assert len(turns) == n_turns and all(turns)
    assert "".join(turns) == payload
    assert max(map(len, turns)) - min(map(len, turns)) <= 1


def _transforms() -> tuple[Callable[[str], object], ...]:
    """Adapt each public transform to a common negative-test signature."""
    rng = SimRNG(7).generator
    return (
        base64_wrap, lambda text: leetspeak(text, rng),
        lambda text: zero_width_insert(text, rng), translate_roundtrip,
        embed_in_document, instruction_splitting,
    )


@pytest.mark.parametrize("payload", ["", "x" * (MAX_PAYLOAD_LENGTH + 1), None])
def test_invalid_payloads(payload: str) -> None:
    """Reject empty, oversized and invalid types without echoing raw input."""
    for transform in _transforms():
        with pytest.raises(ValueError, match="payload"):
            transform(payload)


def test_length_boundary() -> None:
    """The inclusive input bound works for every transform."""
    for transform in _transforms():
        assert transform("x" * MAX_PAYLOAD_LENGTH)


@pytest.mark.parametrize("technique", TECHNIQUES)
def test_source_traceability(technique: Technique) -> None:
    """Every technique preserves a real labeled dataset row's source identity."""
    from pidefense.config import DataConfig
    from pidefense.data.loader import load_deepset_prompt_injections

    splits = load_deepset_prompt_injections(DataConfig(), SimRNG(9).generator, offline=True)
    sample = next(row for row in splits.train if row.label == 1)
    variant = generate_variant(sample, technique, SimRNG(7).generator)
    assert (variant.source, variant.source_id, variant.label) == (
        sample.source, sample.source_id, sample.label,
    )
    assert variant.technique == technique
    assert variant.text == "".join(variant.turns)
    assert variant.status == ("skipped" if technique == "translate" else "generated")
    assert sample.text not in repr(variant)
    assert sample.technique is None
    if technique == "base64":
        encoded = variant.text.removeprefix(BASE64_PREFIX)
        assert base64.b64decode(encoded).decode("utf-8") == sample.text
    elif technique == "zero_width":
        assert variant.text.replace("\u200b", "") == sample.text.replace("\u200b", "")
    elif technique == "indirect":
        assert sample.text in variant.text
    elif technique == "multiturn":
        assert variant.text == sample.text and len(variant.turns) == 2
    direct = direct_injection(sample)
    assert direct.text == sample.text and direct.source_id == sample.source_id
    assert direct.technique == "direct"


def test_surface_form_spot_check() -> None:
    """Check a benign instruction: substitution preserves order and changes only the table."""
    payload = "Please write a short summary."
    transformed = leetspeak(payload, SimRNG(42).generator)
    mapping = dict(zip("aAeEiIoOsStT", "443311005577", strict=True))
    assert len(transformed) == len(payload)
    assert all(
        after in {before, mapping.get(before, before)}
        for before, after in zip(payload, transformed, strict=True)
    )
    assert transformed != payload


def test_rng_replay_and_progression() -> None:
    """The shared run stream advances, and identical call order replays exactly."""
    first, replay = SimRNG(19).generator, SimRNG(19).generator
    for _ in range(3):
        assert leetspeak("aeiost" * 20, first) == leetspeak("aeiost" * 20, replay)
        assert zero_width_insert("abc", first) == zero_width_insert("abc", replay)
    assert first.random() == replay.random()
    assert first.random() != SimRNG(19).generator.random()
    assert leetspeak("123!?\u200b", first) == "123!?\u200b "
    assert leetspeak("a", first) == "4"


def test_translation_composition_and_skips() -> None:
    """Verify language routing, changed output, exceptions and invalid backend output."""
    calls: list[tuple[str, str]] = []

    def local(text: str, source_lang: str, target_lang: str) -> str:
        """Exercise a deterministic local backend adapter, not a quality benchmark."""
        calls.append((source_lang, target_lang))
        return text + " "

    sample = PromptSample("test text", 1, "fixture", "row")
    variant = generate_variant(sample, "translate", SimRNG(1).generator, translator=local)
    assert calls == [("en", "vi"), ("vi", "en")]
    assert variant.text == sample.text + "  " and variant.status == "generated"
    assert variant.skip_reason is None

    def broken(text: str, source_lang: str, target_lang: str) -> str:
        """Simulate a backend failure that might contain sensitive text."""
        raise RuntimeError(text)

    for backend in (None, broken, _identity_translation):
        skipped = generate_variant(sample, "translate", SimRNG(1).generator, translator=backend)
        assert skipped.status == "skipped" and skipped.skip_reason
        assert skipped.text == sample.text
    for bad in ("", "x" * (MAX_PAYLOAD_LENGTH + 1), None):
        def invalid(
            text: str, source_lang: str, target_lang: str, response: object = bad
        ) -> str:
            """Return an invalid backend response for graceful fallback validation."""
            return response  # type: ignore[return-value]
        assert translate_roundtrip(sample.text, translator=invalid) == sample.text
    for lang in ("", "en", " "):
        with pytest.raises(ValueError, match="via_lang"):
            translate_roundtrip(sample.text, lang)


def test_carrier_and_turn_validation() -> None:
    """Templates use literal replacement; invalid carriers and turn counts fail clearly."""
    assert embed_in_document("{payload}", "{json}: {payload}") == "{json}: {payload}"
    for template in ("", "{payload}", "x", "{payload}{payload}", "x" * MAX_PAYLOAD_LENGTH):
        with pytest.raises(ValueError, match="carrier_template"):
            embed_in_document("abc", template)
    for count in (0, 1, 4, True, 2.5):
        with pytest.raises(ValueError, match="n_turns"):
            instruction_splitting("abcd", count)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="at least"):
        instruction_splitting("x", 2)
    benign = PromptSample("hello", 0, "fixture", "row")
    with pytest.raises(ValueError, match="injection-labeled"):
        generate_variant(benign, "base64", SimRNG(1).generator)
    with pytest.raises(ValueError, match="unsupported"):
        generate_variant(
            PromptSample("text", 1, "fixture", "row"),
            "unknown", SimRNG(1).generator,  # type: ignore[arg-type]
        )


def test_attack_modules_do_not_log() -> None:
    """M4 generators neither log raw strings nor initiate network requests."""
    import ast

    for path in Path("src/pidefense/attacks").glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert all(
                    alias.name not in {"logging", "requests", "httpx"} for alias in node.names
                )
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                assert node.func.id not in {"print", "open"}
