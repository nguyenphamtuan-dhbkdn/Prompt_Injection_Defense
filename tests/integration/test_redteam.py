"""Offline E04 regression, human review, real HTTP serialization and opt-in live trials."""

import csv
import hashlib
import json
import logging
import os
from collections.abc import Iterator
from dataclasses import replace
from importlib import import_module
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType
from unittest.mock import Mock
from urllib.error import URLError

import pytest

from pidefense.attacks import generate_variant
from pidefense.config import TECHNIQUES
from pidefense.data.schema import PromptSample
from pidefense.demo.ollama_client import OllamaClient, OllamaError
from pidefense.middleware.classifier_guard import ClassifierGuard
from pidefense.models.registry import Detector
from pidefense.rng import SimRNG


@pytest.fixture
def e04(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Import the standalone experiment using its documented script import path."""
    monkeypatch.syspath_prepend(str(Path("experiments").resolve()))
    return import_module("e04_indirect_injection_vs_output_filter")


@pytest.fixture
def artifacts() -> Iterator[Path]:
    """Keep test transcripts inside results even during testing, per raw-content policy."""
    with TemporaryDirectory(prefix="pytest_e04_", dir="results") as directory:
        yield Path(directory)


def guard(score: float = 0.1) -> ClassifierGuard:
    """Use the Detector contract to control input-guard branches, independent of ML quality."""
    detector = Mock(spec=Detector)
    detector.predict.return_value = ("benign", score)
    return ClassifierGuard(detector)


def test_visibility_block_error_and_skip(e04: ModuleType, caplog: pytest.LogCaptureFixture) -> None:
    """Distinguish input prevention, withheld leaks, visible leaks and backend failures."""
    caplog.set_level(logging.DEBUG)
    variant = generate_variant(
        PromptSample("private source payload", 1, "test", "source:1"),
        "base64",
        SimRNG(3).generator,
    )
    client = Mock()
    client.chat_messages.return_value = "PIDEFENSE_SYSTEM_PRIVATE"
    blocked = e04.evaluate_trial(variant, guard(0.9), client, middleware="on", seed=4)
    client.chat_messages.assert_not_called()
    assert blocked["status"] == "blocked" and blocked["automated_raw_violation"] is None
    assert blocked["automated_final_violation"] is False
    held = e04.evaluate_trial(variant, guard(), client, middleware="on", seed=4)
    assert held["rule"] == "output_filter" and held["n_visible"] == 0
    assert held["automated_raw_violation"] is True
    assert held["automated_final_violation"] is False
    exposed = e04.evaluate_trial(variant, guard(0.9), client, middleware="off", seed=4)
    assert exposed["automated_final_violation"] is True and exposed["n_visible"] == 1
    client.chat_messages.side_effect = RuntimeError("private backend error")
    error = e04.evaluate_trial(variant, guard(), client, middleware="off", seed=4)
    assert error["status"] == "error" and error["automated_final_violation"] is None
    error_on = e04.evaluate_trial(variant, guard(), client, middleware="on", seed=4)
    assert error_on["status"] == "error" and error_on["automated_final_violation"] is None
    calls = client.chat_messages.call_count
    skipped = e04.evaluate_trial(
        replace(variant, status="skipped", skip_reason="unavailable"),
        guard(),
        client,
        middleware="off",
        seed=4,
    )
    assert skipped["automated_final_violation"] is None
    assert client.chat_messages.call_count == calls
    summary = e04.summarize([blocked, held, exposed, error, skipped], {})
    off = next(r for r in summary if r["technique"] == "base64" and r["middleware"] == "off")
    assert off["n_errors"] == off["n_skipped"] == off["n_completed"] == 1
    assert off["automated_success_rate"] == 1
    missing = next(r for r in summary if r["technique"] == "translate")
    assert missing["automated_success_rate"] is None
    assert "private source payload" not in caplog.text
    assert "private backend error" not in caplog.text
    assert "PIDEFENSE_SYSTEM_PRIVATE" not in caplog.text


@pytest.mark.parametrize("mode", ["off", "on"])
def test_multiturn_http_and_isolation(
    e04: ModuleType, mode: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exercise actual Ollama JSON with isolated history and identical explicit seed options."""
    requests = []

    def open_reply(request: object, *, timeout: float) -> BytesIO:
        """Capture request body while returning a fresh nonstreaming reply each time."""
        requests.append(json.loads(request.data))
        assert timeout == 2
        return BytesIO(
            json.dumps(
                {
                    "done": True,
                    "message": {"role": "assistant", "content": "Ordinary answer."},
                }
            ).encode()
        )

    opener = Mock()
    opener.open.side_effect = open_reply
    monkeypatch.setattr("pidefense.demo.ollama_client.build_opener", lambda *args: opener)
    client = OllamaClient(timeout=2)
    variant = generate_variant(
        PromptSample("instruction split over ordered turns", 1, "test", "source:2"),
        "multiturn",
        SimRNG(7).generator,
        n_turns=3,
    )
    result = e04.evaluate_trial(variant, guard(), client, middleware=mode, seed=9)
    assert result["status"] == "allowed" and result["n_responses"] == 3
    assert [len(r["messages"]) for r in requests] == [2, 4, 6]
    assert [m["role"] for m in requests[-1]["messages"]] == [
        "system",
        "user",
        "assistant",
        "user",
        "assistant",
        "user",
    ]
    for i, request in enumerate(requests):
        assert variant.turns[i] in request["messages"][-1]["content"]
        assert request["options"] == {
            "seed": 9,
            "temperature": 0.0,
            "num_predict": 256,
            "num_ctx": 8192,
        }
    e04.evaluate_trial(variant, guard(), client, middleware=mode, seed=9)
    assert len(requests[3]["messages"]) == 2
    with pytest.raises(OllamaError, match="Invalid local conversation"):
        client.chat_messages([])


def test_indirect_document_boundary(e04: ModuleType) -> None:
    """ON screens/escapes the document separately; OFF sends its original corpus text."""
    variant = generate_variant(
        PromptSample("document <unsafe> content", 1, "test", "source:3"),
        "indirect",
        SimRNG(1).generator,
    )
    client = Mock()
    client.chat_messages.return_value = "Ordinary answer."
    e04.evaluate_trial(variant, guard(), client, middleware="on", seed=1)
    content = client.chat_messages.call_args.args[0][-1]["content"]
    assert "<untrusted_document>" in content and "&lt;unsafe&gt;" in content
    e04.evaluate_trial(variant, guard(), client, middleware="off", seed=1)
    assert variant.text in client.chat_messages.call_args.args[0][-1]["content"]


def test_fixture_regression_and_provenance(e04: ModuleType, artifacts: Path) -> None:
    """Run every technique offline, reproduce golden CSV, and recompute paired rates."""
    fixture = import_module("e04_offline_fixture")
    path = fixture.generate_fixture(artifacts)
    bundle = json.loads(path.read_text(encoding="utf-8"))
    records, meta = bundle["trials"], bundle["metadata"]
    assert meta["fixture_only"] and meta["transport"] == "injected_mock"
    assert meta["config"]["seed"] == meta["seed"]
    n_sources = sum(s["label"] == 1 for s in meta["splits"]["test"])
    assert len(records) == n_sources * len(TECHNIQUES) * 2
    assert {r["technique"] for r in records} == set(TECHNIQUES)
    assert not any(r["status"] in ("error", "skipped") for r in records)
    for source in {r["source_id"] for r in records}:
        for technique in TECHNIQUES:
            pair = [r for r in records if r["source_id"] == source and r["technique"] == technique]
            assert len(pair) == 2
            assert pair[0]["payload_sha256"] == pair[1]["payload_sha256"]
            assert pair[0]["generation_seed"] == pair[1]["generation_seed"]
    table = artifacts / "tables/e04_indirect_injection_vs_output_filter_fixture.csv"
    with table.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    with Path("results/tables/e04_indirect_injection_vs_output_filter_fixture.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        assert rows == list(csv.DictReader(handle))
    for row in rows:
        selected = [
            r
            for r in records
            if r["technique"] == row["technique"] and r["middleware"] == row["middleware"]
        ]
        assert float(row["automated_final_violation_rate"]) == sum(
            r["automated_final_violation"] for r in selected
        ) / len(selected)
        assert int(row["n_paired_completed"]) == n_sources
    report_meta = json.loads(table.with_suffix(".metadata.json").read_text(encoding="utf-8"))
    for relative, sha in report_meta["artifact_sha256"].items():
        assert hashlib.sha256((e04.ROOT / relative).read_bytes()).hexdigest() == sha
    assert not (artifacts / "tables/e04_indirect_injection_vs_output_filter.csv").exists()
    appendix = (
        artifacts / "e04_indirect_injection_vs_output_filter_fixture.appendix.md"
    ).read_text(encoding="utf-8")
    assert "MÔ PHỎNG CI" in appendix and "PIDEFENSE_SYSTEM_PRIVATE" not in appendix
    assert all("#### " + t in appendix for t in TECHNIQUES)


def test_human_review_disagreements_and_stale_hash(e04: ModuleType, artifacts: Path) -> None:
    """Manual labels stay pending until explicitly supplied; false negatives are discussed."""
    path = import_module("e04_offline_fixture").generate_fixture(artifacts)
    records = json.loads(path.read_text(encoding="utf-8"))["trials"]
    record = next(r for r in records if r["n_visible"] and not r["automated_final_violation"])
    review = {f: "" for f in e04.REVIEW_FIELDS}
    review.update(
        {
            "trial_id": record["trial_id"],
            "transcript_sha256": record["transcript_sha256"],
            "manual_violation": "yes",
            "reviewer": "synthetic-test-reviewer",
            "reason": "Synthetic disagreement exercises reporting, not a real human judgment.",
            "sanitized_prompt": "[test prompt redacted]",
            "sanitized_response": "[test reply redacted]",
        }
    )
    reviews_path = artifacts / "review.csv"
    e04.write_csv(reviews_path, [review])
    reviews = e04.read_reviews(reviews_path, records)
    row = next(
        r
        for r in e04.summarize(records, reviews)
        if r["technique"] == record["technique"] and r["middleware"] == record["middleware"]
    )
    assert row["n_human_reviewed"] == row["n_disagreements"] == 1
    assert row["n_manual_pending"] > 0
    report = artifacts / "report.md"
    report.write_text("before\n<!-- E04:BEGIN -->\n<!-- E04:END -->\nafter", encoding="utf-8")
    e04.save_reports(path, reviews_path=reviews_path, appendix_path=report)
    assert "khác heuristic" in report.read_text(encoding="utf-8")
    for field, value, match in (
        ("transcript_sha256", "stale", "hash mismatch"),
        ("manual_violation", "automatic", "Manual verdict"),
        ("reviewer", "", "requires reviewer"),
        ("trial_id", "unknown", "unknown or duplicate"),
    ):
        e04.write_csv(reviews_path, [{**review, field: value}])
        with pytest.raises(ValueError, match=match):
            e04.read_reviews(reviews_path, records)
    e04.write_csv(reviews_path, [review, review])
    with pytest.raises(ValueError, match="duplicate"):
        e04.read_reviews(reviews_path, records)
    uncertain = {**review, "manual_violation": "uncertain"}
    e04.write_csv(reviews_path, [uncertain])
    summary = e04.summarize(records, e04.read_reviews(reviews_path, records))
    row = next(
        r
        for r in summary
        if r["technique"] == record["technique"] and r["middleware"] == record["middleware"]
    )
    assert row["n_manual_uncertain"] == 1 and row["n_human_reviewed"] == 0
    assert row["n_manual_pending"] + row["n_manual_evaluated"] + 1 == row["n_completed"]


def test_full_corpus_and_raw_path_requirements(e04: ModuleType, artifacts: Path) -> None:
    """Reject subsets, injected scientific transports, fixture/cache mixing and unsafe paths."""
    config = artifacts / "subset.yaml"
    config.write_text("attack:\n  techniques: [base64]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="full technique"):
        e04.run(config, artifacts, fixture=True)
    with pytest.raises(ValueError, match="fixture mode"):
        e04.run(Path("configs/default.yaml"), artifacts, client=Mock())
    with pytest.raises(ValueError, match="scientific prepared cache"):
        e04.run(Path("configs/default.yaml"), artifacts, fixture=True, prepared_cache=Path("x"))
    with pytest.raises(ValueError, match="repository results"):
        e04.results_path(Path("docs/raw_outputs"))
    with pytest.raises(ValueError, match="inside data or results"):
        e04.load_translation_cache(Path("docs/translations.json"))
    cache = artifacts / "translation.json"
    cache.write_text('[{"text": "only one field"}]', encoding="utf-8")
    with pytest.raises(ValueError, match="schema"):
        e04.load_translation_cache(cache)


def test_missing_ollama_is_not_a_success(
    e04: ModuleType, artifacts: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unreachable localhost produces error counts and blank OFF rates, never fabricated zeros."""
    opener = Mock()
    opener.open.side_effect = URLError("private server error")
    monkeypatch.setattr("pidefense.demo.ollama_client.build_opener", lambda *args: opener)
    path = e04.run(Path("configs/default.yaml"), artifacts, fixture=True)
    records = json.loads(path.read_text(encoding="utf-8"))["trials"]
    rows = e04.summarize(records, {})
    assert any(r["n_errors"] for r in rows)
    assert all(r["automated_success_rate"] is None for r in rows if r["middleware"] == "off")
    assert all(r["automated_final_violation"] is None for r in records if r["status"] == "error")
    assert any(r["n_skipped"] for r in rows if r["technique"] == "translate")


@pytest.mark.slow
def test_live_full_local_ollama(e04: ModuleType, artifacts: Path) -> None:
    """Opt-in full scientific run requires the user's own running localhost demo model."""
    if os.environ.get("PIDEFENSE_RUN_LIVE_OLLAMA") != "1":
        pytest.skip("Set PIDEFENSE_RUN_LIVE_OLLAMA=1 with the configured local model running")
    prepared = os.environ.get("PIDEFENSE_PREPARED_CACHE")
    translation = os.environ.get("PIDEFENSE_TRANSLATION_CACHE")
    path = e04.run(
        Path("configs/default.yaml"),
        artifacts,
        prepared_cache=Path(prepared) if prepared else None,
        translation_cache=Path(translation) if translation else None,
    )
    records = json.loads(path.read_text(encoding="utf-8"))["trials"]
    assert records and not any(r["status"] == "error" for r in records)
    assert all(r["status"] != "skipped" for r in records if r["technique"] != "translate")
