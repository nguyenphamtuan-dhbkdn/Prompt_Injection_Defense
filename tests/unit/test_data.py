"""M1 loader, stratification, provenance and leakage acceptance tests (offline)."""

import csv
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest
from datasets import Dataset, DatasetDict

from pidefense.config import DataConfig, load_config
from pidefense.data.loader import (
    BENIGN_SOURCE,
    DATASET_NAME,
    DATASET_REVISION,
    FALLBACK_PATH,
    load_benign_corpus,
    load_deepset_prompt_injections,
)
from pidefense.data.schema import PromptSample, text_identity
from pidefense.data.splits import DatasetSplits, stratified_split
from pidefense.rng import SimRNG


def _rows(n_benign: int = 100, n_injection: int = 100) -> list[PromptSample]:
    """Generate harmless placeholders; actual dataset payloads remain under data/."""
    return [
        PromptSample(f"Example category {label} item {i}", label, "unit", f"unit:{label}:{i}")
        for label, count in ((0, n_benign), (1, n_injection))
        for i in range(count)
    ]


def _all(splits: DatasetSplits) -> tuple[PromptSample, ...]:
    return splits.train + splits.val + splits.test


def _csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["text", "label", "source", "source_id"])
        writer.writeheader()
        writer.writerows(rows)


def test_default_proportions_balance_and_no_leakage() -> None:
    """Default split is 70/10/20, with class ratios preserved and complete coverage."""
    rows = _rows(120, 80)
    splits = stratified_split(rows, DataConfig(), SimRNG(42).generator)
    assert [len(splits.train), len(splits.val), len(splits.test)] == [140, 20, 40]
    report = splits.label_balance()
    for name, total in (("train", 140), ("val", 20), ("test", 40)):
        assert report[name].total == total
        assert report[name].counts == {0: int(total * 0.6), 1: int(total * 0.4)}
        assert report[name].fractions == {0: 0.6, 1: 0.4}
    groups = [
        {text_identity(row.text) for row in part}
        for part in (splits.train, splits.val, splits.test)
    ]
    assert groups[0].isdisjoint(groups[1])
    assert groups[0].isdisjoint(groups[2])
    assert groups[1].isdisjoint(groups[2])
    assert set(_all(splits)) == set(rows)
    assert len({row.source_id for row in _all(splits)}) == len(rows)


def test_same_seed_determinism_different_seeds_and_stream_progress() -> None:
    """Replay the configured stream, while different seeds/call order change assignments."""
    cfg = load_config()
    stream = SimRNG.from_config(cfg)
    before = json.dumps(stream.generator.bit_generator.state)
    first = stratified_split(_rows(), cfg.data, stream.generator)
    assert first == stratified_split(_rows(), cfg.data, SimRNG.from_config(cfg).generator)
    assert first != stratified_split(_rows(), cfg.data, SimRNG(cfg.seed + 1).generator)
    assert before != json.dumps(stream.generator.bit_generator.state)
    assert first != stratified_split(_rows(), cfg.data, stream.generator)


@pytest.mark.parametrize("val,test", [(0.1, 0.2), (0.17, 0.23), (0.0, 0.2)])
def test_uneven_class_rounding_and_zero_validation(val: float, test: float) -> None:
    """Hamilton allocation is within one row per class of each requested fraction."""
    splits = stratified_split(
        _rows(53, 27), DataConfig(val_fraction=val, test_fraction=test), SimRNG(9).generator
    )
    for name, fraction in (("train", 1 - val - test), ("val", val), ("test", test)):
        balance = splits.label_balance()[name]
        for label, count in ((0, 53), (1, 27)):
            assert abs(balance.counts[label] - count * fraction) < 1
        assert abs(balance.total - 80 * fraction) < 2
    assert len(_all(splits)) == 80
    if not val:
        assert splits.val == ()
        assert splits.label_balance()["val"].fractions == {0: 0.0, 1: 0.0}


def test_normalized_duplicates_removed_and_conflicts_rejected() -> None:
    """Case, Unicode width and whitespace duplicates cannot cross partitions."""
    rows = _rows()
    duplicate = replace(
        rows[0], text="  ＥＸＡＭＰＬＥ  category 0 ITEM 0\n", source_id="duplicate"
    )
    split = stratified_split(rows + [duplicate], DataConfig(), SimRNG(1).generator)
    assert len(_all(split)) == len(rows)
    assert rows[0] in _all(split)
    with pytest.raises(ValueError, match="conflicting labels"):
        stratified_split(rows + [replace(duplicate, label=1)], DataConfig(), SimRNG(1).generator)
    with pytest.raises(ValueError, match="source_id"):
        stratified_split(
            rows + [replace(rows[0], text="Different text")], DataConfig(), SimRNG(1).generator
        )


@pytest.mark.parametrize("rows", [[], _rows(0, 20), _rows(20, 0), _rows(2, 2)])
def test_insufficient_class_sizes_fail(rows: list[PromptSample]) -> None:
    """Impossible stratification fails clearly instead of omitting a class silently."""
    with pytest.raises(ValueError, match="Insufficient"):
        stratified_split(rows, DataConfig(), SimRNG(0).generator)


def test_committed_benign_corpus_quality() -> None:
    """Require original diverse queries with unique text/IDs and explicit attribution."""
    path = DataConfig().benign_corpus_path
    rows = load_benign_corpus(path)
    assert len(rows) >= 150
    assert len({text_identity(row.text) for row in rows}) == len(rows)
    assert len({row.source_id for row in rows}) == len(rows)
    assert all(row.label == 0 and row.source == BENIGN_SOURCE for row in rows)
    with path.open(encoding="utf-8", newline="") as handle:
        topics = {row["topic"] for row in csv.DictReader(handle)}
    assert len(topics) >= 15


def test_committed_offline_sample_and_hash() -> None:
    """Offline mode never calls the Hub; fixture matches recorded provenance hash."""
    with patch("pidefense.data.loader.load_dataset", side_effect=AssertionError("network")):
        splits = load_deepset_prompt_injections(DataConfig(), SimRNG(42).generator, offline=True)
    assert splits.origin == "offline-csv-sample"
    assert [len(splits.train), len(splits.val), len(splits.test)] == [42, 6, 12]
    assert all(row.source == DATASET_NAME for row in _all(splits))
    meta = json.loads(Path("data/raw/deepset_sample.metadata.json").read_text(encoding="utf-8"))
    assert meta["sha256"] == hashlib.sha256(FALLBACK_PATH.read_bytes()).hexdigest()
    assert meta["revision"] == DATASET_REVISION
    benign = {
        text_identity(row.text) for row in load_benign_corpus(DataConfig().benign_corpus_path)
    }
    assert benign.isdisjoint(text_identity(row.text) for row in _all(splits))


@pytest.mark.parametrize("error", [ConnectionError("unavailable"), OSError("cache unavailable")])
def test_download_failure_uses_visible_fallback(error: Exception) -> None:
    """Network/cache failures warn without including exception or payload content."""
    with patch("pidefense.data.loader.load_dataset", side_effect=error):
        with pytest.warns(RuntimeWarning, match="small offline CSV") as warnings:
            splits = load_deepset_prompt_injections(DataConfig(), SimRNG(42).generator)
    assert splits.origin == "offline-csv-sample"
    assert "unavailable" not in str(warnings[0].message).replace("Dataset unavailable", "")


def test_online_datasetdict_schema_and_pinned_cache_arguments(tmp_path: Path) -> None:
    """Exercise the datasets adapter with actual in-memory DatasetDict objects."""
    rows = _rows()
    dataset = Dataset.from_dict(
        {"text": [row.text for row in rows], "label": [row.label for row in rows]}
    )
    upstream = DatasetDict(
        {"train": dataset.select(range(180)), "test": dataset.select(range(180, 200))}
    )
    with patch("pidefense.data.loader.load_dataset", return_value=upstream) as loader:
        splits = load_deepset_prompt_injections(
            DataConfig(), SimRNG(1).generator, cache_dir=tmp_path
        )
    loader.assert_called_once_with(DATASET_NAME, revision=DATASET_REVISION, cache_dir=str(tmp_path))
    assert splits.origin == f"{DATASET_NAME}@{DATASET_REVISION}"
    assert len(_all(splits)) == 200
    assert {row.source_id.split(":")[1] for row in _all(splits)} == {"train", "test"}


@pytest.mark.parametrize(
    "bad",
    [
        DatasetDict({"train": Dataset.from_dict({"text": ["A harmless example"], "label": [2]})}),
        "unexpected",
    ],
)
def test_invalid_online_schema_does_not_fallback(bad: object) -> None:
    """Schema errors must be visible, rather than disguised as network failures."""
    with patch("pidefense.data.loader.load_dataset", return_value=bad):
        with pytest.raises(ValueError):
            load_deepset_prompt_injections(DataConfig(), SimRNG(0).generator)


def test_invalid_dataset_name_and_missing_files(tmp_path: Path) -> None:
    """Reject unsupported sources and surface missing local files."""
    with pytest.raises(ValueError, match="supports only"):
        load_deepset_prompt_injections(DataConfig(dataset_name="other"), SimRNG(0).generator)
    with pytest.raises(FileNotFoundError):
        load_deepset_prompt_injections(
            DataConfig(), SimRNG(0).generator, offline=True, fallback_path=tmp_path / "missing.csv"
        )
    with pytest.raises(FileNotFoundError):
        load_benign_corpus(tmp_path / "missing.csv")


@pytest.mark.parametrize(
    "field,value",
    [
        ("text", ""),
        ("text", " \n"),
        ("source", ""),
        ("source_id", ""),
        ("label", 2),
        ("label", True),
        ("label", "0"),
        ("technique", ""),
    ],
)
def test_sample_validation(field: str, value: object) -> None:
    """Public records validate runtime callers as well as static annotations."""
    with pytest.raises(ValueError):
        replace(_rows(1, 0)[0], **{field: value})


@pytest.mark.parametrize(
    "content",
    [
        "wrong,columns\nvalue,other\n",
        "text,label,source,source_id\n",
        "text,label,source,source_id\nhello,0,source\n",
        "text,label,source,source_id\nhello,0,source,id,extra\n",
        "text,label,source,source_id\nhello,x,source,id\n",
    ],
)
def test_malformed_csv_rejected(tmp_path: Path, content: str) -> None:
    """Headers, row width, empty files and invalid labels have clear failures."""
    path = tmp_path / "invalid.csv"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError):
        load_benign_corpus(path)


def test_benign_labels_source_duplicates_and_csv_unicode(tmp_path: Path) -> None:
    """CSV quoting and Unicode round-trip; wrong labels/sources and IDs fail."""
    path = tmp_path / "corpus.csv"
    row: dict[str, object] = {
        "text": 'Chào bạn, "hello"\nsecond line',
        "label": 0,
        "source": BENIGN_SOURCE,
        "source_id": "one",
    }
    _csv(path, [row])
    assert load_benign_corpus(path)[0].text == row["text"]
    for invalid in ({**row, "label": 1}, {**row, "source": "unknown"}):
        _csv(path, [invalid])
        with pytest.raises(ValueError, match="Benign CSV"):
            load_benign_corpus(path)
    _csv(path, [row, row])
    with pytest.raises(ValueError, match="source_id"):
        load_benign_corpus(path)
