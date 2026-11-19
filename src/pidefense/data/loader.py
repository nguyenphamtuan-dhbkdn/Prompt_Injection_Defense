"""Load validated source records with an explicit offline evaluation fallback."""

import csv
import warnings
from pathlib import Path
from typing import cast

from datasets import DatasetDict, load_dataset  # type: ignore[import-untyped]
from numpy.random import Generator

from pidefense.config import DataConfig
from pidefense.data.schema import Label, PromptSample
from pidefense.data.splits import DatasetSplits, stratified_split

DATASET_NAME = "deepset/prompt-injections"
DATASET_REVISION = "4f61ecb038e9c3fb77e21034b22511b523772cdd"
FALLBACK_PATH = Path("data/raw/deepset_sample.csv")
BENIGN_SOURCE = "student-curated, not scraped"


def _sample(text: object, label: object, source: str, source_id: str) -> PromptSample:
    if not isinstance(text, str):
        raise ValueError("Dataset text must be a string")
    if type(label) is not int or label not in (0, 1):
        raise ValueError("Dataset label must be integer 0 or 1")
    return PromptSample(text, cast(Label, label), source, source_id)


def _load_csv(path: Path, *, benign: bool = False) -> list[PromptSample]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"text", "label", "source", "source_id"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("CSV requires text, label, source and source_id columns")
        samples = []
        ids: set[str] = set()
        for row in reader:
            if None in row or any(row.get(key) is None for key in required):
                raise ValueError("CSV row has missing or extra fields")
            if row["label"] not in ("0", "1"):
                raise ValueError("CSV label must be 0 or 1")
            sample = _sample(row["text"], int(row["label"]), row["source"], row["source_id"])
            if sample.source_id in ids:
                raise ValueError("CSV source_id must be unique")
            if benign and (sample.label != 0 or sample.source != BENIGN_SOURCE):
                raise ValueError(
                    "Benign CSV requires label 0 and student-curated, not scraped source"
                )
            ids.add(sample.source_id)
            samples.append(sample)
    if not samples:
        raise ValueError("CSV must contain at least one sample")
    return samples


def load_benign_corpus(path: Path) -> list[PromptSample]:
    """Read separately held-out, student-curated queries; never merge into training."""
    return _load_csv(path, benign=True)


def load_deepset_prompt_injections(
    cfg: DataConfig,
    rng: Generator,
    *,
    offline: bool = False,
    fallback_path: Path = FALLBACK_PATH,
    cache_dir: Path = Path("data/raw/hf_cache"),
) -> DatasetSplits:
    """Pool upstream splits and resplit with the run's seeded Generator.

    Online mode uses datasets 3.6.0 download/cache at a pinned revision. Only
    connection/cache I/O failures select the CSV, with a payload-free warning.
    Schema errors fail rather than silently substituting a different corpus.
    Offline mode makes no Hub request. ``origin`` distinguishes the small sample
    from the full dataset; the sample is for CI, not performance claims.
    """
    if cfg.dataset_name != DATASET_NAME:
        raise ValueError("M1 supports only deepset/prompt-injections")
    if not offline:
        try:
            dataset = load_dataset(
                cfg.dataset_name, revision=DATASET_REVISION, cache_dir=str(cache_dir)
            )
        except (OSError, ConnectionError):
            warnings.warn(
                "Dataset unavailable; using small offline CSV evaluation sample",
                RuntimeWarning,
                stacklevel=2,
            )
        else:
            if not isinstance(dataset, DatasetDict):
                raise ValueError("Expected a DatasetDict of upstream splits")
            samples = [
                _sample(row["text"], row["label"], DATASET_NAME, f"deepset:{split}:{index}")
                for split in sorted(dataset)
                for index, row in enumerate(dataset[split])
            ]
            return stratified_split(samples, cfg, rng, origin=f"{DATASET_NAME}@{DATASET_REVISION}")
    return stratified_split(_load_csv(fallback_path), cfg, rng, origin="offline-csv-sample")
