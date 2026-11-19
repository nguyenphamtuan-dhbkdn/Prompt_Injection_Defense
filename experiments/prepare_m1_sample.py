"""Rebuild the offline fixture from pinned upstream data, never synthetic payloads."""

import csv
import hashlib
import json
import platform
import subprocess
from importlib.metadata import version
from pathlib import Path

from datasets import DatasetDict, load_dataset

from pidefense.data.loader import DATASET_NAME, DATASET_REVISION, FALLBACK_PATH
from pidefense.data.schema import text_identity


def main() -> None:
    """Select the first 30 unique rows per label in sorted split/row order.

    Selection is deterministic, not random; this fixture only tests plumbing.
    Full scientific experiments must use the complete upstream dataset.
    """
    dataset = load_dataset(DATASET_NAME, revision=DATASET_REVISION, cache_dir="data/raw/hf_cache")
    if not isinstance(dataset, DatasetDict):
        raise ValueError("Expected DatasetDict")
    counts = {0: 0, 1: 0}
    seen: set[str] = set()
    rows = []
    for split in sorted(dataset):
        for index, row in enumerate(dataset[split]):
            label = row["label"]
            key = text_identity(row["text"])
            if counts[label] >= 30 or key in seen:
                continue
            seen.add(key)
            counts[label] += 1
            rows.append(
                {
                    "text": row["text"],
                    "label": label,
                    "source": DATASET_NAME,
                    "source_id": f"deepset:{split}:{index}",
                }
            )
    if counts != {0: 30, 1: 30}:
        raise ValueError("Not enough upstream samples")
    with FALLBACK_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["text", "label", "source", "source_id"])
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        "dataset": DATASET_NAME,
        "revision": DATASET_REVISION,
        "url": f"https://huggingface.co/datasets/{DATASET_NAME}/tree/{DATASET_REVISION}",
        "selection": "first 30 unique texts per label, sorted upstream splits then row order",
        "seed": None,
        "randomness": "none",
        "label_counts": counts,
        "sha256": hashlib.sha256(FALLBACK_PATH.read_bytes()).hexdigest(),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "versions": {name: version(name) for name in ("datasets", "numpy", "pyarrow")},
        "python": platform.python_version(),
        "purpose": "offline CI fixture only; not a representative evaluation dataset",
    }
    Path("data/raw/deepset_sample.metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print("Offline CSV and provenance metadata written under data/raw/.")


if __name__ == "__main__":
    main()
