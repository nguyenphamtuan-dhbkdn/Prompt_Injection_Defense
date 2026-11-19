"""M3 optional E01 DistilBERT metrics and batch-one latency; no E5 comparison yet."""

import argparse
import csv
import hashlib
import json
import platform
import subprocess
from dataclasses import asdict
from importlib import import_module
from importlib.metadata import version
from pathlib import Path

from pidefense.config import load_config
from pidefense.data.loader import load_deepset_prompt_injections
from pidefense.eval.latency import measure_latency
from pidefense.eval.metrics import compute_classification_metrics
from pidefense.models.transformer import DISTILBERT_REVISION, train_transformer
from pidefense.rng import SimRNG


def run(
    config_path: Path, output: Path, *, fixture: bool = False,
    revision: str = DISTILBERT_REVISION,
    local_files_only: bool = False,
) -> None:
    """Train once on train only, evaluate test, persist metrics and provenance.

    Validation remains unused. Fixture output is strictly for plumbing tests;
    there is no test-set tuning or claim of statistical latency confidence.
    """
    if fixture and output.name == "e01_transformer.csv":
        raise ValueError("CI fixture must not overwrite the scientific results table")
    # This optional experiment, like the model module, must fail helpfully without torch.
    try:
        torch = import_module("torch")
        import_module("transformers")
    except ImportError:
        raise ImportError("Install the optional transformer extra to run M3") from None
    config = load_config(config_path)
    rng = SimRNG.from_config(config)
    splits = load_deepset_prompt_injections(config.data, rng.generator, offline=fixture)
    if not fixture and splits.origin == "offline-csv-sample":
        raise RuntimeError("E01 requires the full pinned dataset, not the offline CI fixture")
    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(2)
        detector = train_transformer(
            splits.train, config.transformer_model, rng.generator, revision=revision,
            local_files_only=local_files_only,
        )
        texts = [sample.text for sample in splits.test]
        metrics = compute_classification_metrics(
            [sample.label for sample in splits.test], detector.score_batch(texts),
        )
        latency = measure_latency(detector, texts)
        checkpoint = output.parent / (output.stem + "_model")
        detector.save(checkpoint)
    finally:
        torch.set_num_threads(threads)
    row = {
        "model": "distilbert", **asdict(metrics), "threshold": 0.5,
        "n_train": len(splits.train), "n_val": len(splits.val), "n_test": len(splits.test),
        "latency_mean_ms": latency.mean_ms, "latency_median_ms": latency.median_ms,
        "latency_p95_ms": latency.p95_ms,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    metadata = {
        "config": asdict(config), "seed": config.seed, "fixture_only": fixture,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], text=True)),
        "versions": {name: version(name) for name in (
            "numpy", "torch", "transformers", "tokenizers", "safetensors", "datasets",
            "scikit-learn", "PyYAML",
        )},
        "python": platform.python_version(), "platform": platform.platform(),
        "processor": platform.processor(), "device": "cpu", "torch_threads": 2,
        "dataset_origin": splits.origin,
        "split_protocol": "pool upstream splits; deduplicate; seeded stratified resplit",
        "splits": {
            name: [{
                "source_id": sample.source_id, "label": sample.label,
                "text_sha256": hashlib.sha256(sample.text.encode("utf-8")).hexdigest(),
            } for sample in getattr(splits, name)] for name in ("train", "val", "test")
        },
        "label_balance": {name: asdict(value) for name, value in splits.label_balance().items()},
        "model": {
            "random_state": detector.random_state, "revision_requested": revision,
            "revision_resolved": detector.model.config._commit_hash,
            "architecture": detector.model.config.to_dict(),
            "optimizer": "torch.optim.AdamW", "lr": 5e-5, "weight_decay": 0.01,
            "max_grad_norm": 1.0, "deterministic_algorithms": True,
            "epoch_shuffle": "run numpy Generator permutation", "fit_partition": "train",
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": {
                path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted(checkpoint.iterdir()) if path.is_file()
            },
        },
        "latency": {**asdict(latency), "batch_size": 1, "includes_tokenization": True,
                    "clock": "perf_counter_ns", "excludes": ["load", "train", "warmup"]},
        "csv_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "experiment_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(Path("src/pidefense").rglob("*.py"))
        },
    }
    output.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str) + "\n", encoding="utf-8"
    )
    print(f"M3 results written to {output}")
    print(output.read_text(encoding="utf-8"))


def main() -> None:
    """Run the optional experiment directly from Windows PowerShell."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/transformer_distilbert.yaml"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--fixture", action="store_true")
    parser.add_argument("--revision", default=DISTILBERT_REVISION)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    name = "e01_transformer_fixture.csv" if args.fixture else "e01_transformer.csv"
    run(args.config, args.output or Path("results/tables") / name, fixture=args.fixture,
        revision=args.revision, local_files_only=args.local_files_only)


if __name__ == "__main__":
    main()
