"""E01 classical-only M2 evaluation; never substitute the small CI fixture."""

import argparse
import csv
import hashlib
import json
import platform
import subprocess
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

from pidefense.config import ClassicalModelConfig, Classifier, load_config
from pidefense.data.loader import load_deepset_prompt_injections
from pidefense.eval.metrics import compute_classification_metrics
from pidefense.models.classical import train_classical
from pidefense.rng import SimRNG

CLASSIFIERS: tuple[Classifier, ...] = ("logreg", "svm", "rf")
F1_SANITY_FLOOR = 0.80


def run(
    config_path: Path,
    output: Path,
    *,
    fixture: bool = False,
    prepared_cache: Path | None = None,
) -> None:
    """Fit train only, leave validation unused, score test once per fixed model.

    Estimator seeds are drawn after splitting from the sole PCG64 run stream.
    The fixed F1 sanity floor is checked without tuning against held-out test.
    """
    config = load_config(config_path)
    rng = SimRNG.from_config(config)
    if fixture and output.name == "e01_classical.csv":
        raise ValueError("CI fixture must not overwrite the scientific results table")
    if prepared_cache is not None:
        from m5_evaluation import load_prepared_cache

        if fixture:
            raise ValueError("Fixture cannot use a scientific prepared cache")
        splits = load_prepared_cache(prepared_cache, config, rng)
    else:
        splits = load_deepset_prompt_injections(config.data, rng.generator, offline=fixture)
    if not fixture and splits.origin == "offline-csv-sample":
        raise RuntimeError("E01 requires the full pinned dataset, not the offline CI fixture")
    rows = []
    model_metadata = {}
    for classifier in CLASSIFIERS:
        cfg = ClassicalModelConfig(config.classical_model.vectorizer, classifier)
        detector = train_classical(splits.train, cfg, rng.generator)
        metrics = compute_classification_metrics(
            [sample.label for sample in splits.test],
            detector.score_batch([sample.text for sample in splits.test]),
        )
        rows.append(
            {
                "model": classifier,
                "vectorizer": cfg.vectorizer,
                **asdict(metrics),
                "threshold": 0.5,
                "f1_sanity_floor": F1_SANITY_FLOOR,
                "sanity_pass": metrics.f1 >= F1_SANITY_FLOOR,
                "n_train": len(splits.train),
                "n_val": len(splits.val),
                "n_test": len(splits.test),
            }
        )
        model_metadata[classifier] = {
            "random_state": detector.random_state,
            "estimator_params": detector.pipeline.named_steps["classifier"].get_params(),
            "score": "sigmoid_margin_uncalibrated" if classifier == "svm" else "predict_proba",
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        "config": asdict(config),
        "seed": config.seed,
        "fixture_only": fixture,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], text=True)),
        "versions": {
            name: version(name)
            for name in ("numpy", "scipy", "scikit-learn", "datasets", "pandas", "PyYAML")
        },
        "python": platform.python_version(),
        "dataset_origin": splits.origin,
        "split_protocol": "pool upstream splits; deduplicate; seeded stratified resplit",
        "splits": {
            name: [
                {
                    "source_id": row.source_id,
                    "label": row.label,
                    "text_sha256": hashlib.sha256(row.text.encode("utf-8")).hexdigest(),
                }
                for row in getattr(splits, name)
            ]
            for name in ("train", "val", "test")
        },
        "label_balance": {name: asdict(value) for name, value in splits.label_balance().items()},
        "models": model_metadata,
        "features": {
            "word_ngram_range": [1, 2],
            "char_ngram_range": [3, 5],
            "norm_per_branch": "l2",
            "smooth_idf": True,
            "sublinear_tf": False,
            "fit_partition": "train",
            "branch_weights": "equal",
        },
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
    print(f"E01 written to {output}; persisted metrics:")
    print(output.read_text(encoding="utf-8"))
    if not fixture and not all(row["sanity_pass"] for row in rows):
        raise RuntimeError("E01 F1 sanity check failed; inspect CSV without tuning on test")


def main() -> None:
    """Run the full classical baseline from native PowerShell or Make."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/classical_tfidf.yaml"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--fixture", action="store_true", help="Offline CI regression only")
    parser.add_argument("--prepared-cache", type=Path)
    args = parser.parse_args()
    name = "e01_classical_fixture.csv" if args.fixture else "e01_classical.csv"
    run(
        args.config,
        args.output or Path("results/tables") / name,
        fixture=args.fixture,
        prepared_cache=args.prepared_cache,
    )


if __name__ == "__main__":
    main()
