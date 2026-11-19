"""Shared E05/E06 runner: same held-out sources, hardware, fitted models and RNG."""

import argparse
import hashlib
import json
import platform
import subprocess
from contextlib import ExitStack
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
from m5_evaluation import identities, load_prepared_cache, score_texts, write_csv
from threadpoolctl import threadpool_info, threadpool_limits

from pidefense.attacks.perturb import MAX_SITES, MAX_SUBSTITUTIONS, PREAMBLES, SYNONYMS
from pidefense.config import ClassicalModelConfig, Config, load_config
from pidefense.data.loader import load_deepset_prompt_injections
from pidefense.data.splits import DatasetSplits
from pidefense.eval.latency import bootstrap_latency, measure_latency
from pidefense.eval.metrics import choose_threshold
from pidefense.eval.robustness import adversarial_min_perturbation
from pidefense.models.classical import train_classical
from pidefense.models.registry import Detector
from pidefense.models.transformer import _backend, load_transformer
from pidefense.rng import SimRNG


def verified_checkpoint(checkpoint: Path, splits: DatasetSplits, config: Config) -> dict[str, Any]:
    """Check full M3 split/config/checkpoint hashes before loading optional heavy code."""
    meta_path = checkpoint.parent / (checkpoint.name.removesuffix("_model") + ".metadata.json")
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    partitions = {n: identities(getattr(splits, n)) for n in ("train", "val", "test")}
    if metadata["fixture_only"] or metadata["splits"] != partitions:
        raise ValueError("Checkpoint does not match full evaluation partitions")
    if metadata["config"]["transformer_model"] != asdict(config.transformer_model):
        raise ValueError("Checkpoint transformer config mismatch")
    hashes = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(checkpoint.iterdir())
        if p.is_file()
    }
    if hashes != metadata["model"]["checkpoint_sha256"]:
        raise ValueError("Checkpoint hash mismatch")
    return {
        "path": str(checkpoint),
        "checkpoint_sha256": hashes,
        "metadata_sha256": hashlib.sha256(meta_path.read_bytes()).hexdigest(),
    }


def run(
    config_path: Path,
    output_dir: Path,
    *,
    quick: bool = False,
    checkpoint: Path | None = None,
    prepared_cache: Path | None = None,
    n_bootstrap: int | None = None,
) -> None:
    """Measure batch-one inference and bounded flips, saving raw measurements and provenance.

    Quick mode uses the committed offline data fixture, fewer repeated calls,
    bootstrap draws and search queries, and different output names. No optional
    checkpoint is silently substituted. All stochastic steps share one Generator.
    """
    root_results = Path(__file__).resolve().parents[1] / "results"
    if not output_dir.resolve().is_relative_to(root_results):
        raise ValueError("E05/E06 output directory must stay inside repository results")
    if quick and (checkpoint is not None or prepared_cache is not None):
        raise ValueError("Quick fixture cannot use a scientific checkpoint/cache")
    count = n_bootstrap if n_bootstrap is not None else (100 if quick else 2000)
    if type(count) is not int or count < 1:
        raise ValueError("n_bootstrap must be positive")
    config = load_config(config_path)
    rng = SimRNG.from_config(config)
    splits = (
        load_prepared_cache(prepared_cache, config, rng)
        if prepared_cache
        else (load_deepset_prompt_injections(config.data, rng.generator, offline=quick))
    )
    if not quick and splits.origin == "offline-csv-sample":
        raise RuntimeError("Full E05/E06 require the pinned full dataset")
    detectors: dict[str, Detector] = {}
    models: dict[str, Any] = {}
    # Match E01/M5 model initialization order before any other random draw.
    for name in ("logreg", "svm", "rf"):
        detector = train_classical(
            splits.train,
            ClassicalModelConfig(config.classical_model.vectorizer, name),
            rng.generator,
        )
        detectors[name] = detector
        models[name] = {
            "random_state": detector.random_state,
            "params": detector.pipeline.named_steps["classifier"].get_params(),
        }
    thresholds = {
        name: choose_threshold(
            [s.label for s in splits.val], score_texts(detector, [s.text for s in splits.val])
        )
        for name, detector in detectors.items()
    }
    latency_rows: list[dict[str, Any]] = []
    search_rows: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []
    timing_rows: list[dict[str, Any]] = []
    summary: list[dict[str, Any]] = []
    limit = min(len(splits.test), 8 if quick else len(splits.test))
    selected = sorted(int(i) for i in rng.generator.choice(len(splits.test), limit, replace=False))
    samples = [splits.test[i] for i in selected]
    texts = [s.text for s in samples]
    max_budget, max_queries, repeats = (3, 24, 2) if quick else (8, 128, 5)
    with ExitStack() as stack:
        if checkpoint is not None:
            models["distilbert"] = verified_checkpoint(checkpoint, splits, config)
            torch, _ = _backend()
            original_threads = torch.get_num_threads()
            stack.callback(torch.set_num_threads, original_threads)
            torch.set_num_threads(2)
            detectors["distilbert"] = load_transformer(checkpoint)
            thresholds["distilbert"] = choose_threshold(
                [s.label for s in splits.val],
                score_texts(detectors["distilbert"], [s.text for s in splits.val]),
            )
        stack.enter_context(threadpool_limits(limits=2))
        pools = threadpool_info()
        # Shared text order, randomized model order to reduce fixed ordering bias.
        order = [list(detectors)[int(i)] for i in rng.generator.permutation(len(detectors))]
        for name in order:
            detector = detectors[name]
            timing = measure_latency(detector, texts, warmup=3, repeats=repeats)
            interval = bootstrap_latency(timing, rng.generator, n_bootstrap=count)
            latency_rows.append(
                {
                    "model": name,
                    "status": "evaluated",
                    "n_texts": len(texts),
                    "n_measurements": len(timing.samples_ms),
                    "mean_ms": timing.mean_ms,
                    "median_ms": timing.median_ms,
                    "p95_ms": timing.p95_ms,
                    **asdict(interval),
                    "warmup": timing.warmup,
                    "repeats": repeats,
                }
            )
            for i, duration in enumerate(timing.samples_ms):
                sample = samples[i % len(samples)]
                timing_rows.append(
                    {
                        "model": name,
                        "source_id": sample.source_id,
                        "text_sha256": hashlib.sha256(sample.text.encode()).hexdigest(),
                        "repeat": i // len(samples),
                        "duration_ms": duration,
                    }
                )
        # All models search the same source list; RNG draw order is recorded.
        for name, detector in detectors.items():
            selected_rows = []
            for sample in splits.test:
                if sample.label != 1:
                    continue
                report = adversarial_min_perturbation(
                    detector,
                    sample.text,
                    rng.generator,
                    threshold=thresholds[name],
                    max_budget=max_budget,
                    max_queries=max_queries,
                )
                row = {
                    "model": name,
                    "source_id": sample.source_id,
                    "threshold": thresholds[name],
                    **{k: v for k, v in asdict(report).items() if k != "perturbed_text"},
                    "candidate_sha256": hashlib.sha256(report.perturbed_text.encode()).hexdigest()
                    if report.perturbed_text is not None
                    else None,
                }
                search_rows.append(row)
                selected_rows.append(row)
                if report.perturbed_text is not None:
                    details.append(
                        {
                            "model": name,
                            "source_id": sample.source_id,
                            "original": sample.text,
                            **asdict(report),
                        }
                    )
            eligible = [r for r in selected_rows if r["status"] != "already_benign"]
            found = [r for r in eligible if r["status"] == "flipped"]
            budgets = [r["minimum_budget"] for r in found]
            summary.append(
                {
                    "model": name,
                    "status": "evaluated",
                    "threshold": thresholds[name],
                    "n_injections": len(selected_rows),
                    "n_eligible": len(eligible),
                    "n_already_benign": len(selected_rows) - len(eligible),
                    "n_flipped": len(found),
                    "flip_rate": len(found) / len(eligible) if eligible else None,
                    "minimum_observed_budget": min(budgets) if budgets else None,
                    "median_successful_budget": float(np.median(budgets)) if budgets else None,
                    "max_budget": max_budget,
                    "max_queries": max_queries,
                }
            )
    if checkpoint is None:
        latency_rows.append(
            {
                k: "distilbert"
                if k == "model"
                else "not_requested_optional_checkpoint"
                if k == "status"
                else None
                for k in latency_rows[0]
            }
        )
        summary.append(
            {
                k: "distilbert"
                if k == "model"
                else "not_requested_optional_checkpoint"
                if k == "status"
                else None
                for k in summary[0]
            }
        )
    suffix = "_quick" if quick else ""
    tables = output_dir / "tables"
    outputs = []
    for stem, rows in (
        ("e05_latency", latency_rows),
        ("e05_latency_samples", timing_rows),
        ("e06_robustness", summary),
        ("e06_search", search_rows),
    ):
        path = tables / f"{stem}{suffix}.csv"
        write_csv(path, rows)
        outputs.append(path)
    # Found examples contain raw corpus text: constrain persistence to results/.
    examples = tables / f"e06_examples{suffix}.json"
    examples.write_text(json.dumps(details, indent=2) + "\n", encoding="utf-8")
    outputs.append(examples)
    metadata = {
        "config": asdict(config),
        "seed": config.seed,
        "fixture_only": quick,
        "quick": quick,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], text=True)),
        "versions": {
            n: version(n)
            for n in (
                "numpy",
                "scipy",
                "scikit-learn",
                "datasets",
                "threadpoolctl",
                "matplotlib",
                "PyYAML",
                *(("torch", "transformers") if checkpoint else ()),
            )
        },
        "python": platform.python_version(),
        "dataset_origin": splits.origin,
        "splits": {n: identities(getattr(splits, n)) for n in ("train", "val", "test")},
        "models": models,
        "thresholds": thresholds,
        "hardware": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "cpu_threads": 2,
            "batch_size": 1,
            "threadpools": pools,
        },
        "latency": {
            "samples": identities(samples),
            "model_order": order,
            "repeats": repeats,
            "warmup": 3,
            "n_bootstrap": count,
            "confidence": 0.95,
            "bootstrap_unit": "input cluster, retaining repeats",
            "includes": "predict/tokenization/host score; excludes load/train/warmup",
        },
        "robustness": {
            "max_sites": MAX_SITES,
            "max_substitutions": MAX_SUBSTITUTIONS,
            "synonyms": SYNONYMS,
            "preambles": PREAMBLES,
            "max_budget": max_budget,
            "max_queries": max_queries,
            "budget": "substituted words + inserted preamble words",
            "claim": "minimum observed among queried candidates; semantics unverified",
        },
        "artifact_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in outputs},
        "source_sha256": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for d in ("src/pidefense", "experiments")
            for p in sorted(Path(d).rglob("*.py"))
        },
    }
    for stem in ("e05_latency", "e06_robustness"):
        (tables / f"{stem}{suffix}.metadata.json").write_text(
            json.dumps(metadata, indent=2, default=str) + "\n", encoding="utf-8"
        )
    print(f"E05/E06 written under {output_dir}; quick={quick}; checkpoint={checkpoint is not None}")


def main() -> None:
    """Run shared experiments from PowerShell without GNU make."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--prepared-cache", type=Path)
    parser.add_argument("--n-bootstrap", type=int)
    args = parser.parse_args()
    run(
        args.config,
        args.output_dir,
        quick=args.quick,
        checkpoint=args.checkpoint,
        prepared_cache=args.prepared_cache,
        n_bootstrap=args.n_bootstrap,
    )
