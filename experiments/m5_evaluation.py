"""Shared E02/E03 runner: validation-only thresholds and untouched test evaluation."""

import argparse
import csv
import hashlib
import json
import platform
import subprocess
from collections.abc import Sequence
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
from typing import Any

from datasets import Dataset
from plotting import plot_bypass, plot_pr

from pidefense.attacks import generate_variant
from pidefense.config import ClassicalModelConfig, Config, load_config
from pidefense.data.loader import (
    DATASET_NAME,
    DATASET_REVISION,
    load_benign_corpus,
    load_deepset_prompt_injections,
)
from pidefense.data.schema import PromptSample, text_identity
from pidefense.data.splits import DatasetSplits, stratified_split
from pidefense.eval.bypass_rate import summarize_bypass
from pidefense.eval.metrics import (
    bootstrap_metrics,
    bootstrap_rate,
    choose_threshold,
    compute_classification_metrics,
    false_positive_rate,
    pr_curve,
)
from pidefense.models.classical import ClassicalDetector, train_classical
from pidefense.models.registry import Detector
from pidefense.models.transformer import TransformerDetector, _backend, load_transformer
from pidefense.rng import SimRNG


def score_texts(detector: Detector, texts: Sequence[str]) -> list[float]:
    """Batch fitted models with bounded transformer CPU threads, restoring state."""
    if isinstance(detector, TransformerDetector):
        torch, _ = _backend()
        previous = torch.get_num_threads()
        try:
            torch.set_num_threads(2)
            return detector.score_batch(texts)
        finally:
            torch.set_num_threads(previous)
    if isinstance(detector, ClassicalDetector):
        return detector.score_batch(texts)
    return [detector.predict(text)[1] for text in texts]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Persist computed rows without rounding scores or serializing payloads."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def identities(samples: Sequence[PromptSample]) -> list[dict[str, Any]]:
    """Keep source identity and SHA-256, never raw text, in provenance."""
    return [
        {
            "source_id": s.source_id,
            "label": s.label,
            "text_sha256": hashlib.sha256(s.text.encode("utf-8")).hexdigest(),
        }
        for s in samples
    ]


def load_prepared_cache(directory: Path, config: Config, rng: SimRNG) -> DatasetSplits:
    """Read pinned Arrow files offline and verify all rows against E01 provenance.

    The existing E01 split ID/label/text hashes independently verify the prepared
    cache; a corrupt or different cache fails closed before fitting a model.
    """
    if directory.name != DATASET_REVISION:
        raise ValueError("Prepared cache must identify the pinned dataset revision")
    samples = []
    info = json.loads((directory / "dataset_info.json").read_text(encoding="utf-8"))
    for split in sorted(info["splits"]):
        dataset = Dataset.from_file(str(directory / f"prompt-injections-{split}.arrow"))
        if len(dataset) != info["splits"][split]["num_examples"]:
            raise ValueError("Prepared cache row count mismatch")
        samples.extend(
            PromptSample(row["text"], row["label"], DATASET_NAME, f"deepset:{split}:{i}")
            for i, row in enumerate(dataset)
        )
    splits = stratified_split(
        samples, config.data, rng.generator, origin=f"{DATASET_NAME}@{DATASET_REVISION}"
    )
    baseline = json.loads(
        Path("results/tables/e01_classical.metadata.json").read_text(encoding="utf-8")
    )
    if baseline["fixture_only"] or baseline["splits"] != {
        name: identities(getattr(splits, name)) for name in ("train", "val", "test")
    }:
        raise ValueError("Prepared cache does not match full E01 split provenance")
    return splits


def run(
    config_path: Path,
    output_dir: Path,
    *,
    fixture: bool = False,
    checkpoint: Path | None = None,
    n_bootstrap: int = 2000,
    prepared_cache: Path | None = None,
) -> None:
    """Fit M2 train only, optionally load verified M3, then run E02 and E03 together.

    A shared run fixes the same model, threshold and attack draw for both tables.
    The sole Generator drives splitting, estimator seeds, variants and bootstrap.
    """
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be positive")
    if fixture and (checkpoint is not None or prepared_cache is not None):
        raise ValueError("Fixture evaluation cannot use scientific checkpoint/cache")
    config = load_config(config_path)
    rng = SimRNG.from_config(config)
    splits = (
        load_prepared_cache(prepared_cache, config, rng)
        if prepared_cache
        else load_deepset_prompt_injections(config.data, rng.generator, offline=fixture)
    )
    if not fixture and splits.origin == "offline-csv-sample":
        raise RuntimeError("M5 requires the full pinned dataset, not the CI fixture")
    benign = load_benign_corpus(config.data.benign_corpus_path)
    if {text_identity(s.text) for s in benign} & {
        text_identity(s.text) for s in (*splits.train, *splits.val, *splits.test)
    }:
        raise ValueError("Separate benign corpus overlaps a dataset partition")
    detectors: dict[str, Detector] = {}
    model_metadata: dict[str, Any] = {}
    # Keep exactly the E01 model order and seed draws before any other randomness.
    for classifier in ("logreg", "svm", "rf"):
        model = train_classical(
            splits.train,
            ClassicalModelConfig(config.classical_model.vectorizer, classifier),
            rng.generator,
        )
        detectors[classifier] = model
        model_metadata[classifier] = {
            "random_state": model.random_state,
            "estimator_params": model.pipeline.named_steps["classifier"].get_params(),
        }
    partitions = {name: identities(getattr(splits, name)) for name in ("train", "val", "test")}
    if checkpoint is not None:
        meta_path = checkpoint.parent / (checkpoint.name.removesuffix("_model") + ".metadata.json")
        m3 = json.loads(meta_path.read_text(encoding="utf-8"))
        if m3["fixture_only"] or m3["splits"] != partitions:
            raise ValueError("M3 checkpoint provenance does not match the evaluation splits")
        if m3["config"]["transformer_model"] != asdict(config.transformer_model):
            raise ValueError("M3 transformer settings do not match the evaluation config")
        hashes = {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(checkpoint.iterdir())
            if p.is_file()
        }
        if hashes != m3["model"]["checkpoint_sha256"]:
            raise ValueError("M3 checkpoint hash mismatch")
        # Heavy imports are still lazy and guarded in the existing backend.
        torch, _ = _backend()
        original_threads = torch.get_num_threads()
        torch.set_num_threads(2)
        try:
            detectors["distilbert"] = load_transformer(checkpoint)
        finally:
            torch.set_num_threads(original_threads)
        model_metadata["distilbert"] = {
            "checkpoint": str(checkpoint),
            "sha256": hashes,
            "m3_metadata_sha256": hashlib.sha256(meta_path.read_bytes()).hexdigest(),
        }
    suffix = "_fixture" if fixture else ""
    tables = output_dir / "tables"
    figures = output_dir / "figures"
    e02: list[dict[str, Any]] = []
    e03: list[dict[str, Any]] = []
    curves: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    selections: list[dict[str, Any]] = []
    variants = [
        generate_variant(s, technique, rng.generator)
        for s in splits.test
        if s.label == 1
        for technique in config.attack.techniques
    ]
    for name, detector in detectors.items():
        val_scores = score_texts(detector, [s.text for s in splits.val])
        val_labels = [s.label for s in splits.val]
        threshold = choose_threshold(val_labels, val_scores)
        candidates = sorted(set(val_scores) | {0.5})
        for candidate in candidates:
            selections.append(
                {
                    "model": name,
                    "threshold": candidate,
                    **asdict(compute_classification_metrics(val_labels, val_scores, candidate)),
                    "selected": candidate == threshold,
                }
            )
        test_scores = score_texts(detector, [s.text for s in splits.test])
        benign_scores = score_texts(detector, [s.text for s in benign])
        for partition, samples, scores in (
            ("val", splits.val, val_scores),
            ("test", splits.test, test_scores),
            ("benign", benign, benign_scores),
        ):
            for sample, score in zip(samples, scores, strict=True):
                predictions.append(
                    {
                        "model": name,
                        "partition": partition,
                        "technique": "original",
                        "source_id": sample.source_id,
                        "label": sample.label,
                        "score": score,
                        "status": "evaluated",
                    }
                )
        for partition, labels, scores in (
            ("val", val_labels, val_scores),
            ("test", [s.label for s in splits.test], test_scores),
        ):
            for t, p, r in pr_curve(labels, scores):
                curves.append(
                    {
                        "model": name,
                        "partition": partition,
                        "threshold": t,
                        "precision": p,
                        "recall": r,
                    }
                )
        for technique in config.attack.techniques:
            selected = [v for v in variants if v.technique == technique]
            turn_scores = iter(
                score_texts(
                    detector, [turn for v in selected if v.status != "skipped" for turn in v.turns]
                )
            )
            scores = [
                max(next(turn_scores) for _ in v.turns) if v.status != "skipped" else None
                for v in selected
            ]
            for variant, score in zip(selected, scores, strict=True):
                predictions.append(
                    {
                        "model": name,
                        "partition": "test",
                        "technique": technique,
                        "source_id": variant.source_id,
                        "label": 1,
                        "score": score,
                        "status": variant.skip_reason or "evaluated",
                    }
                )
            for policy, t in (("default", 0.5), ("selected", threshold)):
                e02.append(
                    {
                        "model": name,
                        "policy": policy,
                        "threshold": t,
                        **asdict(
                            summarize_bypass(
                                technique, scores, t, rng.generator, n_bootstrap=n_bootstrap
                            )
                        ),
                    }
                )
        for policy, t in (("default", 0.5), ("selected", threshold)):
            low, high = bootstrap_rate(
                [s >= t for s in benign_scores], rng.generator, n_bootstrap=n_bootstrap
            )
            report = compute_classification_metrics([s.label for s in splits.test], test_scores, t)
            intervals = bootstrap_metrics(
                [s.label for s in splits.test],
                test_scores,
                t,
                rng.generator,
                n_bootstrap=n_bootstrap,
            )
            e03.append(
                {
                    "model": name,
                    "policy": policy,
                    "threshold": t,
                    "n_benign": len(benign),
                    "false_positives": sum(s >= t for s in benign_scores),
                    "fpr": false_positive_rate(benign_scores, t),
                    "ci_low": low,
                    "ci_high": high,
                    **{f"test_{k}": v for k, v in asdict(report).items()},
                    **{
                        f"test_{k}_ci_{side}": value
                        for k, pair in intervals.items()
                        for side, value in zip(("low", "high"), pair, strict=True)
                    },
                }
            )
    outputs = {
        f"e02_bypass_rate{suffix}.csv": e02,
        f"e03_fpr{suffix}.csv": e03,
        f"m5_predictions{suffix}.csv": predictions,
        f"e03_pr_curve{suffix}.csv": curves,
        f"e03_thresholds{suffix}.csv": selections,
    }
    for filename, rows in outputs.items():
        write_csv(tables / filename, rows)
    summary_path = tables / f"e03_operating_thresholds{suffix}.md"
    summary = [
        "# Lựa chọn ngưỡng vận hành M5\n",
        "Tự sinh từ E03; không sửa số đo bằng tay. Chọn riêng cho từng mô hình bằng "
        "F1 lớn nhất trên validation; hòa thì ưu tiên recall cao hơn rồi ngưỡng nhỏ hơn. "
        "F1 cân bằng precision/recall khi chưa có chi phí vận hành được xác định. "
        "Đây là lựa chọn nghiên cứu, không phải cam kết giới hạn FPR trong sản phẩm.\n",
    ]
    for row in e03:
        if row["policy"] != "selected":
            continue
        chosen = next(r for r in selections if r["model"] == row["model"] and r["selected"])
        summary.append(
            f"- **{row['model']}**: ngưỡng `{row['threshold']:.10g}`; "
            f"F1 validation `{chosen['f1']:.6f}`; recall validation `{chosen['recall']:.6f}`. "
            f"FPR corpus lành `{row['fpr']:.6f}` "
            f"({row['false_positives']}/{row['n_benign']}); "
            f"recall test `{row['test_recall']:.6f}`."
        )
    summary.extend(
        [
            "\nXem `e03_thresholds" + suffix + ".csv` để kiểm tra toàn bộ ứng viên, "
            "`e03_fpr" + suffix + ".csv` để so sánh với ngưỡng mặc định và xem CI. "
            "`m5_predictions" + suffix + ".csv` giữ score/ID để tái tính. "
            "Test và corpus lành chỉ dùng đánh giá sau khi khóa ngưỡng.\n",
            "CI percentile có thể suy biến khi không quan sát lỗi; không chứng minh "
            "tỷ lệ lỗi thật bằng không. Corpus tự biên soạn chưa được duyệt độc lập, "
            "không đại diện lưu lượng thực tế. Bypass chỉ là bỏ lọt qua bộ phân loại, "
            "không đo việc chatbot làm theo chỉ dẫn. Dịch bị bỏ qua khi thiếu backend; "
            "multi-turn được chặn nếu bất kỳ lượt nào có score đạt ngưỡng. "
            "F1 cao vẫn có thể bỏ lọt biến thể chưa thấy; không dùng kết quả này như "
            "bảo đảm an toàn sản phẩm.\n",
        ]
    )
    summary_path.write_text("\n".join(summary) + "\n", encoding="utf-8")
    plot_pr(curves, e03, figures / f"e03_precision_recall{suffix}")
    plot_bypass(e02, figures / f"e02_bypass_rate{suffix}")
    metadata = {
        "config": asdict(config),
        "seed": config.seed,
        "fixture_only": fixture,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], text=True)),
        "python": platform.python_version(),
        "prepared_cache_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(prepared_cache.iterdir())
            if p.is_file()
        }
        if prepared_cache
        else None,
        "dataset_origin": splits.origin,
        "versions": {
            n: version(n)
            for n in (
                "numpy",
                "scipy",
                "scikit-learn",
                "datasets",
                "matplotlib",
                "PyYAML",
                *(("torch", "transformers", "tokenizers", "safetensors") if checkpoint else ()),
            )
        },
        "splits": partitions,
        "benign": identities(benign),
        "models": model_metadata,
        "transformer_status": "evaluated" if checkpoint else "not_requested_optional_dependency",
        "threshold_policy": "validation max F1; tie: max recall, then minimum threshold",
        "bootstrap": {
            "method": "percentile",
            "confidence": 0.95,
            "replicates": n_bootstrap,
            "unit": "source row within technique; class-stratified for PR/F1",
        },
        "bypass_definition": "all turns score < threshold; classifier-only, not chatbot compliance",
        "translation": "skipped: no local translator supplied",
        "figure_dpi": 300,
        "artifact_sha256": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [
                summary_path,
                *(tables / f for f in outputs),
                *(
                    figures / (stem + suffix + ext)
                    for stem in ("e02_bypass_rate", "e03_precision_recall")
                    for ext in (".png", ".pdf")
                ),
            ]
        },
        "source_sha256": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for root in ("src/pidefense", "experiments")
            for p in sorted(Path(root).rglob("*.py"))
        },
    }
    for stem in ("e02_bypass_rate", "e03_fpr"):
        (tables / f"{stem}{suffix}.metadata.json").write_text(
            json.dumps(metadata, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
    print(f"M5 E02/E03 artifacts written under {output_dir}; fixture={fixture}")


def main() -> None:
    """Run both linked evaluations directly from native PowerShell."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/classical_tfidf.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--fixture", action="store_true")
    parser.add_argument("--checkpoint", type=Path, help="Optional verified M3 checkpoint directory")
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument(
        "--prepared-cache",
        type=Path,
        help="Pinned Arrow cache directory; verifies full E01 split hashes",
    )
    args = parser.parse_args()
    run(
        args.config,
        args.output_dir,
        fixture=args.fixture,
        checkpoint=args.checkpoint,
        n_bootstrap=args.n_bootstrap,
        prepared_cache=args.prepared_cache,
    )
