"""Render persisted experimental CSVs with consistent fonts/colors, PNG/PDF at 300 dpi."""

import argparse
import csv
import hashlib
import json
import subprocess
from importlib.metadata import version
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.prop_cycle": plt.cycler(color=["#26547c", "#ef8354", "#2a9d8f", "#9467bd"]),
        "savefig.dpi": 300,
    }
)


def _save(fig: Figure, path: Path) -> None:
    """Save the same figure as raster PNG and vector PDF and release it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    for extension in (".png", ".pdf"):
        metadata = {"CreationDate": None, "ModDate": None} if extension == ".pdf" else None
        fig.savefig(path.with_suffix(extension), dpi=300, metadata=metadata)
    plt.close(fig)


def read_rows(path: Path) -> list[dict[str, str]]:
    """Read a persisted CSV rather than embedding experimental values in plotting code."""
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def plot_pr(
    curves: list[dict[str, Any]],
    reports: list[dict[str, Any]],
    path: Path,
) -> None:
    """Plot held-out PR as steps; selected markers use validation-only thresholds."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5), layout="constrained")
    for report in reports:
        if report["policy"] != "selected":
            continue
        rows = [r for r in curves if r["partition"] == "test" and r["model"] == report["model"]]
        (line,) = ax.step(
            [r["recall"] for r in rows],
            [r["precision"] for r in rows],
            where="post",
            label=report["model"],
        )
        ax.scatter(
            report["test_recall"],
            report["test_precision"],
            color=line.get_color(),
            marker="o",
            s=45,
        )
    ax.set(
        xlabel="Độ bao phủ (recall)",
        ylabel="Độ chính xác (precision)",
        title="PR test; ngưỡng validation" + (" — MẪU CI" if "_fixture" in path.name else ""),
        xlim=(0, 1.02),
        ylim=(0, 1.02),
    )
    ax.legend()
    ax.grid(alpha=0.25)
    _save(fig, path)


def plot_bypass(rows: list[dict[str, Any]], path: Path) -> None:
    """Show per-technique mean bypass and bootstrap CI; missing techniques omitted."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
    techniques = list(dict.fromkeys(r["technique"] for r in rows if r["n_evaluated"]))
    models = list(dict.fromkeys(r["model"] for r in rows))
    width = 0.8 / len(models)
    for i, model in enumerate(models):
        selected = [
            next(
                r
                for r in rows
                if r["policy"] == "selected" and r["model"] == model and r["technique"] == t
            )
            for t in techniques
        ]
        ax.bar(
            [j + (i - (len(models) - 1) / 2) * width for j in range(len(techniques))],
            [r["bypass_rate"] for r in selected],
            width=width,
            label=model,
            yerr=[
                [max(0, r["bypass_rate"] - r["ci_low"]) for r in selected],
                [max(0, r["ci_high"] - r["bypass_rate"]) for r in selected],
            ],
            capsize=2,
        )
    ax.set(
        xticks=range(len(techniques)),
        xticklabels=techniques,
        ylim=(0, 1.05),
        ylabel="Tỷ lệ vượt bộ phân loại",
        title="E02 — ngưỡng validation; CI bootstrap"
        + (" — MẪU CI" if "_fixture" in path.name else ""),
    )
    ax.legend()
    _save(fig, path)


def roc_points(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    """ROC thresholds handle ties together: TPR=TP/P and FPR=FP/N, starting at origin."""
    points = []
    for model in dict.fromkeys(r["model"] for r in rows):
        selected = [
            r
            for r in rows
            if r["model"] == model and r["partition"] == "test" and r["technique"] == "original"
        ]
        positives = sum(int(r["label"]) for r in selected)
        negatives = len(selected) - positives
        if not positives or not negatives:
            continue
        points.append({"model": model, "threshold": None, "fpr": 0.0, "tpr": 0.0})
        for threshold in sorted({float(r["score"]) for r in selected}, reverse=True):
            predicted = [r for r in selected if float(r["score"]) >= threshold]
            tp = sum(int(r["label"]) for r in predicted)
            points.append(
                {
                    "model": model,
                    "threshold": threshold,
                    "fpr": (len(predicted) - tp) / negatives,
                    "tpr": tp / positives,
                }
            )
    return points


def render_all(output_dir: Path, *, quick: bool = False) -> list[Path]:
    """Regenerate all available figures from CSVs; return only artifacts actually produced."""
    suffix = "_fixture" if quick else ""
    m8_suffix = "_quick" if quick else ""
    tables, figures = output_dir / "tables", output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    produced: list[Path] = []
    qualifier = " — MẪU CI" if quick else ""

    def save(fig: Figure, stem: str) -> None:
        """Record both actual files for the run manifest."""
        path = figures / stem
        _save(fig, path)
        produced.extend(path.with_suffix(ext) for ext in (".png", ".pdf"))

    baseline = tables / f"e01_classical{suffix}.csv"
    if baseline.exists():
        rows = read_rows(baseline)
        fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
        ax.bar([r["model"] for r in rows], [float(r["f1"]) for r in rows])
        ax.set(ylabel="F1 held-out", ylim=(0, 1.05), title="E01 — classical" + qualifier)
        save(fig, "e01_baseline" + suffix)
    predictions = tables / f"m5_predictions{suffix}.csv"
    if predictions.exists():
        points = roc_points(read_rows(predictions))
        if points:
            roc_path = tables / f"e01_roc_curve{suffix}.csv"
            with roc_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(points[0]))
                writer.writeheader()
                writer.writerows(points)
            produced.append(roc_path)
            fig, ax = plt.subplots(figsize=(6, 5), layout="constrained")
            for model in dict.fromkeys(p["model"] for p in points):
                selected = [p for p in points if p["model"] == model]
                ax.plot([p["fpr"] for p in selected], [p["tpr"] for p in selected], label=model)
            ax.plot([0, 1], [0, 1], linestyle=":", color="gray")
            ax.set(xlabel="FPR", ylabel="TPR", title="E01 — ROC held-out" + qualifier)
            ax.legend()
            save(fig, "e01_roc" + suffix)
    bypass = tables / f"e02_bypass_rate{suffix}.csv"
    if bypass.exists():
        rows = read_rows(bypass)
        converted = [
            {
                **r,
                "n_evaluated": int(r["n_evaluated"]),
                **{k: float(r[k]) if r[k] else None for k in ("bypass_rate", "ci_low", "ci_high")},
            }
            for r in rows
        ]
        plot_bypass(converted, figures / ("e02_bypass_rate" + suffix))
        produced.extend(figures / ("e02_bypass_rate" + suffix + e) for e in (".png", ".pdf"))
    fpr = tables / f"e03_fpr{suffix}.csv"
    curves = tables / f"e03_pr_curve{suffix}.csv"
    if fpr.exists() and curves.exists():
        reports = [
            {**r, **{k: float(r[k]) for k in ("test_recall", "test_precision")}}
            for r in read_rows(fpr)
            if r["policy"] == "selected"
        ]
        points = [
            {**r, **{k: float(r[k]) for k in ("precision", "recall")}} for r in read_rows(curves)
        ]
        plot_pr(points, reports, figures / ("e03_precision_recall" + suffix))
        produced.extend(figures / ("e03_precision_recall" + suffix + e) for e in (".png", ".pdf"))
    e04 = tables / f"e04_indirect_injection_vs_output_filter{suffix}.csv"
    if e04.exists():
        rows = read_rows(e04)
        techniques = list(dict.fromkeys(r["technique"] for r in rows))
        fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
        for offset, mode in ((-0.2, "off"), (0.2, "on")):
            selected = [
                r for r in rows if r["middleware"] == mode and r["automated_final_violation_rate"]
            ]
            ax.bar(
                [techniques.index(r["technique"]) + offset for r in selected],
                [float(r["automated_final_violation_rate"]) for r in selected],
                width=0.4,
                label=mode,
            )
        ax.set(
            xticks=range(len(techniques)),
            xticklabels=techniques,
            ylim=(0, 1.05),
            ylabel="Tín hiệu vi phạm user-visible",
            title="E04 — heuristic" + qualifier,
        )
        ax.legend()
        save(fig, "e04_violation" + suffix)
    for stem, metric, title, ylabel in (
        ("e05_latency", "mean_ms", "E05 — latency batch một", "Thời gian (ms)"),
        ("e06_robustness", "flip_rate", "E06 — flip trong nguồn ban đầu bị chặn", "Tỷ lệ flip"),
    ):
        path = tables / (stem + m8_suffix + ".csv")
        if not path.exists():
            continue
        rows = [r for r in read_rows(path) if r["status"] == "evaluated" and r[metric]]
        fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
        values = [float(r[metric]) for r in rows]
        errors = (
            None
            if metric != "mean_ms"
            else [
                [max(0, v - float(r["mean_ci_low_ms"])) for r, v in zip(rows, values, strict=True)],
                [
                    max(0, float(r["mean_ci_high_ms"]) - v)
                    for r, v in zip(rows, values, strict=True)
                ],
            ]
        )
        ax.bar([r["model"] for r in rows], values, yerr=errors, capsize=3)
        ax.set(ylabel=ylabel, title=title + qualifier)
        if metric == "flip_rate":
            ax.set_ylim(0, 1.05)
        save(fig, stem + m8_suffix)
    provenance = {
        "config": {"dpi": 300, "font": "DejaVu Sans", "quick": quick},
        "seed": None,
        "randomness": "none; rendering only, measurements are unchanged",
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], text=True)),
        "versions": {"matplotlib": version("matplotlib")},
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "input_sha256": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(tables.glob("*.csv"))
        },
        "artifact_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in produced},
    }
    # Preserve numerical provenance while explicitly recording re-rendered M5 assets.
    for stem in ("e02_bypass_rate", "e03_fpr"):
        path = tables / (stem + suffix + ".metadata.json")
        if not path.exists():
            continue
        metadata = json.loads(path.read_text(encoding="utf-8"))
        prior = metadata.setdefault("prior_figure_sha256", {})
        for name, sha in metadata.get("artifact_sha256", {}).items():
            artifact = Path(name)
            if artifact.suffix not in (".png", ".pdf") or not artifact.exists():
                continue
            current = hashlib.sha256(artifact.read_bytes()).hexdigest()
            if current != sha:
                prior.setdefault(name, sha)
                metadata["artifact_sha256"][name] = current
        metadata["figure_rendering"] = provenance
        path.write_text(json.dumps(metadata, indent=2, default=str) + "\n", encoding="utf-8")
    (
        figures / ("plotting_fixture.metadata.json" if quick else "plotting.metadata.json")
    ).write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    return produced


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    render_all(args.output_dir, quick=args.quick)
