"""Render stored E02/E03 points to publication PNG and PDF at 300 dpi."""

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


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
        title="PR trên tập test; chấm: ngưỡng chọn bằng validation",
        xlim=(0, 1.02),
        ylim=(0, 1.02),
    )
    ax.legend()
    ax.grid(alpha=0.25)
    for extension in (".png", ".pdf"):
        fig.savefig(path.with_suffix(extension), dpi=300)
    plt.close(fig)


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
        title="E02 — ngưỡng chọn bằng validation; CI bootstrap",
    )
    ax.legend()
    for extension in (".png", ".pdf"):
        fig.savefig(path.with_suffix(extension), dpi=300)
    plt.close(fig)
