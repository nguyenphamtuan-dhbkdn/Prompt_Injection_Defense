"""Run reproducible E01–E06 stages; quick mode is fully offline and explicitly a fixture."""

import argparse
import hashlib
import json
import platform
import subprocess
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

import e01_baseline_classical_vs_transformer as e01
import e04_indirect_injection_vs_output_filter as e04
import m5_evaluation
import m8_evaluation
from e04_offline_fixture import generate_fixture
from plotting import render_all
from report import build_report

from pidefense.config import load_config


def run(
    config_path: Path,
    output_dir: Path,
    *,
    quick: bool = False,
    checkpoint: Path | None = None,
    prepared_cache: Path | None = None,
    live_ollama: bool = False,
    translation_cache: Path | None = None,
    report_path: Path = Path("docs/bao_cao.md"),
) -> Path:
    """Execute genuine classical experiments plus requested optional stages, then report.

    Each experiment owns a single Generator and records its own provenance.
    Quick mode uses the committed source fixture and synthetic E04 transport;
    full mode never silently falls back to small data or a simulated chatbot.
    Transformer training is a separate explicit M3 command; --checkpoint reuses
    its verified weights, without retraining or treating a missing model as zero.
    """
    output_dir = e04.results_path(output_dir)
    if quick and (checkpoint or prepared_cache or live_ollama or translation_cache):
        raise ValueError(
            "Quick mode is offline; full checkpoint/cache/live options are incompatible"
        )
    config = load_config(config_path)
    suffix = "_fixture" if quick else ""
    output_dir.mkdir(parents=True, exist_ok=True)
    stages = []
    e01.run(
        config_path,
        output_dir / "tables" / f"e01_classical{suffix}.csv",
        fixture=quick,
        prepared_cache=prepared_cache,
    )
    stages.append({"stage": "E01_classical", "status": "completed"})
    stages.append({"stage": "E01_transformer_training", "status": "separate_optional_M3_command"})
    m5_evaluation.run(
        config_path,
        output_dir,
        fixture=quick,
        checkpoint=checkpoint,
        prepared_cache=prepared_cache,
        n_bootstrap=30 if quick else 2000,
    )
    stages.append({"stage": "E02_E03", "status": "completed"})
    if quick:
        generate_fixture(output_dir, config_path=config_path)
        stages.append({"stage": "E04", "status": "synthetic_fixture_not_live"})
    elif live_ollama:
        trial_path = e04.run(
            config_path,
            output_dir,
            prepared_cache=prepared_cache,
            translation_cache=translation_cache,
        )
        trials = json.loads(trial_path.read_text(encoding="utf-8"))["trials"]
        stages.append(
            {
                "stage": "E04",
                "status": "incomplete_backend_errors"
                if any(t["status"] == "error" for t in trials)
                else "completed_pending_review",
            }
        )
    else:
        stages.append({"stage": "E04", "status": "not_requested_local_live_optional"})
    m8_evaluation.run(
        config_path, output_dir, quick=quick, checkpoint=checkpoint, prepared_cache=prepared_cache
    )
    stages.append({"stage": "E05_E06", "status": "completed"})
    render_all(output_dir, quick=quick)
    # Rebuild historical scientific figures too, without recomputing their measurements.
    if quick:
        render_all(output_dir)
    manifest_path = output_dir / "tables" / ("run_all_quick.json" if quick else "run_all.json")
    artifact_paths = [
        p
        for d in ("tables", "figures")
        for p in sorted((output_dir / d).iterdir())
        if p.is_file() and p != manifest_path and p.name != "report_data.json"
    ]
    manifest = {
        "config": asdict(config),
        "seed": config.seed,
        "quick": quick,
        "fixture_only": quick,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], text=True)),
        "versions": {
            n: version(n) for n in ("numpy", "scikit-learn", "matplotlib", "threadpoolctl")
        },
        "python": platform.python_version(),
        "stages": stages,
        "randomness": "one config-seeded Generator per independently reproducible experiment",
        "artifact_sha256": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in artifact_paths
        },
        "source_sha256": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for root in ("experiments", "src/pidefense")
            for p in sorted(Path(root).rglob("*.py"))
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8")
    build_report(output_dir, report_path)
    print(f"Experiment stages recorded in {manifest_path}; report written to {report_path}")
    return manifest_path


def main() -> None:
    """Run quick offline checks or full experiments with explicitly selected optional backends."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--report", type=Path, default=Path("docs/bao_cao.md"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--prepared-cache", type=Path)
    parser.add_argument("--live-ollama", action="store_true")
    parser.add_argument("--translation-cache", type=Path)
    args = parser.parse_args()
    manifest_path = run(
        args.config,
        args.output_dir,
        quick=args.quick,
        checkpoint=args.checkpoint,
        prepared_cache=args.prepared_cache,
        live_ollama=args.live_ollama,
        translation_cache=args.translation_cache,
        report_path=args.report,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if any(stage["status"] == "incomplete_backend_errors" for stage in manifest["stages"]):
        parser.exit(
            1, "Experiments incomplete: local E04 backend errors; inspect the run manifest.\n"
        )


if __name__ == "__main__":
    main()
