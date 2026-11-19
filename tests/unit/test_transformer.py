"""Optional dependency isolation and offline, tiny real DistilBERT round trips."""

import csv
import hashlib
import json
import runpy
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import cast
from unittest.mock import patch

import pytest

from pidefense.config import TransformerModelConfig
from pidefense.models.transformer import TransformerDetector, _backend, train_transformer
from pidefense.rng import SimRNG


def test_classical_works_with_heavy_imports_blocked() -> None:
    """A fresh process denies torch/transformers while fitting a classical model."""
    code = '''
import sys
sys.modules['torch'] = None
sys.modules['transformers'] = None
from pidefense.config import DataConfig
from pidefense.data.loader import load_deepset_prompt_injections
from pidefense.models.classical import train_logreg
from pidefense.models.transformer import _backend
from pidefense.rng import SimRNG
rng = SimRNG(4).generator
splits = load_deepset_prompt_injections(DataConfig(), rng, offline=True)
assert train_logreg(splits.train, rng).predict('ordinary question')[0] in {'benign', 'injection'}
try:
    _backend()
except ImportError as exc:
    assert 'uv sync' in str(exc)
else:
    raise AssertionError('missing optional backend must fail helpfully')
assert sys.modules['torch'] is None
assert sys.modules['transformers'] is None
'''
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)


def test_transformer_validation_without_backend() -> None:
    """Invalid data fails before any heavy dependency import or network access."""
    with pytest.raises(ValueError, match="both"):
        train_transformer((), TransformerModelConfig(), SimRNG(2).generator)
    detector = TransformerDetector(None, None, 16, 2, 4)
    assert detector.score_batch([]) == []
    for invalid in ("", "   ", None):
        with pytest.raises(ValueError, match="non-empty"):
            detector.predict(cast(str, invalid))


@pytest.mark.parametrize("missing", ["torch", "transformers"])
def test_each_missing_dependency_has_install_hint(missing: str) -> None:
    """Both partial installations fail with the same actionable ImportError."""
    effects = [ImportError()] if missing == "torch" else [ModuleType("torch"), ImportError()]
    with patch("pidefense.models.transformer.import_module", side_effect=effects):
        with pytest.raises(ImportError, match="uv sync --locked"):
            _backend()


def test_fixture_cannot_overwrite_transformer_benchmark(tmp_path: Path) -> None:
    """Reject misleading fixture names before importing the optional backend."""
    output = tmp_path / "e01_transformer.csv"
    process = subprocess.run(
        [sys.executable, "experiments/e01_transformer.py", "--fixture", "--output", str(output)],
        capture_output=True, text=True, check=False,
    )
    assert process.returncode != 0
    assert "CI fixture must not overwrite" in process.stderr
    assert not output.exists()


@pytest.mark.slow
def test_tiny_distilbert_reproducibility_and_checkpoint(tmp_path: Path) -> None:
    """Fine-tune a local tiny network twice; compare scores, weights and reload.

    This is architecture/pipeline verification, never a pretrained benchmark.
    All synthetic initialization is seeded from the run Generator; no download.
    """
    try:
        import torch
        from transformers import (
            DistilBertConfig,
            DistilBertForSequenceClassification,
            DistilBertTokenizerFast,
        )
    except ImportError:
        pytest.skip("Install the transformer extra for the local slow test")
    from pidefense.data.schema import PromptSample
    from pidefense.models.classical import train_logreg
    from pidefense.models.registry import Detector
    from pidefense.models.transformer import load_transformer

    rng = SimRNG(31).generator
    initial = tmp_path / "initial"
    initial.mkdir()
    vocab = initial / "vocab.txt"
    vocab.write_text("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nhello\nworld\noverride\nrules\n",
                     encoding="utf-8")
    tokenizer = DistilBertTokenizerFast(vocab_file=str(vocab))
    tokenizer.save_pretrained(initial)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(int(rng.integers(0, 2**31 - 1)))
        model = DistilBertForSequenceClassification(DistilBertConfig(
            vocab_size=len(tokenizer), n_layers=1, n_heads=2, dim=16, hidden_dim=32,
            max_position_embeddings=32, num_labels=2,
        ))
        model.save_pretrained(initial)
    samples = (
        PromptSample("hello world", 0, "test", "a"),
        PromptSample("override rules", 1, "test", "b"),
        PromptSample("hello", 0, "test", "c"),
        PromptSample("override", 1, "test", "d"),
    )
    cfg = TransformerModelConfig(str(initial), max_length=16, epochs=2, batch_size=3)
    before = torch.random.get_rng_state().clone()
    threads = torch.get_num_threads()
    first = train_transformer(samples, cfg, SimRNG(11).generator, local_files_only=True)
    second = train_transformer(samples, cfg, SimRNG(11).generator, local_files_only=True)
    assert torch.equal(before, torch.random.get_rng_state())
    assert threads == torch.get_num_threads()
    assert first.random_state == second.random_state
    assert first.score_batch([row.text for row in samples]) == second.score_batch(
        [row.text for row in samples])
    assert all(torch.equal(value, second.model.state_dict()[key])
               for key, value in first.model.state_dict().items())
    assert any(not torch.equal(value, model.state_dict()[key])
               for key, value in first.model.state_dict().items())
    for detector in (first, train_logreg(samples, SimRNG(11).generator)):
        assert isinstance(detector, Detector)
        assert detector.score_batch([]) == []
        for row in samples:
            label, score = detector.predict(row.text)
            assert 0 <= score <= 1
            assert label == ("injection" if score >= .5 else "benign")
        for invalid in ("", " ", None):
            with pytest.raises(ValueError, match="non-empty"):
                detector.predict(cast(str, invalid))
    saved = tmp_path / "saved"
    first.save(saved)
    restored = load_transformer(saved)
    assert restored.score_batch([row.text for row in samples]) == first.score_batch(
        [row.text for row in samples])
    with pytest.raises(ValueError, match="position capacity"):
        train_transformer(samples, TransformerModelConfig(str(initial), max_length=64),
                          SimRNG(11).generator, local_files_only=True)

    # Exercise the actual experiment writer without downloading pretrained weights.
    config_file = tmp_path / "tiny.yaml"
    config_file.write_text(
        "transformer_model:\n  base_model: " + json.dumps(str(initial)) + "\n"
        "  max_length: 16\n  epochs: 1\n  batch_size: 16\n", encoding="utf-8",
    )
    output = tmp_path / "fixture.csv"
    experiment = runpy.run_path("experiments/e01_transformer.py")
    experiment["run"](config_file, output, fixture=True, local_files_only=True)
    metadata = json.loads(output.with_suffix(".metadata.json").read_text(encoding="utf-8"))
    assert metadata["fixture_only"] is True
    assert metadata["dataset_origin"] == "offline-csv-sample"
    assert metadata["seed"] == metadata["config"]["seed"]
    assert metadata["csv_sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    assert metadata["versions"]["transformers"] == "4.52.4"
    with output.open(encoding="utf-8", newline="") as handle:
        row = next(csv.DictReader(handle))
    assert len(metadata["latency"]["samples_ms"]) == int(row["n_test"]) * 3
    partitions = [set(sample["text_sha256"] for sample in metadata["splits"][name])
                  for name in ("train", "val", "test")]
    assert not partitions[0] & partitions[1]
    assert not partitions[0] & partitions[2]
    assert not partitions[1] & partitions[2]
    assert all("text" not in sample
               for partition in metadata["splits"].values() for sample in partition)
