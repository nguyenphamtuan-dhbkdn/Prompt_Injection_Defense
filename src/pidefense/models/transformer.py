"""Optional CPU DistilBERT fine-tuning with a lazy, ImportError-guarded backend."""

import json
from collections.abc import Sequence
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
from numpy.random import Generator

from pidefense.config import TransformerModelConfig
from pidefense.data.schema import PromptSample
from pidefense.models.registry import Prediction

DISTILBERT_REVISION = "12040accade4e8a0f71eabdb258fecc2e7e948be"


def _backend() -> tuple[ModuleType, ModuleType]:
    """Import heavy libraries only on demand; classical imports stay independent."""
    try:
        torch = import_module("torch")
        transformers = import_module("transformers")
    except ImportError:
        raise ImportError(
            "DistilBERT requires optional dependencies: "
            "uv sync --locked --extra dev --extra transformer"
        ) from None
    return torch, transformers


@dataclass
class TransformerDetector:
    """Return softmax(logits)[injection] using CPU inference with dropout disabled."""

    model: Any
    tokenizer: Any
    max_length: int
    batch_size: int
    random_state: int

    def score_batch(self, texts: Sequence[str]) -> list[float]:
        """Truncate to max_length, dynamically pad each batch, and score class one."""
        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("Detector input must be non-empty text")
        if not texts:
            return []
        torch, _ = _backend()
        self.model.eval()
        scores: list[float] = []
        with torch.inference_mode():
            for start in range(0, len(texts), self.batch_size):
                encoded = self.tokenizer(
                    list(texts[start:start + self.batch_size]), padding=True,
                    truncation=True, max_length=self.max_length, return_tensors="pt",
                )
                logits = self.model(**encoded, return_dict=True).logits
                values = torch.softmax(logits, dim=-1)[:, 1].tolist()
                scores.extend(float(value) for value in values)
        return scores

    def predict(self, text: str) -> Prediction:
        """Return injection for score >= 0.5, matching the classical Detector."""
        score = self.score_batch([text])[0]
        return ("injection" if score >= 0.5 else "benign", score)

    def save(self, directory: Path) -> None:
        """Save tokenizer and safe model weights using Hugging Face's native format."""
        self.model.save_pretrained(str(directory), safe_serialization=True)
        self.tokenizer.save_pretrained(str(directory))
        (directory / "detector.json").write_text(json.dumps({
            "max_length": self.max_length, "batch_size": self.batch_size,
            "random_state": self.random_state,
        }, indent=2) + "\n", encoding="utf-8")


def train_transformer(
    samples: Sequence[PromptSample], cfg: TransformerModelConfig, rng: Generator,
    *, revision: str = DISTILBERT_REVISION, local_files_only: bool = False,
) -> TransformerDetector:
    """Fine-tune all DistilBERT weights with AdamW and binary cross-entropy.

    A torch seed is drawn from the sole NumPy Generator, then used in an isolated
    CPU RNG context for initialization and dropout. Epoch permutations use that
    same NumPy stream. CPU deterministic algorithms and two threads bound resource
    use; original RNG, thread count and deterministic settings are restored.
    Learning rate is fixed at 5e-5, weight decay at 0.01, with norm clipping at 1.
    No validation/test samples, scheduler, early stopping or checkpoint selection.
    """
    if {sample.label for sample in samples} != {0, 1}:
        raise ValueError("Training requires both benign and injection samples")
    torch, transformers = _backend()
    random_state = int(rng.integers(0, np.iinfo(np.int32).max))
    threads = torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(random_state)
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                cfg.base_model, revision=revision, local_files_only=local_files_only,
                cache_dir="data/raw/transformer_cache",
            )
            model = transformers.AutoModelForSequenceClassification.from_pretrained(
                cfg.base_model, revision=revision, local_files_only=local_files_only,
                cache_dir="data/raw/transformer_cache", num_labels=2,
                id2label={0: "benign", 1: "injection"},
                label2id={"benign": 0, "injection": 1},
                attn_implementation="eager",
            ).to("cpu")
            if model.config.model_type != "distilbert":
                raise ValueError("M3 supports only DistilBERT checkpoints")
            if cfg.max_length > model.config.max_position_embeddings:
                raise ValueError("max_length exceeds model position capacity")
            optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
            model.train()
            for _ in range(cfg.epochs):
                order = rng.permutation(len(samples))
                for start in range(0, len(samples), cfg.batch_size):
                    batch = [samples[int(index)] for index in order[start:start + cfg.batch_size]]
                    encoded = tokenizer(
                        [sample.text for sample in batch], padding=True, truncation=True,
                        max_length=cfg.max_length, return_tensors="pt",
                    )
                    labels = torch.tensor([sample.label for sample in batch], dtype=torch.long)
                    optimizer.zero_grad(set_to_none=True)
                    loss = model(**encoded, labels=labels, return_dict=True).loss
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
            model.eval()
    finally:
        torch.set_num_threads(threads)
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
    return TransformerDetector(model, tokenizer, cfg.max_length, cfg.batch_size, random_state)


def load_transformer(directory: Path) -> TransformerDetector:
    """Load an M3 checkpoint locally with its persisted inference settings."""
    _, transformers = _backend()
    settings = json.loads((directory / "detector.json").read_text(encoding="utf-8"))
    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        str(directory), local_files_only=True, attn_implementation="eager",
    ).to("cpu")
    tokenizer = transformers.AutoTokenizer.from_pretrained(str(directory), local_files_only=True)
    return TransformerDetector(model, tokenizer, **settings)
