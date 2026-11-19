"""Frozen configuration dataclasses and safe, field-validated YAML loading."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from math import isfinite
from pathlib import Path
from typing import Literal, TypeVar, cast
from urllib.parse import urlsplit

import yaml

Vectorizer = Literal["tfidf_word", "tfidf_char", "tfidf_combo"]
Classifier = Literal["logreg", "svm", "rf"]
Technique = Literal["base64", "leetspeak", "zero_width", "translate", "indirect", "multiturn"]
TECHNIQUES: tuple[Technique, ...] = (
    "base64",
    "leetspeak",
    "zero_width",
    "translate",
    "indirect",
    "multiturn",
)
_T = TypeVar("_T", bound=str)


class ConfigError(ValueError):
    """A YAML document or configuration field violates the declared schema."""


def validate_seed(seed: int) -> None:
    """Require a nonnegative integer seed, excluding Python's boolean subtype."""
    if type(seed) is not int or seed < 0:
        raise ConfigError("seed must be a nonnegative integer")


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{name} must be a non-empty string")
    return value


def _integer(value: object, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ConfigError(f"{name} must be a positive integer")
    return value


def _fraction(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, float | int):
        raise ConfigError(f"{name} must be a finite number in [0, 1]")
    try:
        result = float(value)
    except OverflowError as exc:
        raise ConfigError(f"{name} must be a finite number in [0, 1]") from exc
    if not isfinite(result) or not 0 <= result <= 1:
        raise ConfigError(f"{name} must be a finite number in [0, 1]")
    return result


def _choice(value: object, name: str, choices: tuple[_T, ...]) -> _T:
    if not isinstance(value, str) or value not in choices:
        raise ConfigError(f"{name} must be one of {', '.join(choices)}")
    return cast(_T, value)


@dataclass(frozen=True)
class DataConfig:
    """Split fractions refer to the full corpus; train = 1 - test - val."""

    dataset_name: str = "deepset/prompt-injections"
    benign_corpus_path: Path = Path("data/benign_corpus/benign_queries.csv")
    test_fraction: float = 0.2
    val_fraction: float = 0.1

    def __post_init__(self) -> None:
        """Validate the dataset identity, path type and nonempty train/test splits."""
        _text(self.dataset_name, "data.dataset_name")
        if not isinstance(self.benign_corpus_path, Path):
            raise ConfigError("data.benign_corpus_path must be a pathlib.Path")
        test = _fraction(self.test_fraction, "data.test_fraction")
        val = _fraction(self.val_fraction, "data.val_fraction")
        if test == 0 or test + val >= 1:
            raise ConfigError("data fractions require test > 0 and test + val < 1")


@dataclass(frozen=True)
class ClassicalModelConfig:
    """Select a TF-IDF representation and classical classifier; no training in M0."""

    vectorizer: Vectorizer = "tfidf_combo"
    classifier: Classifier = "logreg"

    def __post_init__(self) -> None:
        """Reject names outside the finite model and feature registries."""
        _choice(
            self.vectorizer,
            "classical_model.vectorizer",
            ("tfidf_word", "tfidf_char", "tfidf_combo"),
        )
        _choice(self.classifier, "classical_model.classifier", ("logreg", "svm", "rf"))


@dataclass(frozen=True)
class TransformerModelConfig:
    """Store optional transformer settings without importing heavy dependencies."""

    base_model: str = "distilbert-base-uncased"
    max_length: int = 128
    epochs: int = 3
    batch_size: int = 16

    def __post_init__(self) -> None:
        """Require a model name and positive integer training dimensions."""
        _text(self.base_model, "transformer_model.base_model")
        for name in ("max_length", "epochs", "batch_size"):
            _integer(getattr(self, name), f"transformer_model.{name}")


@dataclass(frozen=True)
class AttackConfig:
    """Immutable technique selection for future authorized local evaluation only."""

    techniques: tuple[Technique, ...] = TECHNIQUES

    def __post_init__(self) -> None:
        """Require a tuple of supported, unique technique identifiers."""
        if not isinstance(self.techniques, tuple):
            raise ConfigError("attack.techniques must be a tuple")
        for technique in self.techniques:
            _choice(technique, "attack.techniques", TECHNIQUES)
        if len(set(self.techniques)) != len(self.techniques):
            raise ConfigError("attack.techniques must not contain duplicates")


@dataclass(frozen=True)
class MiddlewareConfig:
    """Store guard settings; Ollama destinations are restricted to loopback hosts."""

    guard_threshold: float = 0.5
    ollama_model: str = "phi3:mini"
    ollama_host: str = "http://localhost:11434"

    def __post_init__(self) -> None:
        """Validate a probability threshold and an HTTP(S) loopback endpoint."""
        _fraction(self.guard_threshold, "middleware.guard_threshold")
        _text(self.ollama_model, "middleware.ollama_model")
        host = _text(self.ollama_host, "middleware.ollama_host")
        try:
            url = urlsplit(host)
            port = url.port
            valid = (
                url.scheme in ("http", "https")
                and url.hostname in ("localhost", "127.0.0.1", "::1")
                and url.username is None
                and url.password is None
                and url.path in ("", "/")
                and not url.query
                and not url.fragment
                and (port is None or port > 0)
            )
        except ValueError as exc:
            raise ConfigError(
                "middleware.ollama_host must be a valid loopback HTTP(S) URL"
            ) from exc
        if not valid:
            raise ConfigError("middleware.ollama_host must be a loopback HTTP(S) URL")


@dataclass(frozen=True)
class Config:
    """Application configuration with the defaults from SPEC.md Section 7."""

    seed: int = 20260119
    data: DataConfig = field(default_factory=DataConfig)
    classical_model: ClassicalModelConfig = field(default_factory=ClassicalModelConfig)
    transformer_model: TransformerModelConfig = field(default_factory=TransformerModelConfig)
    attack: AttackConfig = field(default_factory=AttackConfig)
    middleware: MiddlewareConfig = field(default_factory=MiddlewareConfig)

    def __post_init__(self) -> None:
        """Validate the root seed; nested dataclasses validate their own fields."""
        validate_seed(self.seed)


def _mapping(value: object, name: str, keys: set[str]) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{name} must be a mapping")
    result: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str) or key not in keys:
            raise ConfigError(f"{name} contains an unknown field: {key!r}")
        result[key] = item
    return result


def load_config(path: str | Path = "configs/default.yaml") -> Config:
    """Load safe YAML, reject unknown fields, and fill missing fields with defaults.

    Paths remain relative to the working directory; no filesystem data is loaded.
    Empty documents are invalid, while an explicit empty mapping selects defaults.
    """
    try:
        content: object = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ConfigError(f"Cannot read a valid YAML configuration from {path}") from exc
    root = _mapping(content, "config", set(Config.__dataclass_fields__))
    seed = root.get("seed", 20260119)
    if type(seed) is not int:
        raise ConfigError("seed must be a nonnegative integer")

    data = _mapping(root.get("data", {}), "data", set(DataConfig.__dataclass_fields__))
    classical = _mapping(
        root.get("classical_model", {}),
        "classical_model",
        set(ClassicalModelConfig.__dataclass_fields__),
    )
    transformer = _mapping(
        root.get("transformer_model", {}),
        "transformer_model",
        set(TransformerModelConfig.__dataclass_fields__),
    )
    attack = _mapping(root.get("attack", {}), "attack", {"techniques"})
    middleware = _mapping(
        root.get("middleware", {}), "middleware", set(MiddlewareConfig.__dataclass_fields__)
    )
    techniques = attack.get("techniques", list(TECHNIQUES))
    if not isinstance(techniques, list):
        raise ConfigError("attack.techniques must be a YAML list")
    return Config(
        seed=seed,
        data=DataConfig(
            dataset_name=_text(
                data.get("dataset_name", "deepset/prompt-injections"), "data.dataset_name"
            ),
            benign_corpus_path=Path(
                _text(
                    data.get("benign_corpus_path", "data/benign_corpus/benign_queries.csv"),
                    "data.benign_corpus_path",
                )
            ),
            test_fraction=_fraction(data.get("test_fraction", 0.2), "data.test_fraction"),
            val_fraction=_fraction(data.get("val_fraction", 0.1), "data.val_fraction"),
        ),
        classical_model=ClassicalModelConfig(
            vectorizer=_choice(
                classical.get("vectorizer", "tfidf_combo"),
                "classical_model.vectorizer",
                ("tfidf_word", "tfidf_char", "tfidf_combo"),
            ),
            classifier=_choice(
                classical.get("classifier", "logreg"),
                "classical_model.classifier",
                ("logreg", "svm", "rf"),
            ),
        ),
        transformer_model=TransformerModelConfig(
            base_model=_text(
                transformer.get("base_model", "distilbert-base-uncased"),
                "transformer_model.base_model",
            ),
            max_length=_integer(transformer.get("max_length", 128), "transformer_model.max_length"),
            epochs=_integer(transformer.get("epochs", 3), "transformer_model.epochs"),
            batch_size=_integer(transformer.get("batch_size", 16), "transformer_model.batch_size"),
        ),
        attack=AttackConfig(
            tuple(_choice(item, "attack.techniques", TECHNIQUES) for item in techniques)
        ),
        middleware=MiddlewareConfig(
            guard_threshold=_fraction(
                middleware.get("guard_threshold", 0.5), "middleware.guard_threshold"
            ),
            ollama_model=_text(
                middleware.get("ollama_model", "phi3:mini"), "middleware.ollama_model"
            ),
            ollama_host=_text(
                middleware.get("ollama_host", "http://localhost:11434"), "middleware.ollama_host"
            ),
        ),
    )
