"""Check documented defaults, safe YAML parsing and validation boundaries."""

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from pidefense.config import (
    AttackConfig,
    Config,
    ConfigError,
    DataConfig,
    MiddlewareConfig,
    TransformerModelConfig,
    load_config,
)


def test_default_yaml_matches_dataclasses() -> None:
    """Keep the specification's default YAML and dataclass defaults synchronized."""
    assert load_config() == Config()


@pytest.mark.parametrize("name", ["classical_tfidf", "transformer_distilbert"])
def test_model_configs_use_missing_defaults(name: str) -> None:
    """Model files may specify a subset while inheriting other dataclass fields."""
    assert load_config(Path("configs") / f"{name}.yaml") == Config()


def test_partial_config(tmp_path: Path) -> None:
    """Apply nested overrides without changing unrelated defaults or resolving paths."""
    path = tmp_path / "custom.yaml"
    path.write_text(
        "seed: 7\ndata:\n  val_fraction: 0\n  benign_corpus_path: inputs/benign.csv\n"
        "classical_model:\n  vectorizer: tfidf_char\n  classifier: svm\n"
        "transformer_model:\n  base_model: local-model\n  epochs: 1\n"
        "attack:\n  techniques: [base64]\n"
        "middleware:\n  guard_threshold: 1\n  ollama_model: local\n"
        "  ollama_host: http://127.0.0.1:11434/\n",
        encoding="utf-8",
    )
    config = load_config(path)
    assert config.seed == 7
    assert config.data == DataConfig(benign_corpus_path=Path("inputs/benign.csv"), val_fraction=0)
    assert config.classical_model.vectorizer == "tfidf_char"
    assert config.classical_model.classifier == "svm"
    assert config.transformer_model == TransformerModelConfig(base_model="local-model", epochs=1)
    assert config.attack.techniques == ("base64",)
    assert config.middleware.guard_threshold == 1
    assert config.middleware.ollama_model == "local"


@pytest.mark.parametrize("content", ["{}", "seed: 0\nattack:\n  techniques: []"])
def test_empty_mapping_and_zero_seed(tmp_path: Path, content: str) -> None:
    """An explicit mapping selects defaults; zero is a valid seed and empty attacks disable them."""
    path = tmp_path / "config.yaml"
    path.write_text(content, encoding="utf-8")
    config = load_config(path)
    if content == "{}":
        assert config == Config()
    else:
        assert config.seed == 0
        assert config.attack.techniques == ()


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("", "config must be a mapping"),
        ("null", "config must be a mapping"),
        ("[1, 2]", "config must be a mapping"),
        ("unknown: 1", "unknown field"),
        ("1: value", "unknown field"),
        ("seed: true", "seed"),
        ("seed: -1", "seed"),
        ("seed: 1.5", "seed"),
        ("seed: '1'", "seed"),
        ("data: null", "data must be a mapping"),
        ("data:\n  unknown: 1", "unknown field"),
        ("data:\n  dataset_name: ''", "dataset_name"),
        ("data:\n  benign_corpus_path: 1", "benign_corpus_path"),
        ("data:\n  test_fraction: 0", "fractions"),
        ("data:\n  test_fraction: 0.9\n  val_fraction: 0.1", "fractions"),
        ("data:\n  val_fraction: -0.1", "val_fraction"),
        ("data:\n  test_fraction: 2", "test_fraction"),
        ("data:\n  test_fraction: .nan", "test_fraction"),
        ("data:\n  test_fraction: .inf", "test_fraction"),
        ("data:\n  test_fraction: false", "test_fraction"),
        ("data:\n  test_fraction: '0.2'", "test_fraction"),
        ("classical_model:\n  classifier: other", "classifier"),
        ("classical_model:\n  vectorizer: other", "vectorizer"),
        ("transformer_model:\n  epochs: 0", "epochs"),
        ("transformer_model:\n  max_length: true", "max_length"),
        ("transformer_model:\n  batch_size: 1.5", "batch_size"),
        ("transformer_model:\n  base_model: null", "base_model"),
        ("attack:\n  techniques: base64", "YAML list"),
        ("attack:\n  techniques: [other]", "attack.techniques"),
        ("attack:\n  techniques: [base64, base64]", "duplicates"),
        ("middleware:\n  guard_threshold: 1.1", "guard_threshold"),
        ("middleware:\n  ollama_model: ''", "ollama_model"),
        ("[invalid", "valid YAML"),
        ("!!python/object/apply:os.getcwd []", "valid YAML"),
    ],
)
def test_invalid_yaml_config(tmp_path: Path, content: str, message: str) -> None:
    """Reject malformed or unsafe documents and give a field-specific error."""
    path = tmp_path / "invalid.yaml"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ConfigError, match=message):
        load_config(path)


@pytest.mark.parametrize(
    "host",
    [
        "https://example.com",
        "http://localhost.example.com",
        "file://localhost",
        "http://user:secret@localhost",
        "http://localhost/api",
        "http://localhost?x=y",
        "http://localhost#fragment",
        "http://localhost:0",
        "http://localhost:99999",
        "http://localhost:bad",
        "http://[::1",
        "localhost:11434",
        "",
    ],
)
def test_reject_nonlocal_or_malformed_host(host: str) -> None:
    """Configuration constrains future HTTP clients to loopback demo endpoints."""
    with pytest.raises(ConfigError, match="ollama_host"):
        MiddlewareConfig(ollama_host=host)


@pytest.mark.parametrize(
    "host", ["http://localhost", "https://127.0.0.1:443", "http://[::1]:11434"]
)
def test_loopback_hosts(host: str) -> None:
    """Support explicit loopback HTTP(S) endpoints including IPv6."""
    assert MiddlewareConfig(ollama_host=host).ollama_host == host


def test_missing_or_non_utf8_file(tmp_path: Path) -> None:
    """Wrap missing-file and decoding failures in the same public ConfigError type."""
    path = tmp_path / "missing.yaml"
    with pytest.raises(ConfigError, match="Cannot read"):
        load_config(path)
    path.write_bytes(b"\xff\xfe\x00")
    with pytest.raises(ConfigError, match="Cannot read"):
        load_config(path)


def test_direct_dataclass_validation_and_immutability() -> None:
    """Invalid direct dataclass construction cannot bypass numeric validation."""
    with pytest.raises(ConfigError, match="seed"):
        Config(seed=-1)
    with pytest.raises(ConfigError, match="epochs"):
        TransformerModelConfig(epochs=0)
    config = Config()
    with pytest.raises(FrozenInstanceError):
        config.seed = 1  # type: ignore[misc]
    assert isinstance(AttackConfig().techniques, tuple)
