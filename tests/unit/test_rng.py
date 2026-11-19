"""Check PCG64 stream identity, repeatability, progression and invalid seeds."""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from pidefense.config import Config, ConfigError, load_config
from pidefense.rng import SimRNG


def test_same_config_replays_entire_stream() -> None:
    """Equal seeds and call order reproduce sampling and permutation exactly."""
    config = load_config()
    first = SimRNG.from_config(config)
    replay = SimRNG.from_config(config)
    assert first.seed == config.seed
    assert isinstance(first.generator.bit_generator, np.random.PCG64)
    np.testing.assert_array_equal(first.generator.random(20), replay.generator.random(20))
    np.testing.assert_array_equal(first.generator.permutation(20), replay.generator.permutation(20))
    np.testing.assert_array_equal(
        first.generator.choice(20, size=10, replace=True),
        replay.generator.choice(20, size=10, replace=True),
    )


def test_access_preserves_stream_and_advances_state() -> None:
    """Repeated access returns the same object and consumes successive PCG64 values."""
    rng = SimRNG(42)
    reference = SimRNG.from_config(Config(seed=42))
    generator = rng.generator
    first = generator.random(10)
    assert generator is rng.generator
    second = rng.generator.random(10)
    np.testing.assert_array_equal(np.concatenate([first, second]), reference.generator.random(20))
    assert not np.array_equal(first, second)


def test_different_seeds_and_immutable_wrapper() -> None:
    """Distinct fixed seeds produce distinct streams; wrapper fields cannot be replaced."""
    rng = SimRNG(0)
    assert not np.array_equal(rng.generator.random(20), SimRNG(1).generator.random(20))
    with pytest.raises(FrozenInstanceError):
        rng.seed = 1  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        rng.generator = SimRNG(1).generator  # type: ignore[misc]


@pytest.mark.parametrize("seed", [-1, True, 1.5, "42", None])
def test_invalid_seeds(seed: object) -> None:
    """Reject invalid runtime inputs even when callers bypass static type checks."""
    with pytest.raises(ConfigError, match="seed"):
        SimRNG(seed)  # type: ignore[arg-type]
