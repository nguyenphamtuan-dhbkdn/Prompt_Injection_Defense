"""One explicitly seeded NumPy PCG64 stream per experiment run."""

from dataclasses import dataclass, field

import numpy as np

from pidefense.config import Config, validate_seed


@dataclass(frozen=True, eq=False)
class SimRNG:
    """Own one Generator(PCG64(seed)); pass its generator to stochastic steps.

    Reproducibility requires the same seed, NumPy version and call order. Accessing
    ``generator`` never reseeds or constructs a second stream. The frozen wrapper
    prevents replacing the seed or generator while the generator state advances.
    """

    seed: int
    generator: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate the configured seed and construct exactly one PCG64 generator."""
        validate_seed(self.seed)
        object.__setattr__(self, "generator", np.random.Generator(np.random.PCG64(self.seed)))

    @classmethod
    def from_config(cls, config: Config) -> "SimRNG":
        """Start a run's sole random stream using its validated configuration seed."""
        return cls(config.seed)
