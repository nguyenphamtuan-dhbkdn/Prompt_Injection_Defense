"""Stratification using only the run's existing NumPy Generator."""

from collections.abc import Sequence
from dataclasses import dataclass
from math import floor

from numpy.random import Generator

from pidefense.config import DataConfig
from pidefense.data.schema import Label, PromptSample, text_identity

LABELS: tuple[Label, ...] = (0, 1)


@dataclass(frozen=True)
class LabelBalance:
    """Counts and proportions n_label / n_split; empty splits have zero fractions."""

    total: int
    counts: dict[Label, int]
    fractions: dict[Label, float]


@dataclass(frozen=True)
class DatasetSplits:
    """Disjoint partitions, with the data origin explicitly attached."""

    train: tuple[PromptSample, ...]
    val: tuple[PromptSample, ...]
    test: tuple[PromptSample, ...]
    origin: str

    def label_balance(self) -> dict[str, LabelBalance]:
        """Report both labels, including zero counts, without logging raw text."""
        report = {}
        for name, samples in (("train", self.train), ("val", self.val), ("test", self.test)):
            counts: dict[Label, int] = {
                label: sum(row.label == label for row in samples) for label in LABELS
            }
            total = len(samples)
            fractions = {label: count / total if total else 0.0 for label, count in counts.items()}
            report[name] = LabelBalance(total, counts, fractions)
        return report


def stratified_split(
    samples: Sequence[PromptSample], cfg: DataConfig, rng: Generator, *, origin: str = "local"
) -> DatasetSplits:
    """Deduplicate, shuffle and apportion each class by largest remainders.

    Fractions refer to the unique corpus. Each class allocation differs from its
    ideal n_class * fraction by less than one row. Ties use train/val/test order.
    Conflicting duplicate labels or missing classes in requested splits fail.
    The first occurrence retains provenance; no additional RNG is created.
    """
    unique: dict[str, PromptSample] = {}
    ids: set[str] = set()
    for sample in samples:
        key = text_identity(sample.text)
        if key in unique:
            if unique[key].label != sample.label:
                raise ValueError("Duplicate text has conflicting labels")
            continue
        if sample.source_id in ids:
            raise ValueError("source_id must identify a unique text")
        unique[key] = sample
        ids.add(sample.source_id)
    fractions = (1 - cfg.val_fraction - cfg.test_fraction, cfg.val_fraction, cfg.test_fraction)
    partitions: list[list[PromptSample]] = [[], [], []]
    for label in (0, 1):
        group = [row for row in unique.values() if row.label == label]
        ideals = [len(group) * fraction for fraction in fractions]
        counts = [floor(ideal) for ideal in ideals]
        order = sorted(range(3), key=lambda i: -(ideals[i] - counts[i]))
        for i in order[: len(group) - sum(counts)]:
            counts[i] += 1
        if any(
            fraction > 0 and count == 0 for fraction, count in zip(fractions, counts, strict=True)
        ):
            raise ValueError("Insufficient unique samples per label for requested split fractions")
        shuffled = [group[int(i)] for i in rng.permutation(len(group))]
        start = 0
        for partition, count in zip(partitions, counts, strict=True):
            partition.extend(shuffled[start : start + count])
            start += count
    train, val, test = (
        tuple(partition[int(i)] for i in rng.permutation(len(partition)))
        for partition in partitions
    )
    return DatasetSplits(train, val, test, origin)
