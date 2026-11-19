"""M2 sparse features, shared Detector contract, metrics and leakage checks."""

from collections.abc import Callable
from math import log, sqrt
from typing import cast

import numpy as np
import pytest
from numpy.random import Generator
from scipy.sparse import issparse

from pidefense.config import DataConfig, Vectorizer
from pidefense.data.loader import load_deepset_prompt_injections
from pidefense.data.schema import PromptSample
from pidefense.eval.metrics import compute_classification_metrics
from pidefense.features.tfidf import build_vectorizer
from pidefense.models.classical import (
    ClassicalDetector,
    train_logreg,
    train_random_forest,
    train_svm,
)
from pidefense.models.registry import Detector
from pidefense.rng import SimRNG

TRAINERS = (train_logreg, train_svm, train_random_forest)
Trainer = Callable[[tuple[PromptSample, ...], Generator], ClassicalDetector]


@pytest.mark.parametrize("kind", ["tfidf_word", "tfidf_char", "tfidf_combo"])
def test_sparse_features_and_training_vocabulary(kind: Vectorizer) -> None:
    """Held-out tokens must never enter the fitted word/char vocabulary."""
    features = build_vectorizer(kind)
    matrix = features.fit_transform(["apple apple pear", "pear orange"])
    assert issparse(matrix)
    before = features.get_feature_names_out().copy()
    features.transform(["zzzzheldouttokenzzzz"])
    assert np.array_equal(before, features.get_feature_names_out())
    names = set(before)
    if kind == "tfidf_combo":
        assert "word__apple pear" in names
        assert "char__apple" in names
        assert "char__app" in names
    elif kind == "tfidf_word":
        assert "apple pear" in names
        assert "app" not in names
    else:
        assert "app" in names and "apple" in names


def test_tfidf_formula() -> None:
    """Check smooth IDF and per-row L2 normalization against hand arithmetic."""
    vectorizer = build_vectorizer("tfidf_word")
    matrix = vectorizer.fit_transform(["apple apple", "pear"])
    apple = vectorizer.vocabulary_["apple"]
    apple_pair = vectorizer.vocabulary_["apple apple"]
    assert vectorizer.idf_[apple] == pytest.approx(log(3 / 2) + 1)
    assert matrix[0, apple] == pytest.approx(2 / sqrt(5))
    assert matrix[0, apple_pair] == pytest.approx(1 / sqrt(5))
    with pytest.raises(ValueError, match="Unsupported"):
        build_vectorizer(cast(Vectorizer, "unknown"))


@pytest.mark.parametrize("trainer", TRAINERS)
def test_shared_detector_contract_and_reproducibility(trainer: Trainer) -> None:
    """Every classifier returns injection-oriented scores with reproducible fitting."""
    def fit() -> ClassicalDetector:
        rng = SimRNG(17).generator
        splits = load_deepset_prompt_injections(DataConfig(), rng, offline=True)
        return trainer(splits.train, rng)

    first, second = fit(), fit()
    assert isinstance(first, Detector)
    assert first.random_state == second.random_state
    texts = ["ordinary astronomy question", "unseen input"]
    assert first.score_batch(texts) == second.score_batch(texts)
    assert first.score_batch([]) == []
    for text in texts:
        label, score = first.predict(text)
        assert 0 <= score <= 1
        assert label == ("injection" if score >= 0.5 else "benign")
        if first.classifier == "svm":
            margin = float(first.pipeline.decision_function([text])[0])
            assert score == pytest.approx(1 / (1 + np.exp(-margin)))
        else:
            assert score == pytest.approx(first.pipeline.predict_proba([text])[0, 1])
    for invalid in ("", "   ", None):
        with pytest.raises(ValueError, match="non-empty"):
            first.predict(invalid)  # type: ignore[arg-type]
    # A transform/prediction cannot grow the train-only vocabulary.
    names = first.pipeline.named_steps["tfidf"].get_feature_names_out()
    first.predict("zzzzuniquetesttokenzzzz")
    assert np.array_equal(names, first.pipeline.named_steps["tfidf"].get_feature_names_out())
    with pytest.raises(ValueError, match="both"):
        trainer((), SimRNG(17).generator)


def test_metrics_hand_computed() -> None:
    """TP=2, FP=1, FN=1; positives win six of nine positive-negative pairs."""
    report = compute_classification_metrics([1, 0, 1, 0, 1, 0], [.9, .8, .7, .3, .2, .1])
    assert report.precision == pytest.approx(2 / 3)
    assert report.recall == pytest.approx(2 / 3)
    assert report.f1 == pytest.approx(2 / 3)
    assert report.roc_auc == pytest.approx(6 / 9)
    zero = compute_classification_metrics([0, 1], [.1, .1])
    assert (zero.precision, zero.recall, zero.f1, zero.roc_auc) == (0, 0, 0, .5)
    assert compute_classification_metrics([0, 1], [.2, .5]).recall == 1


@pytest.mark.parametrize("truth,scores,threshold", [
    ([], [], .5), ([0, 1], [.2], .5), ([0, 2], [.2, .7], .5),
    ([0, 1], [float("nan"), .7], .5), ([0, 1], [-.1, .7], .5),
    ([0, 1], [.2, 1.1], .5), ([0, 1], [.2, .7], float("inf")),
    ([0, 1], [.2, .7], -1),
])
def test_invalid_metrics(truth: list[int], scores: list[float], threshold: float) -> None:
    """Reject undefined AUC, mismatched arrays and nonfinite/out-of-range scores."""
    with pytest.raises(ValueError):
        compute_classification_metrics(truth, scores, threshold)
