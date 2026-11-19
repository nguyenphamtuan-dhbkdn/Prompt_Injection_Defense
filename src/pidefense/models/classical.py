"""Train sparse classical detectors using seeds drawn from the run's Generator."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.random import Generator
from scipy.special import expit  # type: ignore[import-untyped]
from sklearn.ensemble import RandomForestClassifier  # type: ignore[import-untyped]
from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]
from sklearn.pipeline import Pipeline  # type: ignore[import-untyped]
from sklearn.svm import LinearSVC  # type: ignore[import-untyped]

from pidefense.config import ClassicalModelConfig, Classifier, Vectorizer
from pidefense.data.schema import PromptSample
from pidefense.features.tfidf import build_vectorizer
from pidefense.models.registry import Prediction


@dataclass
class ClassicalDetector:
    """Own fitted features and estimator; SVM scores are uncalibrated sigmoid margins."""

    pipeline: Pipeline
    classifier: Classifier
    random_state: int

    def score_batch(self, texts: Sequence[str]) -> list[float]:
        """Return P(injection), or sigmoid(decision_function) for Linear SVM.

        Sigmoid preserves the SVM margin ranking for ROC-AUC and maps margin
        zero to the fixed decision threshold 0.5. It is not calibration.
        """
        if not texts:
            return []
        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("Detector input must be non-empty text")
        if self.classifier == "svm":
            values = expit(self.pipeline.decision_function(list(texts)))
        else:
            # Training requires both classes, sorted by sklearn as [0, 1].
            values = self.pipeline.predict_proba(list(texts))[:, 1]
        return [float(value) for value in values]

    def predict(self, text: str) -> Prediction:
        """Return injection when its score >= 0.5, otherwise benign."""
        score = self.score_batch([text])[0]
        return ("injection" if score >= 0.5 else "benign", score)


def train_classical(
    samples: Sequence[PromptSample], cfg: ClassicalModelConfig, rng: Generator
) -> ClassicalDetector:
    """Fit TF-IDF and a binary estimator on training samples only.

    sklearn 1.6.1 accepts integer random_state, not numpy Generator. Draw its
    seed from the sole run stream; never use an unseeded sklearn estimator.
    """
    if {sample.label for sample in samples} != {0, 1}:
        raise ValueError("Training requires both benign and injection samples")
    random_state = int(rng.integers(0, np.iinfo(np.int32).max))
    estimator: LogisticRegression | LinearSVC | RandomForestClassifier
    if cfg.classifier == "logreg":
        estimator = LogisticRegression(max_iter=1000, random_state=random_state)
    elif cfg.classifier == "svm":
        estimator = LinearSVC(max_iter=10000, random_state=random_state)
    else:
        estimator = RandomForestClassifier(
            n_estimators=100, n_jobs=1, random_state=random_state
        )
    pipeline = Pipeline([("tfidf", build_vectorizer(cfg.vectorizer)), ("classifier", estimator)])
    pipeline.fit([sample.text for sample in samples], [sample.label for sample in samples])
    return ClassicalDetector(pipeline, cfg.classifier, random_state)


def train_logreg(
    samples: Sequence[PromptSample], rng: Generator, vectorizer: Vectorizer = "tfidf_combo"
) -> ClassicalDetector:
    """Fit L2-regularized logistic regression with default C=1 using LBFGS."""
    return train_classical(samples, ClassicalModelConfig(vectorizer, "logreg"), rng)


def train_svm(
    samples: Sequence[PromptSample], rng: Generator, vectorizer: Vectorizer = "tfidf_combo"
) -> ClassicalDetector:
    """Fit a linear squared-hinge SVM with default C=1 and dual='auto'."""
    return train_classical(samples, ClassicalModelConfig(vectorizer, "svm"), rng)


def train_random_forest(
    samples: Sequence[PromptSample], rng: Generator, vectorizer: Vectorizer = "tfidf_combo"
) -> ClassicalDetector:
    """Fit bootstrap Gini trees with sqrt feature sampling and no depth limit."""
    return train_classical(samples, ClassicalModelConfig(vectorizer, "rf"), rng)
