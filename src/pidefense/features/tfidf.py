"""Sparse word and character TF-IDF features fitted exclusively on training text."""

from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore[import-untyped]
from sklearn.pipeline import FeatureUnion  # type: ignore[import-untyped]

from pidefense.config import Vectorizer


def build_vectorizer(kind: Vectorizer = "tfidf_combo") -> TfidfVectorizer | FeatureUnion:
    """Build word (1,2) and/or char (3,5) features without fitting a vocabulary.

    Each branch uses tf * (log((1+n)/(1+df)) + 1), then row L2 normalization.
    Combo concatenates the independently normalized sparse branches equally.
    """
    word = TfidfVectorizer(analyzer="word", ngram_range=(1, 2))
    char = TfidfVectorizer(analyzer="char", ngram_range=(3, 5))
    if kind == "tfidf_word":
        return word
    if kind == "tfidf_char":
        return char
    if kind == "tfidf_combo":
        return FeatureUnion([("word", word), ("char", char)], n_jobs=1)
    raise ValueError("Unsupported TF-IDF representation")
