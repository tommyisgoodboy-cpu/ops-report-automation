"""Train ticket classifiers from labelled summaries (standard library only).

Two small models, to compare with the keyword baseline in `classify.py`:

- `NaiveBayes`: multinomial Naive Bayes over words and Chinese character pairs. "Training" = counting.
- `CentroidClassifier`: embed every training summary with a neural embedder and average the vectors per
  category. "Training" = averaging; prediction = nearest category by cosine similarity.
"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

from .rag import _STOP, cosine

_WORD = re.compile(r"[a-z0-9]+")
_CJK_RUN = re.compile(r"[一-鿿]+")


def features(text: str) -> list[str]:
    """English words (stop words removed) plus pairs of neighbouring Chinese characters."""
    text = text.lower()
    feats = [w for w in _WORD.findall(text) if w not in _STOP]
    for run in _CJK_RUN.findall(text):
        feats += [run] if len(run) == 1 else [run[i:i + 2] for i in range(len(run) - 1)]
    return feats


class NaiveBayes:
    """Multinomial Naive Bayes with add-one smoothing. Says "Other" when it recognises no word at all."""

    def __init__(self, fallback: str = "Other"):
        self.fallback = fallback
        self.priors: dict[str, float] = {}
        self.counts: dict[str, Counter] = {}
        self.totals: dict[str, int] = {}
        self.vocab: set[str] = set()

    def train(self, texts: list[str], labels: list[str]) -> "NaiveBayes":
        counts: dict[str, Counter] = defaultdict(Counter)
        docs = Counter(labels)
        for text, label in zip(texts, labels):
            counts[label].update(features(text))
        self.counts = dict(counts)
        self.vocab = {w for c in counts.values() for w in c}
        self.totals = {label: sum(c.values()) for label, c in counts.items()}
        n = sum(docs.values())
        self.priors = {label: math.log(k / n) for label, k in docs.items()}
        return self

    def predict(self, text: str) -> str:
        words = [w for w in features(text) if w in self.vocab]
        if not words:
            return self.fallback
        v = len(self.vocab)
        scores = {
            label: self.priors[label] + sum(math.log((self.counts[label][w] + 1) / (self.totals[label] + v)) for w in words)
            for label in self.priors
        }
        return max(scores, key=scores.get)


class CentroidClassifier:
    """Nearest-centroid over neural embeddings."""

    def __init__(self, embedder):
        self.embedder = embedder
        self.centroids: dict[str, list[float]] = {}

    def train(self, texts: list[str], labels: list[str]) -> "CentroidClassifier":
        vectors = self.embedder.embed(texts)
        sums: dict[str, list[float]] = {}
        for vec, label in zip(vectors, labels):
            acc = sums.setdefault(label, [0.0] * len(vec))
            for i, x in enumerate(vec):
                acc[i] += x
        self.centroids = {label: _unit(acc) for label, acc in sums.items()}
        return self

    def predict(self, text: str) -> str:
        embed_query = getattr(self.embedder, "embed_query", None)
        q = embed_query(text) if embed_query else self.embedder.embed([text])[0]
        return max(self.centroids, key=lambda label: cosine(q, self.centroids[label]))


def _unit(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]
