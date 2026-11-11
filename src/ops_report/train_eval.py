"""Compare the keyword classifier with trained ones, without letting the test leak into training.

Run: PYTHONPATH=src python -m ops_report.train_eval [--embedder ollama|vertex]

The synthetic tickets contain only a handful of distinct sentences per category, so scoring a model on
sentences it was trained on proves nothing. Three views, from least to most honest:

1. in-sample: trained and tested on the same sentences (shown only to show why it is not used);
2. leave one sentence out: train on every other sentence, test on the one that was held out;
3. paraphrases: train on all sentences, test on `eval/ticket_paraphrases.json` (new wording, some Cantonese).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .classify import classify
from .data import CATEGORIES
from .rag import make_embedder
from .train import CentroidClassifier, NaiveBayes

PARAPHRASES = Path(__file__).resolve().parents[2] / "eval" / "ticket_paraphrases.json"


class CachingEmbedder:
    """Remembers vectors so the same sentence is embedded once across all folds."""

    def __init__(self, inner):
        self.inner, self.cache = inner, {}
        self.default_min_score = getattr(inner, "default_min_score", 0.0)

    def embed(self, texts):
        missing = [t for t in dict.fromkeys(texts) if t not in self.cache]
        if missing:
            self.cache.update(zip(missing, self.inner.embed(missing)))
        return [self.cache[t] for t in texts]

    def embed_query(self, text):
        if text not in self.cache:
            inner_query = getattr(self.inner, "embed_query", None)
            self.cache[text] = inner_query(text) if inner_query else self.inner.embed([text])[0]
        return self.cache[text]


def training_set() -> tuple[list[str], list[str]]:
    pairs = [(text, label) for label, texts in CATEGORIES.items() for text in texts]
    return [t for t, _ in pairs], [l for _, l in pairs]


def accuracy(predict, items: list[tuple[str, str]]) -> float:
    return sum(predict(text) == label for text, label in items) / len(items)


def models(embedder):
    makers = {"keyword rules (no training)": None, "naive bayes": lambda: NaiveBayes()}
    if embedder is not None:
        makers["embedding centroid"] = lambda: CentroidClassifier(embedder)
    return makers


def evaluate(embedder=None, paraphrases: list[dict] | None = None) -> dict:
    texts, labels = training_set()
    paraphrases = paraphrases if paraphrases is not None else json.loads(PARAPHRASES.read_text(encoding="utf-8"))
    para = [(p["q"], p["category"]) for p in paraphrases]
    results: dict[str, dict] = {}
    for name, make in models(embedder).items():
        if make is None:  # the keyword baseline has nothing to train
            in_sample = accuracy(classify, list(zip(texts, labels)))
            loo = in_sample
            predict = classify
        else:
            full = make().train(texts, labels)
            in_sample = accuracy(full.predict, list(zip(texts, labels)))
            hits = 0
            for i, (t, l) in enumerate(zip(texts, labels)):
                rest = make().train(texts[:i] + texts[i + 1:], labels[:i] + labels[i + 1:])
                hits += rest.predict(t) == l
            loo = hits / len(texts)
            predict = full.predict
        wrong = [(q, c, predict(q)) for q, c in para if predict(q) != c]
        results[name] = {"in_sample": in_sample, "leave_one_out": loo, "paraphrases": 1 - len(wrong) / len(para),
                         "wrong": wrong}
    return results


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedder", choices=["none", "ollama", "vertex"], default="none")
    args = parser.parse_args(argv)
    inner = make_embedder(args.embedder) if args.embedder != "none" else None
    embedder = CachingEmbedder(inner) if inner is not None else None
    results = evaluate(embedder)
    print(f"{'model':<30}{'in-sample':>11}{'leave-one-out':>15}{'paraphrases':>13}")
    for name, r in results.items():
        print(f"{name:<30}{r['in_sample']:>11.0%}{r['leave_one_out']:>15.0%}{r['paraphrases']:>13.0%}")
    for name, r in results.items():
        if r["wrong"]:
            print(f"\n{name}: {len(r['wrong'])} wrong on paraphrases")
            for q, want, got in r["wrong"]:
                print(f"   {q!r:55} want {want:<15} got {got}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
