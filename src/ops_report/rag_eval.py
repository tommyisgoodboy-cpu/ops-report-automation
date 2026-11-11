"""Measure retrieval quality against a small labelled question set.

Run: PYTHONPATH=src python -m ops_report.rag_eval [--embedder ollama] [--min-score 0.5]

Metrics
- top1:   the first retrieved chunk is the expected one (answerable questions)
- top3:   the expected chunk is within the first three
- refuse: out-of-scope questions return nothing (the system says "I do not know")
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .rag import Retriever, make_embedder

QUESTIONS = Path(__file__).resolve().parents[2] / "eval" / "questions.json"


def evaluate(retriever: Retriever, questions: list[dict], k: int = 3) -> dict:
    rows = []
    for item in questions:
        hits = retriever.search(item["q"], k=k)
        headings = [c.heading for c, _ in hits]
        expect = item["expect"]
        if expect is None:  # out of scope: success means no hits at all
            ok1 = ok3 = not hits
        else:
            ok1 = bool(headings) and headings[0] == expect
            ok3 = expect in headings
        rows.append({**item, "got": headings[:1], "top1": ok1, "top3": ok3,
                     "score": round(hits[0][1], 3) if hits else None})

    def rate(kind_filter, key):
        sel = [r for r in rows if kind_filter(r)]
        return sum(r[key] for r in sel) / len(sel) if sel else 0.0

    answerable = lambda r: r["expect"] is not None  # noqa: E731
    return {
        "rows": rows,
        "top1": rate(answerable, "top1"),
        "top3": rate(answerable, "top3"),
        "top1_literal": rate(lambda r: r["type"] == "literal", "top1"),
        "top1_paraphrase": rate(lambda r: r["type"] == "paraphrase", "top1"),
        "top1_cantonese": rate(lambda r: r["type"] == "cantonese", "top1"),
        "refuse": rate(lambda r: r["expect"] is None, "top1"),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedder", choices=["tfidf", "ollama", "vertex"], default="tfidf")
    parser.add_argument("--min-score", type=float, default=None, help="override the refuse threshold")
    args = parser.parse_args(argv)

    questions = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    embedder = make_embedder(args.embedder)
    result = evaluate(Retriever(embedder=embedder, min_score=args.min_score), questions)
    for r in result["rows"]:
        mark = "OK  " if r["top1"] else "MISS"
        want = r["expect"] or "(refuse)"
        got = r["got"][0] if r["got"] else "(nothing)"
        print(f"{mark} [{r['type']:<12}] {r['q']}\n       want: {want}\n       got:  {got}  score={r['score']}")
    print()
    print(f"top-1 (answerable)   : {result['top1']:.0%}")
    print(f"  literal questions  : {result['top1_literal']:.0%}")
    print(f"  paraphrased        : {result['top1_paraphrase']:.0%}")
    print(f"  Cantonese / mixed  : {result['top1_cantonese']:.0%}")
    print(f"top-3 (answerable)   : {result['top3']:.0%}")
    print(f"refuses out-of-scope : {result['refuse']:.0%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
