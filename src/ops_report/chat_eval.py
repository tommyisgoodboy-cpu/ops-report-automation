"""Measure the chat behaviour on labelled multi-turn conversations.

Run: PYTHONPATH=src python -m ops_report.chat_eval [--embedder ollama] [--llm ollama:qwen2.5:7b]

Per case:
- action "chat":   the assistant must treat the last message as small talk (no search);
- action "search": it must search, and the top source must be the expected passage;
- action "any" with expect_no_sources: nothing may be retrieved (an unrelated question).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .rag import Retriever, chat_turn, make_embedder, make_llm

CASES = Path(__file__).resolve().parents[2] / "eval" / "conversations.json"


def run_case(case: dict, retriever: Retriever, llm) -> dict:
    messages = [{"role": r, "content": c} for r, c in case["history"]] + [{"role": "user", "content": case["q"]}]
    out = chat_turn(messages, retriever, llm=llm)
    top = out["sources"][0]["heading"] if out["sources"] else None
    if case["action"] == "chat":
        ok = out["action"] == "chat"
    elif case["action"] == "search":
        ok = out["action"] == "search" and top == case["expect"]
    else:
        ok = not out["sources"]
    return {"id": case["id"], "ok": ok, "action": out["action"], "query": out["query"], "top": top}


def evaluate(retriever: Retriever, llm, cases: list[dict]) -> dict:
    rows = [run_case(c, retriever, llm) for c in cases]
    group = lambda a: [r for r, c in zip(rows, cases) if c["action"] == a]  # noqa: E731
    rate = lambda rs: sum(r["ok"] for r in rs) / len(rs) if rs else 0.0  # noqa: E731
    return {"rows": rows, "overall": rate(rows), "search": rate(group("search")), "chat": rate(group("chat")),
            "unrelated": rate(group("any"))}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedder", choices=["tfidf", "ollama", "vertex"], default="tfidf")
    parser.add_argument("--llm", default="none", help="none, or provider:model such as ollama:qwen2.5:7b")
    args = parser.parse_args(argv)
    cases = json.loads(CASES.read_text(encoding="utf-8"))
    result = evaluate(Retriever(embedder=make_embedder(args.embedder)), make_llm(args.llm), cases)
    for r in result["rows"]:
        print(f"{'OK  ' if r['ok'] else 'MISS'} {r['id']:<22} action={r['action']:<6} top={r['top']}  query={r['query']!r}")
    print(f"\noverall {result['overall']:.0%} | follow-up questions {result['search']:.0%} | "
          f"small talk {result['chat']:.0%} | unrelated {result['unrelated']:.0%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
