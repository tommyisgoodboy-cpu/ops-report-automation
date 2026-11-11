"""Minimal RAG: chunk a small knowledge base, embed, retrieve, answer with sources.

The default embedder is TF-IDF (lexical, standard library only). It finds chunks that
share words with the question. It does NOT understand meaning the way a neural
embedding model does. The `Embedder` interface lets a real model be swapped in later
without changing the retrieval or API code.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

KNOWLEDGE_DIR = Path(__file__).parent / "knowledge"
_WORD = re.compile(r"[a-z0-9]+")
_STOP = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "is", "it", "for", "with",
    "if", "then", "that", "this", "at", "as", "be", "by", "not", "are", "can", "my",
    "i", "how", "do", "does", "what", "why", "when", "user", "users",
}


@dataclass(frozen=True)
class Chunk:
    doc: str      # source file name, used as the citation
    heading: str  # section heading inside the document
    text: str


def load_chunks(directory: Path = KNOWLEDGE_DIR) -> list[Chunk]:
    """Split every markdown file into one chunk per '## ' section."""
    chunks: list[Chunk] = []
    for path in sorted(directory.glob("*.md")):
        heading, body = None, []
        for line in path.read_text(encoding="utf-8").splitlines() + ["## "]:
            if line.startswith("## "):
                if heading and body:
                    chunks.append(Chunk(path.name, heading, " ".join(body).strip()))
                heading, body = line[3:].strip(), []
            elif heading is not None and not line.startswith("# "):
                if line.strip():
                    body.append(line.strip())
    return chunks


def tokenize(text: str) -> list[str]:
    return [w for w in _WORD.findall(text.lower()) if w not in _STOP]


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[dict[str, float]]: ...


class TfidfEmbedder:
    """Sparse TF-IDF vectors as {word: weight}. Fit on the corpus, then embed anything."""

    def __init__(self, corpus: list[str]):
        n = len(corpus)
        df = Counter(w for text in corpus for w in set(tokenize(text)))
        # Smoothed IDF: rare words weigh more than words found in every chunk.
        self.idf = {w: math.log((1 + n) / (1 + c)) + 1 for w, c in df.items()}

    def embed(self, texts: list[str]) -> list[dict[str, float]]:
        vectors = []
        for text in texts:
            tf = Counter(w for w in tokenize(text) if w in self.idf)
            vec = {w: c * self.idf[w] for w, c in tf.items()}
            norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
            vectors.append({w: v / norm for w, v in vec.items()})
        return vectors


def cosine(a: dict[str, float], b: dict[str, float]) -> float:
    """Vectors are already normalised, so the dot product is the cosine similarity."""
    if len(a) > len(b):
        a, b = b, a
    return sum(v * b.get(w, 0.0) for w, v in a.items())


class Retriever:
    def __init__(self, chunks: list[Chunk] | None = None, embedder: Embedder | None = None):
        self.chunks = chunks if chunks is not None else load_chunks()
        texts = [f"{c.heading}. {c.text}" for c in self.chunks]
        self.embedder = embedder or TfidfEmbedder(texts)
        self.vectors = self.embedder.embed(texts)

    def search(self, question: str, k: int = 3, min_score: float = 0.05) -> list[tuple[Chunk, float]]:
        q = self.embedder.embed([question])[0]
        scored = [(c, cosine(q, v)) for c, v in zip(self.chunks, self.vectors)]
        scored = [(c, s) for c, s in scored if s >= min_score]
        return sorted(scored, key=lambda cs: cs[1], reverse=True)[:k]


def build_prompt(question: str, hits: list[tuple[Chunk, float]]) -> str:
    """The prompt an LLM would receive: retrieved context + the question."""
    context = "\n\n".join(f"[{i}] ({c.doc}) {c.heading}: {c.text}" for i, (c, _) in enumerate(hits, 1))
    return (
        "Answer using ONLY the context below. Cite sources like [1]. "
        "If the context does not contain the answer, say you do not know.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}"
    )


def answer(question: str, retriever: Retriever, k: int = 3) -> dict:
    """Extractive answer: return the best chunk as the answer, plus all sources.

    No LLM is called here. `build_prompt` shows how the same hits would be given to one.
    """
    hits = retriever.search(question, k=k)
    if not hits:
        return {"question": question, "answer": "I do not know. No relevant article found.", "sources": []}
    best = hits[0][0]
    return {
        "question": question,
        "answer": f"{best.heading}: {best.text}",
        "sources": [{"doc": c.doc, "heading": c.heading, "score": round(s, 3)} for c, s in hits],
    }
