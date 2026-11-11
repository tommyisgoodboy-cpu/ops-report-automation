"""HTTP API (FastAPI) on top of the same report and classifier code as the CLI.

Run: uvicorn ops_report.api:app --reload   (needs `pip install -r requirements.txt`)
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Literal

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, PlainTextResponse
from pydantic import BaseModel, Field

from . import db
from .classify import classify
from .data import generate_tickets
from .ratelimit import RateLimiter, make_limit_dependency
from .rag import Retriever, answer, make_embedder, make_llm
from .report import build_report

WEB_PAGE = (Path(__file__).parent / "web" / "index.html").read_text(encoding="utf-8")

Priority = Literal["P1", "P2", "P3", "P4"]
Status = Literal["Open", "Resolved"]


class ClassifyIn(BaseModel):
    summary: str = Field(min_length=1, max_length=500)


class ClassifyOut(BaseModel):
    summary: str
    category: str


class AskIn(BaseModel):
    question: str = Field(min_length=1, max_length=300)
    k: int = Field(3, ge=1, le=5)
    # Which way to write the answer. Checked against the server's own allow-list, never trusted.
    method: str = Field("passage", min_length=1, max_length=20)


class Source(BaseModel):
    doc: str
    heading: str
    score: float


class AskOut(BaseModel):
    question: str
    answer: str
    passage: str  # the original best passage, so a client can show it next to a generated answer
    sources: list[Source]
    mode: Literal["extractive", "generated"]


def create_app(n_tickets: int = 300, seed: int = 42, rate_limit: int | None = None) -> FastAPI:
    """Build an app with its own in-memory DB of synthetic tickets.

    rate_limit: max requests per client per minute on the data endpoints.
    None reads RATE_LIMIT_PER_MINUTE (default 30); 0 disables the limit.
    """
    app = FastAPI(title="ops-report-automation", version="0.2.0")
    if rate_limit is None:
        rate_limit = int(os.environ.get("RATE_LIMIT_PER_MINUTE", "30"))
    trust_proxy = os.environ.get("RATE_LIMIT_TRUST_PROXY") == "1"
    limited = Depends(make_limit_dependency(RateLimiter(rate_limit) if rate_limit > 0 else None, trust_proxy))
    # Chunk and embed the knowledge base once at startup, not on every question.
    # RAG_EMBEDDER = tfidf (default) | ollama (local) | vertex (Google Cloud, no key needed on Cloud Run)
    embedder_name = (os.environ.get("RAG_EMBEDDER") or "tfidf").lower()
    try:
        retriever = Retriever(embedder=make_embedder(embedder_name))
    except Exception as exc:  # noqa: BLE001 - an external dependency must not stop the service booting
        logging.getLogger(__name__).warning("Embedder %r unavailable (%s); falling back to tfidf", embedder_name, exc)
        embedder_name, retriever = "tfidf", Retriever()
    # Answer methods this server offers. "passage" (the original best passage) is always there.
    # RAG_LLMS = comma list of extra methods: ollama (local Qwen), vertex (Gemini). RAG_LLM is the old single name.
    methods: dict[str, tuple[object | None, str]] = {"passage": (None, "Original passage")}
    for name in (os.environ.get("RAG_LLMS") or os.environ.get("RAG_LLM") or "").lower().replace(" ", "").split(","):
        if name and name != "none" and name not in methods:
            llm = make_llm(name)  # unknown names fail at startup, not on a visitor's request
            methods[name] = (llm, f"{'Local' if name == 'ollama' else 'Cloud'} model: {llm.model}")
    # check_same_thread=False: FastAPI runs sync endpoints in a thread pool.
    conn = db.connect(check_same_thread=False)
    db.load(conn, generate_tickets(n=n_tickets, seed=seed))

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def home() -> str:
        return WEB_PAGE

    @app.get("/health")
    def health() -> dict:
        return {
            "status": "ok",
            "embedder": embedder_name,
            "answer_methods": [{"id": k, "label": label} for k, (_, label) in methods.items()],
        }

    @app.get("/report", response_class=PlainTextResponse, dependencies=[limited])
    def report() -> str:
        return build_report(conn)

    @app.get("/tickets", dependencies=[limited])
    def tickets(
        category: str | None = None,
        priority: Priority | None = None,
        status: Status | None = None,
        limit: int = Query(20, ge=1, le=100),
    ) -> list[dict]:
        # Filters are bound parameters, never string-formatted into the SQL.
        sql = "SELECT * FROM tickets WHERE 1=1"
        params: list = []
        for column, value in (("category", category), ("priority", priority), ("status", status)):
            if value is not None:
                sql += f" AND {column} = ?"
                params.append(value)
        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        return [dict(r) for r in conn.execute(sql, params).fetchall()]

    @app.post("/classify", response_model=ClassifyOut, dependencies=[limited])
    def classify_ticket(body: ClassifyIn) -> ClassifyOut:
        return ClassifyOut(summary=body.summary, category=classify(body.summary))

    @app.post("/ask", response_model=AskOut, dependencies=[limited])
    def ask(body: AskIn) -> dict:
        if body.method not in methods:
            raise HTTPException(status_code=400, detail=f"Unknown answer method. Use one of: {', '.join(methods)}")
        try:
            return answer(body.question, retriever, k=body.k, llm=methods[body.method][0])
        except RuntimeError as exc:  # e.g. the embedding service is unreachable
            logging.getLogger(__name__).error("ask failed: %s", exc)
            raise HTTPException(status_code=503, detail="Retrieval service temporarily unavailable") from exc

    return app


app = create_app()
