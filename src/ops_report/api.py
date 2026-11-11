"""HTTP API (FastAPI) on top of the same report and classifier code as the CLI.

Run: uvicorn ops_report.api:app --reload   (needs `pip install -r requirements.txt`)
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, PlainTextResponse
from pydantic import BaseModel, Field

from . import db
from .classify import classify
from .data import generate_tickets
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


class Source(BaseModel):
    doc: str
    heading: str
    score: float


class AskOut(BaseModel):
    question: str
    answer: str
    sources: list[Source]
    mode: Literal["extractive", "generated"]


def create_app(n_tickets: int = 300, seed: int = 42) -> FastAPI:
    """Build an app with its own in-memory DB of synthetic tickets."""
    app = FastAPI(title="ops-report-automation", version="0.2.0")
    # Chunk and embed the knowledge base once at startup, not on every question.
    # RAG_EMBEDDER = tfidf (default) | ollama (local) | vertex (Google Cloud, no key needed on Cloud Run)
    retriever = Retriever(embedder=make_embedder(os.environ.get("RAG_EMBEDDER")))
    # RAG_LLM = none (default, returns the best passage) | vertex (Gemini writes the answer from the passages)
    llm = make_llm(os.environ.get("RAG_LLM"))
    # check_same_thread=False: FastAPI runs sync endpoints in a thread pool.
    conn = db.connect(check_same_thread=False)
    db.load(conn, generate_tickets(n=n_tickets, seed=seed))

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def home() -> str:
        return WEB_PAGE

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/report", response_class=PlainTextResponse)
    def report() -> str:
        return build_report(conn)

    @app.get("/tickets")
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

    @app.post("/classify", response_model=ClassifyOut)
    def classify_ticket(body: ClassifyIn) -> ClassifyOut:
        return ClassifyOut(summary=body.summary, category=classify(body.summary))

    @app.post("/ask", response_model=AskOut)
    def ask(body: AskIn) -> dict:
        return answer(body.question, retriever, k=body.k, llm=llm)

    return app


app = create_app()
