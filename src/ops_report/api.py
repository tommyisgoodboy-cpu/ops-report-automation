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
from .rag import (MAX_HISTORY_MESSAGES, PAID_PROVIDERS, PROVIDER_LABELS, CappedEmbedder, CappedLLM, FallbackRetriever,
                  Retriever, answer, chat_turn, list_ollama_models, make_embedder, make_llm, parse_llm_spec)
from .report import build_report

WEB_PAGE = (Path(__file__).parent / "web" / "index.html").read_text(encoding="utf-8")
CHAT_PAGE = (Path(__file__).parent / "web" / "chat.html").read_text(encoding="utf-8")

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
    method: str = Field("passage", min_length=1, max_length=80)


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


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=300)


class ChatIn(BaseModel):
    # The client sends the conversation on every call; the server keeps no state. Oldest first, last = user.
    messages: list[ChatMessage] = Field(min_length=1, max_length=12)
    method: str = Field("passage", min_length=1, max_length=80)
    k: int = Field(3, ge=1, le=5)


class ChatOut(BaseModel):
    reply: str
    action: Literal["chat", "search"]
    query: str  # the standalone question that was actually searched
    passage: str
    sources: list[Source]
    mode: Literal["extractive", "generated"]


def create_app(n_tickets: int = 300, seed: int = 42, rate_limit: int | None = None) -> FastAPI:
    """Build an app with its own in-memory DB of synthetic tickets.

    rate_limit: max requests per client per minute on the data endpoints.
    None reads RATE_LIMIT_PER_MINUTE (default 30); 0 disables the limit.
    """
    app = FastAPI(title="ops-report-automation", version="0.2.0")

    @app.middleware("http")
    async def security_headers(request, call_next):
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        if request.url.path in ("/", "/chat"):  # the demo pages only; /docs (Swagger UI) loads its own assets
            response.headers.setdefault(
                "Content-Security-Policy",
                "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; "
                "connect-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'",
            )
        return response
    if rate_limit is None:
        rate_limit = int(os.environ.get("RATE_LIMIT_PER_MINUTE", "30"))
    trust_proxy = os.environ.get("RATE_LIMIT_TRUST_PROXY") == "1"
    limited = Depends(make_limit_dependency(RateLimiter(rate_limit) if rate_limit > 0 else None, trust_proxy))
    # Chunk and embed the knowledge base once at startup, not on every question.
    # RAG_EMBEDDER = tfidf (default) | ollama (local) | vertex (Google Cloud, no key needed on Cloud Run)
    embedder_name = (os.environ.get("RAG_EMBEDDER") or "tfidf").lower()
    try:
        embedder = make_embedder(embedder_name)
        embed_limit = int(os.environ.get("EMBED_DAILY_LIMIT", "3000"))  # paid embedders only; 0 turns it off
        if embedder is not None and embedder_name == "vertex" and embed_limit > 0:
            embedder = CappedEmbedder(embedder, embed_limit)
        retriever = Retriever(embedder=embedder)
    except Exception as exc:  # noqa: BLE001 - an external dependency must not stop the service booting
        logging.getLogger(__name__).warning("Embedder %r unavailable (%s); falling back to tfidf", embedder_name, exc)
        embedder_name, retriever = "tfidf", Retriever()
    if embedder_name != "tfidf":
        # If the neural embedder fails later (outage, daily cap), keep answering with plain TF-IDF.
        retriever = FallbackRetriever(retriever, Retriever(retriever.chunks))
    # Answer methods this server offers. "passage" (the original best passage) is always there.
    # RAG_LLMS = comma list of "provider" or "provider:model", for example
    #   ollama:*                    every model downloaded in the local Ollama (found at startup)
    #   ollama:qwen2.5:7b,vertex    one local model plus the cloud default
    # RAG_LLM (a single name) still works. Unknown providers fail at startup, not on a visitor's request.
    methods: dict[str, tuple[object | None, str]] = {"passage": (None, "Original passage")}
    # Hard stop on paid answers per day (in memory, per instance); 0 turns it off. The cloud side has no
    # per-request quota to lower for these models, so this is the cap the code controls.
    daily_limit = int(os.environ.get("LLM_DAILY_LIMIT", "200"))
    specs: list[str] = []
    for spec in (os.environ.get("RAG_LLMS") or os.environ.get("RAG_LLM") or "").replace(" ", "").split(","):
        if not spec or spec.lower() == "none":
            continue
        specs += [f"ollama:{m}" for m in list_ollama_models()] if spec.lower() == "ollama:*" else [spec]
    for spec in specs:
        if spec in methods:
            continue
        llm = make_llm(spec)
        provider = parse_llm_spec(spec)[0]
        if provider in PAID_PROVIDERS and daily_limit > 0:
            llm = CappedLLM(llm, daily_limit)
        methods[spec] = (llm, f"{PROVIDER_LABELS.get(provider, provider)}: {llm.model}")
    # Models that exist but are switched off here (for example a local model that runs on the author's
    # own computer). They are listed so visitors can see them, but they cannot be called.
    # RAG_LLMS_ON_REQUEST = comma list of "provider:model".
    switched_off: dict[str, str] = {}
    for spec in (os.environ.get("RAG_LLMS_ON_REQUEST") or "").replace(" ", "").split(","):
        if spec and spec not in methods:
            llm = make_llm(spec)
            provider = parse_llm_spec(spec)[0]
            switched_off[spec] = f"{PROVIDER_LABELS.get(provider, provider)}: {llm.model}"
    # check_same_thread=False: FastAPI runs sync endpoints in a thread pool.
    conn = db.connect(check_same_thread=False)
    db.load(conn, generate_tickets(n=n_tickets, seed=seed))

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def home() -> str:
        return WEB_PAGE

    @app.get("/chat", response_class=HTMLResponse, include_in_schema=False)
    def chat_page() -> str:
        return CHAT_PAGE

    @app.get("/health")
    def health() -> dict:
        return {
            "status": "ok",
            "embedder": embedder_name,
            "answer_methods": [{"id": k, "label": label, "available": True} for k, (_, label) in methods.items()]
            + [{"id": k, "label": label, "available": False} for k, label in switched_off.items()],
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
        if body.method in switched_off:
            raise HTTPException(status_code=400, detail="This model is switched off. Ask the author to switch it on.")
        if body.method not in methods:
            raise HTTPException(status_code=400, detail=f"Unknown answer method. Use one of: {', '.join(methods)}")
        try:
            return answer(body.question, retriever, k=body.k, llm=methods[body.method][0])
        except RuntimeError as exc:  # e.g. the embedding service is unreachable
            logging.getLogger(__name__).error("ask failed: %s", exc)
            raise HTTPException(status_code=503, detail="Retrieval service temporarily unavailable") from exc

    @app.post("/chat", response_model=ChatOut, dependencies=[limited])
    def chat(body: ChatIn) -> dict:
        if body.messages[-1].role != "user":
            raise HTTPException(status_code=422, detail="The last message must be from the user")
        if body.method in switched_off:
            raise HTTPException(status_code=400, detail="This model is switched off. Ask the author to switch it on.")
        if body.method not in methods:
            raise HTTPException(status_code=400, detail=f"Unknown answer method. Use one of: {', '.join(methods)}")
        messages = [m.model_dump() for m in body.messages][-(MAX_HISTORY_MESSAGES + 1):]
        try:
            return chat_turn(messages, retriever, llm=methods[body.method][0], k=body.k)
        except RuntimeError as exc:  # e.g. the embedding service is unreachable
            logging.getLogger(__name__).error("chat failed: %s", exc)
            raise HTTPException(status_code=503, detail="Retrieval service temporarily unavailable") from exc

    return app


app = create_app()
