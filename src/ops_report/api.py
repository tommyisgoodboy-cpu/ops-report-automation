"""HTTP API (FastAPI) on top of the same report and classifier code as the CLI.

Run: uvicorn ops_report.api:app --reload   (needs `pip install -r requirements.txt`)
"""
from __future__ import annotations

from typing import Literal

from fastapi import FastAPI, Query
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from . import db
from .classify import classify
from .data import generate_tickets
from .report import build_report

Priority = Literal["P1", "P2", "P3", "P4"]
Status = Literal["Open", "Resolved"]


class ClassifyIn(BaseModel):
    summary: str = Field(min_length=1, max_length=500)


class ClassifyOut(BaseModel):
    summary: str
    category: str


def create_app(n_tickets: int = 300, seed: int = 42) -> FastAPI:
    """Build an app with its own in-memory DB of synthetic tickets."""
    app = FastAPI(title="ops-report-automation", version="0.1.0")
    # check_same_thread=False: FastAPI runs sync endpoints in a thread pool.
    conn = db.connect(check_same_thread=False)
    db.load(conn, generate_tickets(n=n_tickets, seed=seed))

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

    return app


app = create_app()
