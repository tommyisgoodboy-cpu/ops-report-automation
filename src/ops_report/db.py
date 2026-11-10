"""SQLite storage for tickets."""
from __future__ import annotations

import sqlite3

SCHEMA = """
CREATE TABLE IF NOT EXISTS tickets (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    resolved_at TEXT,
    category TEXT NOT NULL,
    priority TEXT NOT NULL,
    summary TEXT NOT NULL,
    sla_minutes INTEGER NOT NULL,
    status TEXT NOT NULL
);
"""


def connect(path: str = ":memory:") -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


def load(conn: sqlite3.Connection, tickets) -> int:
    conn.execute("DELETE FROM tickets")
    conn.executemany(
        "INSERT INTO tickets VALUES (:id,:created_at,:resolved_at,:category,:priority,:summary,:sla_minutes,:status)",
        tickets,
    )
    conn.commit()
    return conn.execute("SELECT COUNT(*) FROM tickets").fetchone()[0]
