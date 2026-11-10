"""Command line entry point: `python -m ops_report`."""
from __future__ import annotations

import argparse
from pathlib import Path

from . import db
from .data import generate_tickets
from .report import build_report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="ops_report", description=__doc__)
    parser.add_argument("--db", default="tickets.db", help="SQLite file (default: tickets.db)")
    parser.add_argument("--out", default="report.md", help="Markdown report path (default: report.md)")
    parser.add_argument("--tickets", type=int, default=300, help="Number of synthetic tickets")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (reproducible data)")
    args = parser.parse_args(argv)

    conn = db.connect(args.db)
    n = db.load(conn, generate_tickets(n=args.tickets, seed=args.seed))
    Path(args.out).write_text(build_report(conn), encoding="utf-8")
    print(f"Loaded {n} synthetic tickets into {args.db}; wrote {args.out}")
    return 0
