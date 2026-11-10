"""Build a Markdown operations report from the tickets table using plain SQL."""
from __future__ import annotations

import sqlite3

from .classify import classify


def _fmt_minutes(minutes: float | None) -> str:
    if minutes is None:
        return "n/a"
    h, m = divmod(int(round(minutes)), 60)
    return f"{h}h {m:02d}m" if h else f"{m}m"


def build_report(conn: sqlite3.Connection) -> str:
    total, resolved, open_ = conn.execute(
        "SELECT COUNT(*), SUM(status='Resolved'), SUM(status='Open') FROM tickets"
    ).fetchone()

    by_cat = conn.execute(
        "SELECT category, COUNT(*) AS n FROM tickets GROUP BY category ORDER BY n DESC"
    ).fetchall()

    by_pri = conn.execute(
        """
        SELECT priority,
               COUNT(*) AS n,
               AVG((julianday(resolved_at) - julianday(created_at)) * 1440) AS mttr,
               SUM((julianday(resolved_at) - julianday(created_at)) * 1440 > sla_minutes) AS breached,
               SUM(status='Resolved') AS done
        FROM tickets GROUP BY priority ORDER BY priority
        """
    ).fetchall()

    repeats = conn.execute(
        "SELECT summary, COUNT(*) AS n FROM tickets GROUP BY summary ORDER BY n DESC LIMIT 5"
    ).fetchall()

    rows = conn.execute("SELECT summary, category FROM tickets").fetchall()
    correct = sum(1 for r in rows if classify(r["summary"]) == r["category"])
    accuracy = correct / len(rows) if rows else 0.0

    lines = [
        "# Operations report",
        "",
        f"- Tickets: **{total}** (resolved {resolved}, open {open_})",
        f"- Classifier accuracy vs. recorded category: **{accuracy:.1%}** ({correct}/{len(rows)})",
        "",
        "## Tickets by category",
        "",
        "| Category | Tickets |",
        "|---|---:|",
    ]
    lines += [f"| {r['category']} | {r['n']} |" for r in by_cat]
    lines += ["", "## Priority: resolution time and SLA", "", "| Priority | Tickets | MTTR | SLA breach rate |", "|---|---:|---:|---:|"]
    for r in by_pri:
        rate = (r["breached"] or 0) / r["done"] if r["done"] else 0.0
        lines.append(f"| {r['priority']} | {r['n']} | {_fmt_minutes(r['mttr'])} | {rate:.1%} |")
    lines += ["", "## Top repeating issues", "", "| Issue | Count |", "|---|---:|"]
    lines += [f"| {r['summary']} | {r['n']} |" for r in repeats]
    lines.append("")
    return "\n".join(lines)
