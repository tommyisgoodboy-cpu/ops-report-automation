"""Synthetic ticket generator. No real customer, employer or government data is used."""
from __future__ import annotations

import random
from datetime import datetime, timedelta

CATEGORIES = {
    "Hardware": [
        "Laptop will not power on after docking",
        "Monitor flickers and loses signal",
        "Keyboard keys not responding on desktop PC",
        "Mouse pointer freezes intermittently",
    ],
    "Software": [
        "Excel crashes when opening large workbook",
        "Outlook keeps asking for password",
        "Application update fails with error code",
        "Cannot install approved software package",
    ],
    "Network": [
        "No network connection at desk, cable plugged in",
        "VPN disconnects every few minutes",
        "Wi-Fi very slow in meeting room",
        "Cannot reach shared drive over network",
    ],
    "Access": [
        "Account locked after password change",
        "Need access to shared mailbox",
        "Password reset request for new joiner",
        "Permission denied on project folder",
    ],
    "Printer": [
        "Printer shows offline for floor 3",
        "Paper jam and toner warning on printer",
        "Cannot add network printer on laptop",
    ],
    "Virtualisation": [
        "Virtual machine very slow and unresponsive",
        "VM snapshot request before patching",
        "Virtual desktop session drops after login",
    ],
}

# (priority, SLA in minutes, weight)
PRIORITIES = [("P1", 60, 0.05), ("P2", 240, 0.20), ("P3", 480, 0.50), ("P4", 1440, 0.25)]


def generate_tickets(n: int = 300, days: int = 14, seed: int = 42, end: datetime | None = None):
    """Return a list of ticket dicts. Same seed -> same data (reproducible)."""
    rng = random.Random(seed)
    end = end or datetime(2026, 11, 10, 18, 0, 0)
    start = end - timedelta(days=days)
    names = [p for p, _, _ in PRIORITIES]
    weights = [w for _, _, w in PRIORITIES]
    sla = {p: m for p, m, _ in PRIORITIES}
    tickets = []
    for i in range(1, n + 1):
        category = rng.choice(list(CATEGORIES))
        summary = rng.choice(CATEGORIES[category])
        priority = rng.choices(names, weights)[0]
        created = start + timedelta(seconds=rng.randint(0, int((end - start).total_seconds())))
        # resolution time: mostly within SLA, sometimes breaches
        factor = rng.lognormvariate(-0.5, 0.7)
        minutes = max(5, int(sla[priority] * factor))
        resolved = created + timedelta(minutes=minutes)
        status = "Resolved" if resolved <= end else "Open"
        tickets.append(
            {
                "id": f"T{i:05d}",
                "created_at": created.isoformat(timespec="seconds"),
                "resolved_at": resolved.isoformat(timespec="seconds") if status == "Resolved" else None,
                "category": category,
                "priority": priority,
                "summary": summary,
                "sla_minutes": sla[priority],
                "status": status,
            }
        )
    return tickets
