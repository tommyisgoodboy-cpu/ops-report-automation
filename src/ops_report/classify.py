"""Rule-based ticket classifier (keyword scoring).

This is the baseline. An LLM-based classifier can be added later and compared
against this one using the accuracy figure shown in the report.
"""
from __future__ import annotations

KEYWORDS = {
    "Hardware": ["laptop", "monitor", "keyboard", "mouse", "power on", "docking", "pc"],
    "Software": ["excel", "outlook", "application", "install", "update", "software"],
    "Network": ["network", "vpn", "wi-fi", "wifi", "shared drive", "connection"],
    "Access": ["account", "password", "permission", "access", "locked", "mailbox"],
    "Printer": ["printer", "toner", "paper jam"],
    "Virtualisation": ["virtual", "vm", "snapshot"],
}


def classify(summary: str) -> str:
    text = summary.lower()
    scores = {cat: sum(1 for kw in kws if kw in text) for cat, kws in KEYWORDS.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "Other"
