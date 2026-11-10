import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ops_report import db  # noqa: E402
from ops_report.classify import classify  # noqa: E402
from ops_report.data import generate_tickets  # noqa: E402
from ops_report.report import build_report  # noqa: E402


class OpsReportTests(unittest.TestCase):
    def test_generation_is_reproducible(self):
        self.assertEqual(generate_tickets(50, seed=1), generate_tickets(50, seed=1))

    def test_classifier_known_cases(self):
        self.assertEqual(classify("VPN disconnects every few minutes"), "Network")
        self.assertEqual(classify("Printer shows offline for floor 3"), "Printer")
        self.assertEqual(classify("completely unrelated text"), "Other")

    def test_report_contains_sections(self):
        conn = db.connect()
        db.load(conn, generate_tickets(120, seed=7))
        report = build_report(conn)
        for heading in ("# Operations report", "## Tickets by category", "## Priority", "## Top repeating issues"):
            self.assertIn(heading, report)

    def test_report_counts_match_database(self):
        conn = db.connect()
        n = db.load(conn, generate_tickets(80, seed=3))
        self.assertIn(f"Tickets: **{n}**", build_report(conn))


if __name__ == "__main__":
    unittest.main()
