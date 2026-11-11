import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    from fastapi.testclient import TestClient
    from ops_report.api import create_app
except ImportError:  # API deps are optional; CLI tests still run without them
    TestClient = None


@unittest.skipIf(TestClient is None, "fastapi / httpx not installed")
class ApiTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(create_app(n_tickets=100, seed=5))

    def test_home_page_is_html_with_form(self):
        r = self.client.get("/")
        self.assertEqual(r.status_code, 200)
        self.assertIn("text/html", r.headers["content-type"])
        self.assertIn('id="form"', r.text)
        self.assertIn("/ask", r.text)
        self.assertNotIn("innerHTML", r.text)  # answers must be shown as text, not HTML

    def test_health(self):
        r = self.client.get("/health")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json(), {"status": "ok", "embedder": "tfidf"})

    def test_report_is_markdown_with_ticket_count(self):
        r = self.client.get("/report")
        self.assertEqual(r.status_code, 200)
        self.assertIn("# Operations report", r.text)
        self.assertIn("Tickets: **100**", r.text)

    def test_tickets_filter_and_limit(self):
        r = self.client.get("/tickets", params={"category": "Network", "limit": 5})
        self.assertEqual(r.status_code, 200)
        rows = r.json()
        self.assertTrue(0 < len(rows) <= 5)
        self.assertTrue(all(t["category"] == "Network" for t in rows))

    def test_tickets_rejects_bad_priority_and_limit(self):
        self.assertEqual(self.client.get("/tickets", params={"priority": "P9"}).status_code, 422)
        self.assertEqual(self.client.get("/tickets", params={"limit": 0}).status_code, 422)

    def test_sql_injection_attempt_returns_nothing(self):
        r = self.client.get("/tickets", params={"category": "x' OR '1'='1"})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json(), [])

    def test_classify(self):
        r = self.client.post("/classify", json={"summary": "VPN disconnects every few minutes"})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["category"], "Network")

    def test_classify_rejects_empty(self):
        self.assertEqual(self.client.post("/classify", json={"summary": ""}).status_code, 422)

    def test_ask_returns_answer_with_sources(self):
        r = self.client.post("/ask", json={"question": "printer says offline"})
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertIn("Printer shows offline", body["answer"])
        self.assertEqual(body["sources"][0]["doc"], "printer.md")
        self.assertEqual(body["mode"], "extractive")  # default build has no LLM

    def test_ask_unknown_topic_says_do_not_know(self):
        body = self.client.post("/ask", json={"question": "how to bake bread"}).json()
        self.assertIn("do not know", body["answer"].lower())
        self.assertEqual(body["sources"], [])

    def test_ask_validates_input(self):
        self.assertEqual(self.client.post("/ask", json={"question": ""}).status_code, 422)
        self.assertEqual(self.client.post("/ask", json={"question": "vpn", "k": 99}).status_code, 422)

    def test_unavailable_embedder_at_startup_falls_back_to_tfidf(self):
        import os
        from unittest import mock

        with mock.patch.dict(os.environ, {"RAG_EMBEDDER": "ollama"}), \
             mock.patch("ops_report.rag.OllamaEmbedder.embed", side_effect=RuntimeError("down")):
            client = TestClient(create_app(n_tickets=10))
        self.assertEqual(client.get("/health").json()["embedder"], "tfidf")
        self.assertEqual(client.post("/ask", json={"question": "printer offline"}).status_code, 200)

    def test_embedding_failure_during_a_question_returns_503(self):
        from unittest import mock

        with mock.patch("ops_report.api.answer", side_effect=RuntimeError("down")):
            r = self.client.post("/ask", json={"question": "printer offline"})
        self.assertEqual(r.status_code, 503)


if __name__ == "__main__":
    unittest.main()
