import os
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    from fastapi.testclient import TestClient
    from ops_report.api import create_app
except ImportError:
    TestClient = None


@unittest.skipIf(TestClient is None, "fastapi / httpx not installed")
class SecurityTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(create_app(n_tickets=10))

    def test_security_headers_on_every_response(self):
        for path in ("/", "/health", "/docs"):
            h = self.client.get(path).headers
            self.assertEqual(h["x-content-type-options"], "nosniff", path)
            self.assertEqual(h["x-frame-options"], "DENY", path)
            self.assertEqual(h["referrer-policy"], "no-referrer", path)

    def test_csp_blocks_framing_and_outside_connections_on_the_demo_page_only(self):
        csp = self.client.get("/").headers["content-security-policy"]
        for part in ("default-src 'none'", "connect-src 'self'", "frame-ancestors 'none'"):
            self.assertIn(part, csp)
        self.assertNotIn("content-security-policy", self.client.get("/docs").headers)  # would break Swagger UI

    def test_switched_off_models_are_listed_but_cannot_be_called(self):
        with mock.patch.dict(os.environ, {"RAG_LLMS_ON_REQUEST": "ollama:qwen2.5:7b"}):
            client = TestClient(create_app(n_tickets=10))
            methods = client.get("/health").json()["answer_methods"]
            r = client.post("/ask", json={"question": "printer offline", "method": "ollama:qwen2.5:7b"})
        self.assertEqual([(m["id"], m["available"]) for m in methods], [("passage", True), ("ollama:qwen2.5:7b", False)])
        self.assertEqual(r.status_code, 400)
        self.assertIn("switched off", r.json()["detail"])

    def test_page_explains_switched_off_models_and_the_privacy_note(self):
        html = self.client.get("/").text
        for text in ("switched off", "schedule a test", "do not enter personal or confidential information"):
            self.assertIn(text, html)

    def test_requirements_are_pinned(self):
        lines = (Path(__file__).resolve().parents[1] / "requirements.txt").read_text().split()
        self.assertTrue(lines and all("==" in line for line in lines), lines)


if __name__ == "__main__":
    unittest.main()
