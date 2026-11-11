import datetime
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ops_report import rag  # noqa: E402
from ops_report.rag import CappedEmbedder, FallbackRetriever, Retriever  # noqa: E402

try:
    from fastapi.testclient import TestClient
    from ops_report.api import create_app
except ImportError:
    TestClient = None


class FakeEmbedder:
    default_min_score = 0.5

    def __init__(self):
        self.queries = 0

    def embed(self, texts):
        return [[1.0, 0.0] for _ in texts]

    def embed_query(self, text):
        self.queries += 1
        return [1.0, 0.0]


class CappedEmbedderTests(unittest.TestCase):
    def test_allows_questions_up_to_the_limit_then_refuses(self):
        capped = CappedEmbedder(FakeEmbedder(), 2)
        capped.embed_query("a"), capped.embed_query("b")
        with self.assertRaises(RuntimeError) as ctx:
            capped.embed_query("c")
        self.assertIn("Daily limit of 2", str(ctx.exception))

    def test_embedding_the_knowledge_base_does_not_use_up_the_daily_allowance(self):
        capped = CappedEmbedder(FakeEmbedder(), 1)
        capped.embed(["chunk"] * 50)
        capped.embed_query("first question")  # still allowed

    def test_the_count_resets_on_a_new_day(self):
        day = [datetime.date(2026, 11, 11)]
        capped = CappedEmbedder(FakeEmbedder(), 1, today=lambda: day[0])
        capped.embed_query("a")
        with self.assertRaises(RuntimeError):
            capped.embed_query("b")
        day[0] = datetime.date(2026, 11, 12)
        capped.embed_query("c")

    def test_the_threshold_of_the_wrapped_embedder_is_kept(self):
        self.assertEqual(CappedEmbedder(FakeEmbedder(), 1).default_min_score, 0.5)


class FallbackRetrieverTests(unittest.TestCase):
    def test_uses_tfidf_when_the_main_retrieval_fails(self):
        class Broken:
            chunks = Retriever().chunks

            def search(self, *a, **k):
                raise RuntimeError("service down")

        out = FallbackRetriever(Broken(), Retriever()).search("printer shows offline")
        self.assertEqual(out[0][0].doc, "printer.md")

    def test_does_not_touch_the_fallback_while_the_main_one_works(self):
        class Fallback:
            def search(self, *a, **k):
                raise AssertionError("fallback should not be used")

        main = Retriever()
        self.assertEqual(FallbackRetriever(main, Fallback()).search("printer shows offline")[0][0].doc, "printer.md")


@unittest.skipIf(TestClient is None, "fastapi / httpx not installed")
class ApiDegradationTests(unittest.TestCase):
    def test_if_the_neural_embedder_fails_for_a_question_the_answer_still_comes_from_tfidf(self):
        with mock.patch.dict(os.environ, {"RAG_EMBEDDER": "vertex"}), \
             mock.patch.object(rag.VertexEmbedder, "embed", lambda self, texts: [[1.0, 0.0] for _ in texts]), \
             mock.patch.object(rag.VertexEmbedder, "embed_query", side_effect=RuntimeError("403")):
            client = TestClient(create_app(n_tickets=10))
            r = client.post("/ask", json={"question": "printer shows offline"})
            chat = client.post("/chat", json={"messages": [{"role": "user", "content": "printer shows offline"}]})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["sources"][0]["doc"], "printer.md")
        self.assertEqual(chat.status_code, 200)

    def test_the_daily_embedding_cap_is_applied_to_vertex_and_degrades_instead_of_failing(self):
        with mock.patch.dict(os.environ, {"RAG_EMBEDDER": "vertex", "EMBED_DAILY_LIMIT": "1"}), \
             mock.patch.object(rag.VertexEmbedder, "embed", lambda self, texts: [[1.0, 0.0] for _ in texts]), \
             mock.patch.object(rag.VertexEmbedder, "embed_query", return_value=[1.0, 0.0]) as q:
            client = TestClient(create_app(n_tickets=10))
            codes = [client.post("/ask", json={"question": "printer shows offline"}).status_code for _ in range(3)]
        self.assertEqual(codes, [200, 200, 200])
        self.assertEqual(q.call_count, 1)  # the 2nd and 3rd question never reached the paid service


if __name__ == "__main__":
    unittest.main()
