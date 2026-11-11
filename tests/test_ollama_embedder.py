import json
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ops_report.rag import Chunk, OllamaEmbedder, Retriever, cosine  # noqa: E402


class FakeResponse:
    def __init__(self, payload):
        self.payload = json.dumps(payload).encode()

    def read(self, *a):
        return self.payload

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class OllamaEmbedderTests(unittest.TestCase):
    def test_vectors_are_normalised(self):
        fake = FakeResponse({"embeddings": [[3.0, 4.0]]})
        with mock.patch("urllib.request.urlopen", return_value=fake):
            vec = OllamaEmbedder().embed(["x"])[0]
        self.assertAlmostEqual(vec[0], 0.6)
        self.assertAlmostEqual(vec[1], 0.8)

    def test_cosine_works_for_dense_vectors(self):
        self.assertAlmostEqual(cosine([1.0, 0.0], [1.0, 0.0]), 1.0)
        self.assertAlmostEqual(cosine([1.0, 0.0], [0.0, 1.0]), 0.0)

    def test_unreachable_server_gives_a_helpful_error(self):
        with self.assertRaises(RuntimeError) as ctx:
            OllamaEmbedder(host="http://127.0.0.1:1", timeout=1).embed(["x"])
        self.assertIn("ollama serve", str(ctx.exception))

    def test_retriever_uses_the_embedders_default_threshold(self):
        class Fixed:
            default_min_score = 0.9

            def embed(self, texts):
                return [[1.0, 0.0] if "match" in t else [0.6, 0.8] for t in texts]

        chunks = [Chunk("a.md", "match", "match"), Chunk("b.md", "other", "other")]
        r = Retriever(chunks, embedder=Fixed())
        self.assertEqual(r.min_score, 0.9)
        hits = r.search("match")
        self.assertEqual([c.doc for c, _ in hits], ["a.md"])  # 0.6 < 0.9 is dropped


if __name__ == "__main__":
    unittest.main()
