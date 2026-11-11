import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ops_report.rag import Chunk, Retriever, answer, build_prompt, load_chunks  # noqa: E402


class RagTests(unittest.TestCase):
    def test_load_chunks_one_per_section_with_source(self):
        chunks = load_chunks()
        self.assertGreaterEqual(len(chunks), 10)
        self.assertTrue(all(c.doc.endswith(".md") and c.heading and c.text for c in chunks))

    def test_retrieves_the_right_article(self):
        r = Retriever()
        top, score = r.search("VPN keeps disconnecting every few minutes")[0]
        self.assertEqual(top.doc, "vpn.md")
        self.assertGreater(score, 0.3)

    def test_unrelated_question_returns_nothing(self):
        self.assertEqual(Retriever().search("how to bake bread"), [])

    def test_scores_are_sorted_and_k_is_respected(self):
        hits = Retriever().search("account password locked vpn", k=2)
        self.assertLessEqual(len(hits), 2)
        self.assertEqual([s for _, s in hits], sorted((s for _, s in hits), reverse=True))

    def test_prompt_contains_context_and_citation_rule(self):
        r = Retriever([Chunk("a.md", "Reset", "Reset the router.")])
        prompt = build_prompt("how reset router", r.search("reset router"))
        self.assertIn("Reset the router.", prompt)
        self.assertIn("ONLY the context", prompt)

    def test_answer_has_sources(self):
        out = answer("printer shows offline", Retriever())
        self.assertEqual(out["sources"][0]["doc"], "printer.md")

    def test_too_short_question_asks_for_detail_without_searching(self):
        out = answer("hi", Retriever())
        self.assertEqual(out["sources"], [])
        self.assertIn("describe the problem", out["answer"])
        self.assertEqual(Retriever().search("你好"), [])

    def test_short_but_real_questions_still_work(self):
        self.assertEqual(Retriever().search("VPN keeps dropping")[0][0].doc, "vpn.md")


if __name__ == "__main__":
    unittest.main()
