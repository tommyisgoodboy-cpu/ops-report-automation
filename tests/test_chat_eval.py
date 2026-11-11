import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ops_report.chat_eval import CASES, evaluate  # noqa: E402
from ops_report.rag import Retriever  # noqa: E402


class ChatEvalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = json.loads(CASES.read_text(encoding="utf-8"))
        cls.retriever = Retriever()

    def test_every_case_has_a_known_action_and_existing_passages(self):
        headings = {c.heading for c in self.retriever.chunks}
        for case in self.cases:
            self.assertIn(case["action"], ("chat", "search", "any"), case["id"])
            if case["action"] == "search":
                self.assertIn(case["expect"], headings, case["id"])
            for role, _ in case["history"]:
                self.assertIn(role, ("user", "assistant"))

    def test_small_talk_and_unrelated_cases_pass_without_a_model(self):
        # These do not depend on which retriever is used, so they can be a hard floor in CI.
        result = evaluate(self.retriever, None, self.cases)
        self.assertEqual(result["chat"], 1.0)
        self.assertEqual(result["unrelated"], 1.0)

    def test_the_set_covers_follow_ups_topic_switches_and_small_talk(self):
        ids = {c["id"] for c in self.cases}
        for needed in ("why-follow-up", "topic-switch", "greeting-english", "thanks-cantonese", "unrelated"):
            self.assertIn(needed, ids)


if __name__ == "__main__":
    unittest.main()
