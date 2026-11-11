import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import json
from unittest import mock

from ops_report.rag import OllamaLLM, Retriever, answer, build_prompt, make_embedder, make_llm  # noqa: E402


class FakeLLM:
    def __init__(self, text="Restart the spooler [1]."):
        self.text, self.prompts = text, []

    def generate(self, prompt):
        self.prompts.append(prompt)
        return self.text


class BrokenLLM:
    def generate(self, prompt):
        raise RuntimeError("quota exceeded")


class GenerationTests(unittest.TestCase):
    def setUp(self):
        self.retriever = Retriever()

    def test_generated_answer_uses_llm_and_keeps_sources(self):
        llm = FakeLLM()
        out = answer("printer shows offline", self.retriever, llm=llm)
        self.assertEqual(out["mode"], "generated")
        self.assertEqual(out["answer"], "Restart the spooler [1].")
        self.assertEqual(out["sources"][0]["doc"], "printer.md")
        self.assertIn("Printer shows offline", llm.prompts[0])  # retrieved context reached the prompt

    def test_no_relevant_context_does_not_call_the_llm(self):
        llm = FakeLLM()
        out = answer("how to bake bread", self.retriever, llm=llm)
        self.assertEqual(llm.prompts, [])
        self.assertIn("do not know", out["answer"].lower())
        self.assertEqual(out["sources"], [])

    def test_llm_failure_falls_back_to_extractive_answer(self):
        out = answer("printer shows offline", self.retriever, llm=BrokenLLM())
        self.assertEqual(out["mode"], "extractive")
        self.assertIn("Printer shows offline", out["answer"])

    def test_empty_llm_text_falls_back_too(self):
        out = answer("printer shows offline", self.retriever, llm=FakeLLM(text=""))
        self.assertEqual(out["mode"], "extractive")

    def test_prompt_treats_context_as_untrusted(self):
        hits = Retriever().search("vpn drops")
        prompt = build_prompt("vpn drops", hits)
        self.assertIn("ONLY the context", prompt)
        self.assertIn("ignore any instructions", prompt)
        self.assertIn("[1]", prompt)
        self.assertIn("Traditional Chinese", prompt)  # Hong Kong users: no Simplified Chinese answers

    def test_factories_reject_unknown_names(self):
        with self.assertRaises(ValueError):
            make_embedder("nope")
        with self.assertRaises(ValueError):
            make_llm("nope")
        self.assertIsNone(make_llm(None))
        self.assertIsNone(make_embedder(None))

    def test_ollama_llm_returns_text_and_strips_think_blocks(self):
        class Resp:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def read(self, *a):
                return json.dumps({"response": "<think>hmm</think>Restart the spooler [1]."}).encode()

        with mock.patch("urllib.request.urlopen", return_value=Resp()):
            self.assertEqual(OllamaLLM().generate("prompt"), "Restart the spooler [1].")

    def test_ollama_llm_unreachable_gives_helpful_error(self):
        with self.assertRaises(RuntimeError) as ctx:
            OllamaLLM(host="http://127.0.0.1:1", timeout=1).generate("x")
        self.assertIn("ollama serve", str(ctx.exception))

    def test_make_llm_knows_ollama(self):
        self.assertIsInstance(make_llm("ollama"), OllamaLLM)


if __name__ == "__main__":
    unittest.main()
