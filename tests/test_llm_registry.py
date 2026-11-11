import json
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ops_report import gcp, rag  # noqa: E402

try:
    from fastapi.testclient import TestClient
    from ops_report.api import create_app
except ImportError:
    TestClient = None


class Resp:
    def __init__(self, payload):
        self.payload = json.dumps(payload).encode()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self, *a):
        return self.payload


class SpecTests(unittest.TestCase):
    def test_parse_spec_keeps_colons_inside_model_names(self):
        self.assertEqual(rag.parse_llm_spec("ollama"), ("ollama", None))
        self.assertEqual(rag.parse_llm_spec("Ollama:qwen2.5:7b"), ("ollama", "qwen2.5:7b"))

    def test_make_llm_uses_the_requested_model(self):
        self.assertEqual(rag.make_llm("ollama:llama3.1:8b").model, "llama3.1:8b")
        self.assertEqual(rag.make_llm("vertex:gemini-3.8-flash").model, "gemini-3.8-flash")
        self.assertIsNone(rag.make_llm("none"))

    def test_unknown_provider_is_rejected_with_the_available_list(self):
        with self.assertRaises(ValueError) as ctx:
            rag.make_llm("nope:model")
        self.assertIn("ollama", str(ctx.exception))

    def test_list_ollama_models_skips_embedding_models(self):
        tags = {"models": [{"name": "qwen2.5:7b"}, {"name": "bge-m3:latest"}, {"name": "nomic-embed-text:latest"}]}
        with mock.patch("urllib.request.urlopen", return_value=Resp(tags)):
            self.assertEqual(rag.list_ollama_models(), ["qwen2.5:7b"])

    def test_list_ollama_models_is_empty_when_ollama_is_down(self):
        self.assertEqual(rag.list_ollama_models(host="http://127.0.0.1:1", timeout=1), [])


class VertexLocationTests(unittest.TestCase):
    def test_gemini_defaults_to_the_global_location_and_the_working_model(self):
        with mock.patch.dict(os.environ, {"VERTEX_LOCATION": "us-central1"}):
            os.environ.pop("VERTEX_LLM_LOCATION", None)
            os.environ.pop("VERTEX_LLM_MODEL", None)
            gemini = rag.VertexGemini()
            self.assertEqual((gemini.model, gemini.location), ("gemini-3.8-flash", "global"))
            self.assertEqual(rag.VertexEmbedder().location, "us-central1")  # embeddings stay regional

    def test_llm_location_can_be_overridden_separately(self):
        with mock.patch.dict(os.environ, {"VERTEX_LLM_LOCATION": "us-east5"}):
            self.assertEqual(rag.VertexGemini().location, "us-east5")


class VertexUrlTests(unittest.TestCase):
    def test_global_location_uses_the_bare_host(self):
        url = gcp.vertex_model_url("proj", "global", "gemini-x", "generateContent")
        self.assertTrue(url.startswith("https://aiplatform.googleapis.com/v1/projects/proj/locations/global/"))

    def test_regional_location_uses_a_prefixed_host(self):
        url = gcp.vertex_model_url("proj", "us-central1", "m", "predict")
        self.assertTrue(url.startswith("https://us-central1-aiplatform.googleapis.com/"))
        self.assertTrue(url.endswith("/publishers/google/models/m:predict"))


@unittest.skipIf(TestClient is None, "fastapi / httpx not installed")
class ApiRegistryTests(unittest.TestCase):
    def test_several_models_of_one_provider_are_listed_and_individually_selectable(self):
        class Fake:
            def __init__(self, model=None):
                self.model = model

            def generate(self, prompt):
                return f"answer from {self.model} [1]"

        with mock.patch.dict(os.environ, {"RAG_LLMS": "ollama:a:1b,ollama:b:2b"}), \
             mock.patch.dict(rag.LLM_PROVIDERS, {"ollama": Fake}):
            client = TestClient(create_app(n_tickets=10))
            ids = [m["id"] for m in client.get("/health").json()["answer_methods"]]
            first = client.post("/ask", json={"question": "printer offline", "method": "ollama:a:1b"}).json()
            second = client.post("/ask", json={"question": "printer offline", "method": "ollama:b:2b"}).json()
        self.assertEqual(ids, ["passage", "ollama:a:1b", "ollama:b:2b"])
        self.assertEqual(first["answer"], "answer from a:1b [1]")
        self.assertEqual(second["answer"], "answer from b:2b [1]")

    def test_a_new_provider_can_be_registered_without_touching_the_api(self):
        class Echo:
            def __init__(self, model=None):
                self.model = model or "echo-1"

            def generate(self, prompt):
                return "echo [1]"

        with mock.patch.dict(os.environ, {"RAG_LLMS": "echo"}), \
             mock.patch.dict(rag.LLM_PROVIDERS, {"echo": Echo}):
            client = TestClient(create_app(n_tickets=10))
            health = client.get("/health").json()["answer_methods"]
            body = client.post("/ask", json={"question": "printer offline", "method": "echo"}).json()
        self.assertEqual([m["id"] for m in health], ["passage", "echo"])
        self.assertEqual(health[1]["label"], "echo: echo-1")
        self.assertEqual(body["mode"], "generated")

    def test_ollama_wildcard_lists_every_local_model(self):
        with mock.patch.dict(os.environ, {"RAG_LLMS": "ollama:*"}), \
             mock.patch("ops_report.api.list_ollama_models", return_value=["qwen2.5:7b", "llama3.1:8b"]):
            client = TestClient(create_app(n_tickets=10))
            ids = [m["id"] for m in client.get("/health").json()["answer_methods"]]
        self.assertEqual(ids, ["passage", "ollama:qwen2.5:7b", "ollama:llama3.1:8b"])

    def test_long_real_world_model_ids_are_accepted_by_the_request_validation(self):
        class Fake:
            def __init__(self, model=None):
                self.model = model

            def generate(self, prompt):
                return "ok [1]"

        long_id = "vertex:gemini-3.8-flash"  # 23 characters: an earlier limit of 20 rejected it with HTTP 422
        with mock.patch.dict(os.environ, {"RAG_LLMS": long_id}), mock.patch.dict(rag.LLM_PROVIDERS, {"vertex": Fake}):
            client = TestClient(create_app(n_tickets=10))
            r = client.post("/ask", json={"question": "printer offline", "method": long_id})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["mode"], "generated")

    def test_unknown_provider_fails_at_startup(self):
        with mock.patch.dict(os.environ, {"RAG_LLMS": "nope"}):
            with self.assertRaises(ValueError):
                create_app(n_tickets=10)


if __name__ == "__main__":
    unittest.main()
