import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ops_report.data import CATEGORIES  # noqa: E402
from ops_report.train import CentroidClassifier, NaiveBayes, features  # noqa: E402
from ops_report.train_eval import PARAPHRASES, CachingEmbedder, evaluate, training_set  # noqa: E402


class FakeEmbedder:
    """Two dimensions: 'print' words point one way, 'network' words the other."""

    def __init__(self):
        self.calls = 0

    def vec(self, text):
        t = text.lower()
        return [1.0, 0.0] if any(w in t for w in ("print", "toner", "paper")) else [0.0, 1.0]

    def embed(self, texts):
        self.calls += len(texts)
        return [self.vec(t) for t in texts]


class FeatureTests(unittest.TestCase):
    def test_words_lose_stop_words_and_chinese_becomes_character_pairs(self):
        self.assertEqual(features("The printer is jammed"), ["printer", "jammed"])
        self.assertEqual(features("打印機連唔到"), ["打印", "印機", "機連", "連唔", "唔到"])
        self.assertEqual(features("VPN 斷線"), ["vpn", "斷線"])


class ModelTests(unittest.TestCase):
    def test_naive_bayes_learns_obvious_words_and_abstains_on_unknown_ones(self):
        model = NaiveBayes().train(["printer jammed", "printer toner low", "vpn drops", "wifi slow network"],
                                   ["Printer", "Printer", "Network", "Network"])
        self.assertEqual(model.predict("toner jammed"), "Printer")
        self.assertEqual(model.predict("slow wifi"), "Network")
        self.assertEqual(model.predict("completely unrelated words"), "Other")

    def test_centroid_classifier_picks_the_nearest_category(self):
        model = CentroidClassifier(FakeEmbedder()).train(["paper jam", "toner low", "vpn drops", "wifi slow"],
                                                         ["Printer", "Printer", "Network", "Network"])
        self.assertEqual(model.predict("printer prints blank pages"), "Printer")
        self.assertEqual(model.predict("cannot connect"), "Network")

    def test_the_caching_embedder_embeds_each_sentence_once(self):
        inner = FakeEmbedder()
        cached = CachingEmbedder(inner)
        cached.embed(["a", "b", "a"]), cached.embed(["a", "c"])
        self.assertEqual(inner.calls, 3)


class EvaluationHonestyTests(unittest.TestCase):
    def test_no_paraphrase_is_a_training_sentence(self):
        trained = {t.lower() for t in training_set()[0]}
        items = json.loads(PARAPHRASES.read_text(encoding="utf-8"))
        self.assertTrue(items)
        for item in items:
            self.assertNotIn(item["q"].lower(), trained, item["q"])
            self.assertIn(item["category"], CATEGORIES)

    def test_the_paraphrase_set_covers_every_category_and_includes_cantonese(self):
        items = json.loads(PARAPHRASES.read_text(encoding="utf-8"))
        self.assertEqual({i["category"] for i in items}, set(CATEGORIES))
        self.assertTrue(any(any("一" <= ch <= "鿿" for ch in i["q"]) for i in items))

    def test_leave_one_out_scores_below_in_sample_for_a_model_that_memorises(self):
        result = evaluate(None)["naive bayes"]
        self.assertEqual(result["in_sample"], 1.0)
        self.assertLess(result["leave_one_out"], 0.5)  # held-out sentences share almost no words with the rest

    def test_the_keyword_baseline_is_not_trained_and_reports_the_same_view_twice(self):
        baseline = evaluate(None)["keyword rules (no training)"]
        self.assertEqual(baseline["in_sample"], baseline["leave_one_out"])


if __name__ == "__main__":
    unittest.main()
