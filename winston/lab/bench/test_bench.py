import random
import unittest
from types import SimpleNamespace


class TestPrompts(unittest.TestCase):
    def test_content_words_strips_stopwords_and_dedupes(self):
        from prompts import content_words
        self.assertEqual(content_words("The Leather Boots, leather!"), ["leather", "boots"])

    def test_forbidden_list_comes_from_title_and_features_capped(self):
        from prompts import forbidden_list
        product = {"title": "Merrell Vapor Glove Trail Running Shoe",
                   "features": ["100% Textile", "Rubber sole", "Barefoot-style trail runner"]}
        words = forbidden_list(product, cap=5)
        self.assertEqual(len(words), 5)
        self.assertIn("merrell", words)
        self.assertNotIn("100", words)          # digits are not vocabulary

    def test_relation_matches_department(self):
        from prompts import relation_for
        rng = random.Random(0)
        self.assertIn(relation_for("womens", rng), {"wife", "mum", "sister", "daughter"})
        self.assertIn(relation_for("mens", rng), {"dad", "husband", "brother", "son"})
        self.assertEqual(relation_for(None, rng), "friend")
        self.assertEqual(relation_for("unisex-adult", rng), "friend")


if __name__ == "__main__":
    unittest.main()
