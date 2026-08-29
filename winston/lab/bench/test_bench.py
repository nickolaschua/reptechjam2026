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
        self.assertNotIn("100", forbidden_list(product, cap=40))   # pure digits are not vocabulary

    def test_content_words_keeps_alphanumeric_codes(self):
        from prompts import content_words
        self.assertEqual(content_words("BM8242-08E in black"), ["bm8242", "08e", "black"])

    def test_forbidden_list_handles_missing_fields(self):
        from prompts import forbidden_list
        self.assertEqual(forbidden_list({"title": None, "features": None}), [])

    def test_relation_matches_department(self):
        from prompts import relation_for
        rng = random.Random(0)
        self.assertIn(relation_for("womens", rng), {"wife", "mum", "sister", "daughter"})
        self.assertIn(relation_for("mens", rng), {"dad", "husband", "brother", "son"})
        self.assertEqual(relation_for(None, rng), "friend")
        self.assertEqual(relation_for("unisex-adult", rng), "friend")

    def _product(self):
        return {"title": "Crocs Classic Clog", "features": ["Croslite foam", "Ventilation ports"],
                "details": {"Department": "unisex-adult"}, "description": ["Iconic clog."]}

    def test_lay_prompt_contains_forbidden_words_and_no_rule3(self):
        from prompts import build_system_prompt
        p = build_system_prompt(self._product(), {"hard_constraints": ["foam"], "soft_preferences": []},
                                {"preference_tags": ["comfort"]}, "lay", [])
        self.assertIn("croslite", p)
        self.assertIn("You must not use any of these words", p)
        self.assertNotIn("Drop descriptive hints", p)

    def test_exact_requires_code_and_uses_it(self):
        from prompts import build_system_prompt
        with self.assertRaises(ValueError):
            build_system_prompt(self._product(), {}, {}, "exact", [])
        p = build_system_prompt(self._product(), {}, {}, "exact", [], code="WA1200")
        self.assertIn("WA1200", p)

    def test_compatibility_requires_anchor_and_uses_it(self):
        from prompts import build_system_prompt
        with self.assertRaises(ValueError):
            build_system_prompt(self._product(), {}, {}, "compatibility", [])
        p = build_system_prompt(self._product(), {}, {}, "compatibility", [], anchor="watch")
        self.assertIn("already own a watch", p)

    def test_modifiers_append_relation_fills_and_format_noise_is_exact_only(self):
        from prompts import build_system_prompt
        p = build_system_prompt(self._product(), {}, {}, "plain", ["negation", "for_other"], relation="dad")
        self.assertIn("do NOT want", p)
        self.assertIn("for your dad", p)
        with self.assertRaises(ValueError):
            build_system_prompt(self._product(), {}, {}, "plain", ["format_noise"])
        p = build_system_prompt(self._product(), {}, {}, "exact", ["format_noise"], code="WA1200")
        self.assertIn("different spacing", p)


if __name__ == "__main__":
    unittest.main()
