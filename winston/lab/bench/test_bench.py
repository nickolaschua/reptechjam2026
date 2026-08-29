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

    def test_context_block_renders_card_and_profile(self):
        # if the card or profile silently stopped rendering, every utterance would be
        # generated without its constraints and the run would still "succeed"
        from prompts import build_system_prompt
        p = build_system_prompt(self._product(), {"hard_constraints": ["foam"], "soft_preferences": ["ventilated"]},
                                {"preference_tags": ["comfort"], "summary": "buys clogs"}, "plain", [])
        for needle in ("foam", "ventilated", "comfort", "buys clogs", "Crocs Classic Clog", "unisex-adult"):
            self.assertIn(needle, p)

    def test_description_truncated_to_300(self):
        from prompts import build_system_prompt
        product = {**self._product(), "description": ["x" * 400 + "TAIL"]}
        p = build_system_prompt(product, {}, {}, "plain", [])
        self.assertNotIn("TAIL", p)
        self.assertIn("x" * 300, p)


def _fake_ix():
    """Four products, two buckets. Enough to exercise every covariate branch."""
    import math
    products = {
        "A1": {"title": "Asics E760Y-0143 Gel Tennis Shoe", "features": ["Rubber sole", "GEL cushioning system"],
               "details": {}, "description": "", "price": 80.0, "rating_number": 500, "categories": ["Clothing, Shoes & Jewelry", "Men", "Shoes"]},
        "A2": {"title": "Asics Gel Tennis Shoe Blue", "features": ["Rubber sole"],
               "details": {}, "description": "", "price": None, "rating_number": 3, "categories": ["Clothing, Shoes & Jewelry", "Men", "Shoes"]},
        "B1": {"title": "Sterling Silver 925 Pendant", "features": ["100% Cotton cord"],
               "details": {"Material": "silver"}, "description": "", "price": 12.0, "rating_number": 40, "categories": ["Clothing, Shoes & Jewelry", "Westlake"]},
        "B2": {"title": "Plain Hoodie", "features": [],
               "details": {}, "description": "", "price": 20.0, "rating_number": 10, "categories": ["Clothing, Shoes & Jewelry", "Women", "Clothing", "Hoodies"]},
    }
    text = {a: " ".join([p["title"], *p["features"], str(p["details"]), str(p["description"])]).lower() for a, p in products.items()}
    fields = {a: {"title": p["title"].lower(), "features": " ".join(p["features"]).lower()} for a, p in products.items()}
    df = {}
    for t in text.values():
        for tok in set(t.split()):
            df[tok] = df.get(tok, 0) + 1
    idf = {tok: math.log(4 / c) for tok, c in df.items()}
    bucket_of = {"A1": "men shoes", "A2": "men shoes", "B1": "watches watch bands", "B2": "clothing hoodies"}
    buckets = {}
    for a, b in bucket_of.items():
        buckets.setdefault(b, []).append(a)
    return SimpleNamespace(products=products, text=text, fields=fields, idf=idf, buckets=buckets, bucket_of=bucket_of,
                           categories={a: p["categories"] for a, p in products.items()})


class TestCovariates(unittest.TestCase):
    def test_model_code_accepts_real_codes_and_rejects_grades(self):
        from covariates import model_code
        self.assertEqual(model_code("Asics E760Y-0143 Gel"), "E760Y-0143")
        self.assertEqual(model_code("VICTONY WA1200 Extender"), "WA1200")
        self.assertIsNone(model_code("Sterling Silver 925 Pendant"))
        self.assertIsNone(model_code("316L Surgical Steel Ring"))
        self.assertIsNone(model_code("14K Gold Chain"))
        self.assertIsNone(model_code("Plain Hoodie"))

    def test_near_duplicates_by_title_jaccard_within_bucket(self):
        from covariates import near_duplicates
        dups = near_duplicates(_fake_ix())
        self.assertEqual(dups, {"A1", "A2"})          # B1 and B2 are alone in their buckets

    def test_covariates_for_fields(self):
        from covariates import covariates_for, near_duplicates
        ix = _fake_ix()
        c = covariates_for("B1", ix, near_duplicates(ix))
        self.assertTrue(c["promo_bucket"])             # Westlake path
        self.assertTrue(c["compat_eligible"])          # bucket is a watch-band bucket
        self.assertEqual(c["compat_anchor"], "watch")
        self.assertFalse(c["silent_on_material"])      # "cotton" (silver is not in MATERIAL_RE)
        self.assertFalse(c["has_model_code"])
        self.assertFalse(c["has_near_duplicate"])
        self.assertTrue(c["price_present"])
        self.assertEqual(c["bucket_size"], 1)
        self.assertEqual(c["category_depth"], 2)
        self.assertGreater(c["descriptiveness"], 0.0)
        self.assertTrue(c["jargon"] is None or 0.0 <= c["jargon"] <= 1.0)   # None only without wordfreq
        c2 = covariates_for("B2", ix, set())
        self.assertFalse(c2["compat_eligible"])
        self.assertIsNone(c2["compat_anchor"])
        self.assertIsNone(c2["department"])
        self.assertTrue(c2["silent_on_material"])
        self.assertEqual(c2["descriptiveness"], 0.0)   # no features at all
        self.assertFalse(c2["price_present"] is None)


if __name__ == "__main__":
    unittest.main()
