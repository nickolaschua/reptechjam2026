"""Ollama-free checks for the bolt-on. Run: python3 -m unittest test_bolt_on -v"""
import json
import sys
import unittest
from pathlib import Path

LAB = Path(__file__).resolve().parent
sys.path.insert(0, str(LAB))
import bolt_on  # noqa: E402

KIT = bolt_on.KIT


def slot(attribute, value, declined=False, negated=False):
    return {"attribute": attribute, "value": value, "declined": declined, "negated": negated}


def raw(category="", slots=(), price_max=None, department=None):
    return {"category_phrase": category, "department": department, "slots": list(slots),
            "price_max": price_max, "price_min": None, "quality_prior": "none", "exploring": False}


class TestTemplatePredicate(unittest.TestCase):
    def test_every_evaluator_form_is_template(self):
        for m in ("I'm looking for Athletic Tennis & Racquet Sports, but I'm still exploring.",
                  "I'm looking for Charms & Charm Bracelets Charms. A key requirement is: leather.",
                  "I'm looking for Charms & Charm Bracelets Charms. Imported.",
                  "I’m looking for Athletic Tennis & Racquet Sports, but I’m still exploring.",
                  "For that, what matters is: cotton; color: black.",
                  "Actually, ignore my earlier preference. What I need is: waterproof.",
                  "Actually, please ignore my earlier preference.",
                  "I don't have a preference for material; please use your judgment.",
                  "I don't have an additional preference for color.",
                  "Those options are not quite right yet. Ask me about one specific attribute."):
            self.assertTrue(bolt_on.is_template(m), m)

    def test_all_200_public_initial_messages_are_template(self):
        from evaluator.local_evaluator import catalog_index, coarse_category, intent_card
        _, cats, products = catalog_index(KIT / "data" / "catalog.jsonl")
        n = 0
        for line in (KIT / "data" / "public_set.jsonl").open():
            s = json.loads(line)
            asin = s["ground_truth"]["parent_asin"]
            cat = coarse_category(cats[asin])
            card = intent_card(products[asin])
            for m in (f"I'm looking for {cat}, but I'm still exploring.",
                      f"I'm looking for {cat}. A key requirement is: {card['hard_constraints'][0]}.",
                      f"I'm looking for {cat}. {card['soft_preferences'][-1]}"):
                self.assertTrue(bolt_on.is_template(m), m)
                n += 1
        self.assertEqual(n, 600)

    def test_generated_cases_fall_through(self):
        cases = LAB / "bench" / "cases.jsonl"
        if not cases.exists():
            self.skipTest("no cases.jsonl yet")
        rows = [json.loads(l) for l in cases.open() if l.strip()]
        caught = [r["utterance"][:70] for r in rows if bolt_on.is_template(r["utterance"])]
        self.assertEqual(caught, [], f"{len(caught)}/{len(rows)} generated cases look like templates")

    def test_all_30_probes_fall_through(self):
        for c in json.loads((bolt_on.WINSTON / "probe_gold.json").read_text()):
            self.assertFalse(bolt_on.is_template(c["utterance"]), c["utterance"][:60])


class TestDerivedLabels(unittest.TestCase):
    def test_model_code(self):
        self.assertEqual(bolt_on.model_code("i want the bm8242-08e in black"), "BM8242-08E")
        self.assertIsNone(bolt_on.model_code("925 sterling silver, upf 50"))
        self.assertIsNone(bolt_on.model_code("under 30 dollars"))

    def test_intent_is_specificity(self):
        self.assertEqual(bolt_on.intent_of(raw("shoe", [slot("use_case", "tennis")]), "shoe for tennis"), "browsing")
        self.assertEqual(bolt_on.intent_of(raw("tees", price_max=30), "tees under 30"), "buying")
        self.assertEqual(bolt_on.intent_of(raw("boots", [slot("material", "leather")]), "leather boots"), "buying")
        self.assertEqual(bolt_on.intent_of(raw("boots", [slot("material", "leather")]),
                                           "leather boots maybe, just browsing"), "browsing")
        self.assertEqual(bolt_on.intent_of(raw("watch"), "the WA1200 watch"), "buying")

    def test_message_type(self):
        self.assertEqual(bolt_on.message_type_of(raw("watch"), "the WA1200 please"), "exact")
        self.assertEqual(bolt_on.message_type_of(raw("band"), "i own a watch and need a band"), "compatibility")
        self.assertEqual(bolt_on.message_type_of(raw(""), "my feet hurt"), "symptom")
        self.assertEqual(bolt_on.message_type_of(raw("jeans"), "jeans"), "product_type")
        self.assertEqual(bolt_on.message_type_of(raw("shoe", [slot("use_case", "tennis")]), "x"), "use_case")
        self.assertEqual(bolt_on.message_type_of(raw("shoe", [slot("color", "red")]), "x"), "feature")


class TestMapping(unittest.TestCase):
    MSG = "need plain tees for my husband, he wears them hiking. under 30 dollars. he hates plastic"
    PARSE = raw("plain tees", [slot("style", "plain"), slot("use_case", "hiking"),
                               slot("material", "not plastic", negated=True),
                               slot("budget", "under 30 dollars each"),
                               slot("material", "cotton")], price_max=30, department="mens")

    def test_to_update_shapes(self):
        u = bolt_on.to_update(self.PARSE, self.MSG, turn=2, candidates=("t-shirts",), confidence=0.6)
        self.assertIsInstance(u, bolt_on.FastMemoryUpdate)
        self.assertEqual(u.category, "plain tees")
        self.assertEqual(u.intent, "buying")
        self.assertEqual(u.department, "mens")
        self.assertEqual([(c.value, c.kind.value) for c in u.hard_constraints],
                         [("cotton", "material"), ("under $30", "budget")])
        self.assertEqual([(c.value, c.kind.value) for c in u.soft_preferences],
                         [("plain", "style"), ("hiking", "use_case")])
        self.assertEqual([(c.value, c.negated) for c in u.negatives], [("plastic", True)])
        self.assertTrue(all(c.source_turn == 2 for c in u.hard_constraints + u.soft_preferences + u.negatives))
        self.assertEqual(u.category_candidates, ("t-shirts",))
        self.assertEqual(u.confidence, 0.6)
        self.assertEqual(u.message_type, "feature")
        json.loads(u.to_json())

    def test_declined_and_junk_never_become_constraints(self):
        p = raw("cap", [slot("color", "", declined=True), slot("size", "none")])
        u = bolt_on.to_update(p, "cap", 1)
        self.assertEqual(u.hard_constraints + u.soft_preferences + u.negatives, ())


class TestNegationSupport(unittest.TestCase):
    def test_cue_must_precede_value_closely(self):
        ns = bolt_on.negation_supported
        self.assertTrue(ns("he hates anything that feels like plastic against his skin", "plastic"))
        self.assertTrue(ns("not real gold obviously, plated is fine", "real gold"))
        self.assertTrue(ns("i don't want anything too thick", "too thick"))
        self.assertTrue(ns("nothing with a bunch of logos", "logos"))
        self.assertFalse(ns("i really like brown leather and something comfortable", "leather"))
        self.assertFalse(ns("a decent quality leather that's not too heavy", "leather"))

    def test_clean_parse_drops_unsupported_negation(self):
        p = raw("bag", [slot("material", "leather", negated=True), slot("feature", "too heavy", negated=True)])
        got = bolt_on.clean_parse(p, "a decent quality leather that's not too heavy")
        self.assertEqual([(s["value"], s["negated"]) for s in got["slots"]],
                         [("leather", False), ("too heavy", True)])


class TestContradictions(unittest.TestCase):
    PARSE = raw("tees", [slot("material", "cotton"), slot("brand", "asics"),
                         slot("material", "not plastic", negated=True), slot("size", "xl")],
                price_max=30, department="mens")

    def check(self, product, text):
        return bolt_on.contradictions(self.PARSE, product, text)

    def test_silent_product_is_never_a_contradiction(self):
        self.assertEqual(self.check({"price": None, "store": None, "details": {}}, "plain tee"), [])

    def test_each_kind_of_contradiction(self):
        product = {"price": 45.0, "store": "Nike", "details": {"Department": "womens"}}
        reasons = self.check(product, "polyester tee with a plastic feel")
        self.assertEqual(len(reasons), 5, reasons)          # price, dept, brand, material, negated

    def test_match_and_slack(self):
        product = {"price": 32.5, "store": "ASICS Store", "details": {"Department": "mens"}}
        self.assertEqual(self.check(product, "100% cotton tee"), [])          # 32.5 <= 30 * 1.10
        self.assertEqual(self.check({**product, "price": 34.0}, "cotton tee"), ["price 34.0 > max 30"])

    def test_unisex_and_mixed_material_do_not_contradict(self):
        product = {"price": None, "store": None, "details": {"Department": "unisex-adult"}}
        self.assertEqual(self.check(product, "cotton polyester blend"), [])  # cotton present -> match


class TestParserAndState(unittest.TestCase):
    def test_none_on_template_and_update_state_integration(self):
        from memory.fast_memory import update_state
        from memory.types import FastMemoryState
        calls = []
        bp = bolt_on.BoltOnParser(parse_fn=lambda m: (calls.append(m), TestMapping.PARSE)[1], resolver=False)
        self.assertIsNone(bp.parse("I'm looking for Athletic Tennis & Racquet Sports, but I'm still exploring.", 1))
        self.assertEqual(calls, [])
        state = FastMemoryState(session_id="s", user_id="u", sequence_index=0)
        update_state(state, "I'm looking for Athletic Tennis & Racquet Sports, but I'm still exploring.", 1, bp)
        self.assertEqual((state.category, state.intent), ("Athletic Tennis & Racquet Sports", "browsing"))  # deterministic path
        update_state(state, TestMapping.MSG, 2, bp)
        self.assertEqual(calls, [TestMapping.MSG])
        self.assertEqual(state.category, "plain tees")       # semantic update sets category (fast_memory.apply_semantic)
        self.assertIn("cotton", state.constraint_values)
        self.assertIn("under $30", state.constraint_values)
        self.assertIn("plain", state.constraint_values)
        self.assertEqual([c.value for c in state.negatives], ["plastic"])
        self.assertEqual(state.intent, "buying")


if __name__ == "__main__":
    unittest.main()
