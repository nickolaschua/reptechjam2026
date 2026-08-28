from __future__ import annotations

import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from nickolas.experiments.config import repo_root
from nickolas.experiments.harness import TurnState, _load_official, normalize, rank_metrics, replay_policy


class SuiteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.official = _load_official(repo_root())

    def test_intent_card_reconstruction_is_deterministic(self) -> None:
        product = {"title": "Blue cotton running shirt", "features": ["Machine Wash"], "details": {"Department": "Men"}, "description": [], "categories": ["Shirts"], "store": "X", "price": 20}
        first = self.official.intent_card(product)
        second = self.official.intent_card(product)
        self.assertEqual(first, second)
        self.assertEqual(first["hard_constraints"][:2], ["cotton", "color: blue"])

    def test_classifier_precedence_budget_before_material_and_color(self) -> None:
        self.assertEqual(self.official.classify_constraint("blue cotton under $20"), "budget")
        self.assertEqual(self.official.classify_constraint("blue cotton"), "material")

    def test_accumulated_other_state_discloses_in_order(self) -> None:
        sample = {"scenario_type": "browsing", "intent_card": {"hard_constraints": ["one", "two"], "soft_preferences": ["three", "four"]}, "behavior": {"scenario_type": "browsing"}}
        disclosed: set[str] = set()
        message, boundary = self.official.customer_reply(sample, "other", disclosed, False)
        self.assertEqual(disclosed, {"one", "two"})
        self.assertIn("one; two", message)
        message, _ = self.official.customer_reply(sample, "other", disclosed, boundary)
        self.assertEqual(disclosed, {"one", "two", "three", "four"})

    def test_normalized_exact_match_counting(self) -> None:
        docs = [normalize("Blue   cotton shirt"), normalize("blue wool shirt")]
        self.assertEqual(sum(normalize("BLUE cotton") in doc for doc in docs), 1)

    def test_target_rank_tie_break_and_missing(self) -> None:
        ids = np.asarray(["B", "A", "C"])
        scores = np.asarray([1.0, 1.0, 0.0])
        positive = np.flatnonzero(scores > 0)
        order = positive[np.lexsort((ids[positive], -scores[positive]))]
        self.assertEqual(list(ids[order]), ["A", "B"])
        self.assertNotIn(2, order)

    def test_metric_formulas(self) -> None:
        rows = [{"hit": True, "first_hit_turn": 2, "reciprocal_rank": .5}, {"hit": False, "first_hit_turn": None, "reciprocal_rank": 0.0}]
        metrics = rank_metrics(rows)
        self.assertEqual(metrics["hit_rate_at_10"], .5)
        self.assertEqual(metrics["mrr"], .25)
        self.assertEqual(metrics["mttc"], 6.5)
        self.assertEqual(metrics["technical_score"], .415)

    def test_early_termination_and_override_boundary(self) -> None:
        target_state = TurnState("s", "intent_override", "T", 1, "old", "cat", (), (), False, {})
        hit_state = TurnState("s", "intent_override", "T", 2, "new", "cat", ("new",), ("new",), True, {})
        fake = SimpleNamespace(id_to_idx={"T": 0}, trace_by_session=lambda: {"s": [target_state, hit_state]})
        def ranker(state): return np.asarray([0]), np.asarray([1.0])
        sessions, metrics = replay_policy(fake, ranker, 10)
        self.assertEqual(sessions[0]["first_hit_turn"], 2)
        boundary = {"scenario_type": "boundary", "intent_card": {"hard_constraints": ["x"], "soft_preferences": []}, "behavior": {}}
        reply, used = self.official.customer_reply(boundary, "other", set(), False)
        self.assertTrue(used)
        self.assertIn("don't have a preference", reply)


if __name__ == "__main__":
    unittest.main()
