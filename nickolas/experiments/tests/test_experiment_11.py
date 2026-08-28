from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from nickolas.experiments.experiment_11_candidate_agent import CleanFTSAgent


class Experiment11CandidateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.catalog = Path(self.temporary.name) / "catalog.jsonl"
        products = [
            {"parent_asin": "A", "title": "Blue cotton shirt", "categories": ["Shirts"], "features": ["soft"], "details": {}, "store": "One", "description": "daily shirt", "rating_number": 100},
            {"parent_asin": "B", "title": "Red wool shirt", "categories": ["Shirts"], "features": ["warm"], "details": {}, "store": "Two", "description": "winter shirt", "rating_number": 50},
            {"parent_asin": "C", "title": "Green linen shirt", "categories": ["Shirts"], "features": ["light"], "details": {}, "store": "Three", "description": "summer shirt", "rating_number": 10},
        ]
        self.catalog.write_text("\n".join(json.dumps(product) for product in products) + "\n", encoding="utf-8")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_case_insensitive_parser_and_correct_override_removal(self) -> None:
        agent = CleanFTSAgent(self.catalog)
        agent.reset("session", {"private": "ignored"})
        agent.respond("session", "i'm looking for Shirts. Blue cotton.", 1, 1)
        agent.respond("session", "FOR THAT, WHAT MATTERS IS: warm.", 2, 1)
        agent.respond(
            "session",
            "Actually, ignore my earlier preference. What I need is: Red wool.",
            3,
            1,
        )
        state = agent.sessions["session"]
        self.assertEqual(state["category"], "Shirts")
        self.assertNotIn("Blue cotton", state["constraints"])
        self.assertIn("warm", state["constraints"])
        self.assertIn("Red wool", state["constraints"])

        agent.reset("curly", {})
        agent.respond("curly", "I’m looking for Shirts, but I’m still exploring.", 1, 1)
        self.assertEqual(agent.sessions["curly"]["category"], "Shirts")

    def test_query_pagination_advances_repeats_but_resets_for_new_query(self) -> None:
        agent = CleanFTSAgent(self.catalog, pagination_mode="query")
        agent.reset("session", {})
        first = agent.respond("session", "I'm looking for Shirts, but I'm still exploring.", 1, 1)
        second = agent.respond("session", "I don't have an additional preference for feature.", 2, 1)
        third = agent.respond("session", "For that, what matters is: soft.", 3, 1)
        first_id = first["recommendations"][0]["parent_asin"]
        second_id = second["recommendations"][0]["parent_asin"]
        third_id = third["recommendations"][0]["parent_asin"]
        self.assertNotEqual(first_id, second_id)
        self.assertEqual(first_id, third_id)

    def test_global_pagination_does_not_repeat_after_query_change(self) -> None:
        agent = CleanFTSAgent(self.catalog, pagination_mode="global")
        agent.reset("session", {})
        first = agent.respond("session", "I'm looking for Shirts, but I'm still exploring.", 1, 1)
        second = agent.respond("session", "For that, what matters is: soft.", 2, 1)
        self.assertNotEqual(
            first["recommendations"][0]["parent_asin"],
            second["recommendations"][0]["parent_asin"],
        )

    def test_configuration_validation(self) -> None:
        with self.assertRaises(ValueError):
            CleanFTSAgent(self.catalog, question_policy="oracle")
        with self.assertRaises(ValueError):
            CleanFTSAgent(self.catalog, pagination_mode="unsafe")


if __name__ == "__main__":
    unittest.main()
