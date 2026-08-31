"""The 40 public-set sessions our binary intent scheme has no slot for.

intent_override is 30/200 and boundary is 10/200 - 20% of the real evaluation, and
every override session is graded "hard". Their messages are fixed evaluator
templates, so the transitions can be asserted exactly: no labelling, no LLM, no
gold set. Messages are built by calling the evaluator's OWN generators rather than
copying the strings, so these tests break if the evaluator changes shape.

    python3 -m unittest lab.test_intent_transitions -v
"""
from __future__ import annotations

import random
import sys
import unittest
from pathlib import Path

LAB = Path(__file__).resolve().parent
WINSTON = LAB.parent
REPO = WINSTON.parents[2]
KIT = REPO / "techjam-conversational-search"
for p in (WINSTON / "experiments", WINSTON, LAB, KIT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from evaluator.local_evaluator import (behavior_for, customer_reply,  # noqa: E402
                                       initial_message, intent_card)
from memory.fast_memory import update_state  # noqa: E402
from memory.types import FastMemoryState  # noqa: E402

# A product whose text carries a material and a colour - the two things the
# evaluator promotes to hard_constraints (local_evaluator.intent_card).
PRODUCT = {
    "title": "Classic Crew Neck Tee",
    "features": ["breathable everyday fit", "machine washable"],
    "details": {"Department": "womens"},
    "description": "Made from soft cotton in black.",
    "price": 24.99,
}
CATEGORY = "shirts t-shirts"


def sample_for(scenario: str) -> dict:
    card = intent_card(PRODUCT)
    return {"scenario_type": scenario, "intent_card": card,
            "behavior": behavior_for(scenario, card, random.Random(0))}


def fresh() -> FastMemoryState:
    return FastMemoryState(session_id="s", user_id="u", sequence_index=0)


def drive(messages: list[str]) -> FastMemoryState:
    state = fresh()
    for turn, msg in enumerate(messages, 1):
        state = update_state(state, msg, turn)
    return state


class TestTemplateIntent(unittest.TestCase):
    """The two scenarios we DO handle - these must keep working."""

    def test_buying_opener_is_buying(self):
        s = sample_for("buying")
        msg = initial_message(s, CATEGORY, set())
        self.assertIn("A key requirement is:", msg)
        self.assertEqual(drive([msg]).intent, "buying")

    def test_browsing_opener_is_browsing(self):
        s = sample_for("browsing")
        msg = initial_message(s, CATEGORY, set())
        self.assertIn("still exploring", msg)
        self.assertEqual(drive([msg]).intent, "browsing")


class TestIntentOverride(unittest.TestCase):
    """30/200 sessions, all graded hard. The user opens with a SOFT preference and
    only states a hard requirement at turn 3 or 4."""

    def setUp(self):
        self.sample = sample_for("intent_override")
        self.opener = initial_message(self.sample, CATEGORY, set())
        self.override = self.sample["behavior"]["override"]["message"]

    def test_opener_discloses_only_a_soft_preference(self):
        self.assertNotIn("A key requirement is:", self.opener)
        self.assertNotIn("still exploring", self.opener)

    def test_opener_is_browsing(self):
        """Spec: 'has not yet specified enough information to narrow the search'.
        A soft preference is not a hard constraint."""
        self.assertEqual(drive([self.opener]).intent, "browsing")

    def test_override_turn_flips_to_buying(self):
        """Spec: 'It is a spectrum, not a permanent label.' The override message
        states a real hard constraint, so intent must move."""
        before = drive([self.opener, "Those options are not quite right yet."])
        self.assertEqual(before.intent, "browsing")        # nothing hard stated yet
        after = drive([self.opener, "Those options are not quite right yet.", self.override])
        self.assertEqual(after.intent, "buying")           # a requirement is now on the table
        self.assertGreater(after.intent_source_turn, 1)    # it MOVED, not locked at turn 1

    def test_override_still_revokes_the_seed(self):
        """Whatever we do to intent must not break the constraint handling that
        already works."""
        state = drive([self.opener, "Those options are not quite right yet.", self.override])
        self.assertIsNone(state.override_seed)


class TestBoundary(unittest.TestCase):
    """10/200 sessions. The user explicitly declines to state a preference."""

    def setUp(self):
        self.sample = sample_for("boundary")
        self.opener = initial_message(self.sample, CATEGORY, set())

    def test_boundary_opener_is_browsing(self):
        self.assertIn("still exploring", self.opener)
        self.assertEqual(drive([self.opener]).intent, "browsing")

    def test_declining_never_becomes_a_buying_signal(self):
        reply, used = customer_reply(self.sample, "material", set(), False)
        self.assertTrue(used)
        self.assertIn("do not have a preference", reply.replace("don't", "do not"))
        self.assertEqual(drive([self.opener, reply]).intent, "browsing")


if __name__ == "__main__":
    unittest.main()
