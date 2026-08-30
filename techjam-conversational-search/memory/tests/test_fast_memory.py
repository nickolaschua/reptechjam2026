from __future__ import annotations

import unittest

from memory import (
    ConstraintKind,
    FastMemoryState,
    FastMemoryUpdate,
    TypedConstraint,
    override_intent,
    update_state,
)


class StubSemanticParser:
    def __init__(self, result: FastMemoryUpdate | None) -> None:
        self.result = result
        self.calls: list[tuple[str, int]] = []

    def parse(self, message: str, turn: int) -> FastMemoryUpdate | None:
        self.calls.append((message, turn))
        return self.result


class FastMemoryTests(unittest.TestCase):
    def state(self) -> FastMemoryState:
        return FastMemoryState("session", "user", 0)

    def test_official_templates_preserve_category_hard_soft_and_sources(self) -> None:
        state = self.state()
        update_state(state, "I'm looking for shoes. A key requirement is: under $60.", 1)
        update_state(state, "For that, what matters is: red; leather.", 2)
        self.assertEqual(state.category, "shoes")
        self.assertEqual(state.category_source_turn, 1)
        self.assertEqual(state.intent, "buying")
        self.assertEqual(state.intent_source_turn, 1)
        self.assertEqual(state.hard_constraints[0].kind, ConstraintKind.BUDGET)
        self.assertEqual(state.hard_constraints[0].source_turn, 1)
        self.assertEqual(
            [item.kind for item in state.soft_preferences],
            [ConstraintKind.COLOR, ConstraintKind.MATERIAL],
        )
        self.assertEqual([item.source_turn for item in state.soft_preferences], [2, 2])

    def test_replacement_creates_typed_negative(self) -> None:
        state = self.state()
        update_state(state, "I'm looking for shoes. bright red", 1)
        update_state(
            state,
            "Actually, ignore my earlier preference. What I need is: deep blue.",
            2,
        )
        self.assertEqual(state.constraint_values, ("deep blue",))
        self.assertEqual(state.negatives[0].value, "bright red")
        self.assertEqual(state.negatives[0].kind, ConstraintKind.COLOR)
        self.assertTrue(state.negatives[0].negated)
        self.assertEqual(state.negatives[0].source_turn, 2)

    def test_free_form_slots_and_negative(self) -> None:
        state = self.state()
        update_state(state, "I'm looking for jackets, but I'm still exploring.", 1)
        update_state(state, "I need size wide", 2)
        update_state(state, "I don't want wool", 3)
        self.assertEqual(state.hard_constraints[0].kind, ConstraintKind.SIZE)
        self.assertEqual(state.negatives[0].kind, ConstraintKind.MATERIAL)
        self.assertEqual(state.facts(ConstraintKind.SIZE), (state.hard_constraints[0],))

    def test_topic_override_starts_epoch_and_keeps_only_portable_budget(self) -> None:
        state = self.state()
        update_state(state, "I'm looking for shoes. red hiking", 1)
        update_state(state, "I need under $80", 2)
        result = update_state(state, "Now I'm looking for dresses instead", 3)
        self.assertEqual(result.category, "dresses")
        self.assertEqual(result.category_source_turn, 3)
        self.assertEqual(result.intent_epoch, 1)
        self.assertTrue(result.topic_override)
        self.assertEqual([item.value for item in result.hard_constraints], ["under $80"])
        self.assertEqual(result.soft_preferences, [])

    def test_explicit_override_tracks_intent_source_turn(self) -> None:
        state = self.state()
        update_state(state, "I'm looking for shoes, but I'm still exploring.", 1)
        result = override_intent(state, "watch", 1, intent="buying")
        self.assertEqual(result.category, "watch")
        self.assertEqual(result.intent, "buying")
        self.assertEqual(result.intent_source_turn, 1)
        self.assertTrue(result.topic_override)

    def test_semantic_hook_is_authoritative_and_none_falls_back(self) -> None:
        semantic = StubSemanticParser(FastMemoryUpdate(
            category="boots",
            intent="buying",
            hard_constraints=(TypedConstraint("waterproof", hard=True),),
            confidence=0.95,
        ))
        state = update_state(self.state(), "unstructured input", 1, semantic)
        self.assertEqual(semantic.calls, [("unstructured input", 1)])
        self.assertEqual(state.category, "boots")
        self.assertEqual(state.hard_constraints[0].value, "waterproof")
        self.assertEqual(state.hard_constraints[0].source_turn, 1)
        self.assertNotIn("unstructured input", state.constraint_values)

        fallback = update_state(
            FastMemoryState("fallback", "user", 0),
            "I'm looking for hats, but I'm still exploring.",
            1,
            StubSemanticParser(None),
        )
        self.assertEqual(fallback.category, "hats")

    def test_fast_memory_has_no_embedding(self) -> None:
        state = self.state()
        self.assertFalse(hasattr(state, "embedding"))
        self.assertFalse(hasattr(state, "vector"))


if __name__ == "__main__":
    unittest.main()
