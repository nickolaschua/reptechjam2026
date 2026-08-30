from __future__ import annotations

import unittest

import numpy as np

from nickolas.memory.qlmp import (
    BaselineConfig,
    BaselineMode,
    MemoryItem,
    MemoryPolarity,
    MemorySource,
    bound_query_shift,
    build_cosine_memory_baseline,
    build_naive_memory_baseline,
    is_scope_compatible,
    normalize,
)


Q = np.asarray([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])


def memory(
    memory_id: str,
    embedding: np.ndarray,
    *,
    scope: str | None = "footwear",
    polarity: MemoryPolarity = MemoryPolarity.POSITIVE,
) -> MemoryItem:
    return MemoryItem(
        id=memory_id,
        text=memory_id,
        embedding=embedding,
        source=MemorySource.EXPLICIT_PREFERENCE,
        polarity=polarity,
        scope=scope,
        confidence=0.75,
    )


def cosine_memory(memory_id: str, cosine: float, axis: int) -> MemoryItem:
    vector = np.zeros(8)
    vector[0] = cosine
    vector[axis] = np.sqrt(1.0 - cosine**2)
    return memory(memory_id, vector)


class ScopeCompatibilityTests(unittest.TestCase):
    def test_same_incompatible_and_none_scope_policy(self) -> None:
        self.assertTrue(is_scope_compatible("footwear", "footwear"))
        self.assertFalse(is_scope_compatible("footwear", "electronics"))
        self.assertTrue(is_scope_compatible("footwear", None))
        self.assertFalse(
            is_scope_compatible(
                "footwear", None, allow_unscoped_memory=False
            )
        )
        self.assertTrue(is_scope_compatible(None, "electronics"))

    def test_matching_is_exact_and_invalid_values_fail(self) -> None:
        self.assertFalse(is_scope_compatible("footwear", "Footwear"))
        for query_scope, memory_scope in ((" ", None), (None, ""), (3, None)):
            with self.subTest(query_scope=query_scope, memory_scope=memory_scope):
                with self.assertRaises(ValueError):
                    is_scope_compatible(query_scope, memory_scope)  # type: ignore[arg-type]


class NaiveBaselineTests(unittest.TestCase):
    def test_all_eligible_memories_are_uniformly_aggregated(self) -> None:
        e2, e3, e5 = np.eye(8)[1], np.eye(8)[2], np.eye(8)[4]
        result = build_naive_memory_baseline(
            Q,
            [memory("useful-1", e2), memory("useful-2", e3), memory("other", e5, scope="electronics")],
            query_scope="footwear",
        )
        self.assertIs(result.mode, BaselineMode.NAIVE)
        self.assertEqual(result.selected_memory_ids, ("useful-1", "useful-2"))
        np.testing.assert_allclose(result.aggregate_delta, 0.5 * e2 + 0.5 * e3)
        self.assertEqual(
            [item.aggregation_weight for item in result.memory_diagnostics],
            [0.5, 0.5, 0.0],
        )
        self.assertFalse(result.memory_diagnostics[2].scope_compatible)
        self.assertIsNone(result.memory_diagnostics[2].raw_cosine)

    def test_zero_eligible_memory_is_zero_without_nans(self) -> None:
        result = build_naive_memory_baseline(
            Q, [memory("other", np.eye(8)[1], scope="electronics")], query_scope="footwear"
        )
        self.assertEqual(result.selected_memory_ids, ())
        np.testing.assert_array_equal(result.aggregate_delta, np.zeros(8))
        self.assertTrue(np.all(np.isfinite(result.aggregate_delta)))
        steered = bound_query_shift(Q, result.aggregate_delta)
        np.testing.assert_array_equal(steered.q_star, Q)
        self.assertTrue(steered.diagnostics.delta_zero)

    def test_negative_polarity_is_excluded_not_subtracted(self) -> None:
        negative = memory(
            "negative", np.eye(8)[1], polarity=MemoryPolarity.NEGATIVE
        )
        result = build_naive_memory_baseline(Q, [negative], query_scope="footwear")
        np.testing.assert_array_equal(result.aggregate_delta, np.zeros(8))
        diagnostic = result.memory_diagnostics[0]
        self.assertFalse(diagnostic.polarity_eligible)
        self.assertFalse(diagnostic.selected)
        self.assertIsNone(diagnostic.raw_tangent_norm)

    def test_eligible_distractors_change_naive_direction(self) -> None:
        e2, e5, e6, e7 = np.eye(8)[1], np.eye(8)[4], np.eye(8)[5], np.eye(8)[6]
        useful_only = build_naive_memory_baseline(Q, [memory("useful", e2)])
        distracted = build_naive_memory_baseline(
            Q,
            [memory("useful", e2), memory("d1", e5), memory("d2", e6), memory("d3", e7)],
        )
        np.testing.assert_array_equal(useful_only.aggregate_delta, e2)
        self.assertFalse(np.allclose(distracted.aggregate_delta, e2))
        np.testing.assert_allclose(
            distracted.aggregate_delta, 0.25 * (e2 + e5 + e6 + e7)
        )


class CosineBaselineTests(unittest.TestCase):
    def test_untuned_selection_defaults_are_explicit(self) -> None:
        config = BaselineConfig()
        self.assertEqual(config.memory_top_k, 3)
        self.assertIsNone(config.cosine_threshold)
        self.assertTrue(config.allow_unscoped_memory)
        self.assertEqual(config.epsilon, 1e-8)

    def test_order_top_k_weights_and_raw_aggregate(self) -> None:
        high = cosine_memory("high", 0.9, 1)
        medium = cosine_memory("medium", 0.6, 2)
        low = cosine_memory("low", 0.2, 3)
        result = build_cosine_memory_baseline(
            Q, [low, medium, high], config=BaselineConfig(memory_top_k=2)
        )
        self.assertEqual(result.selected_memory_ids, ("high", "medium"))
        expected = 0.6 * high.embedding.copy() + 0.4 * medium.embedding.copy()
        expected[0] = 0.0
        np.testing.assert_allclose(result.aggregate_delta, expected)
        diagnostics = {item.memory_id: item for item in result.memory_diagnostics}
        self.assertEqual(diagnostics["high"].selection_rank, 1)
        self.assertEqual(diagnostics["medium"].selection_rank, 2)
        self.assertAlmostEqual(diagnostics["high"].aggregation_weight, 0.6)
        self.assertAlmostEqual(diagnostics["medium"].aggregation_weight, 0.4)

    def test_threshold_is_inclusive_and_top_k_can_exceed_count(self) -> None:
        boundary = cosine_memory("boundary", 0.5, 1)
        below = cosine_memory("below", 0.49, 2)
        result = build_cosine_memory_baseline(
            Q,
            [below, boundary],
            config=BaselineConfig(memory_top_k=20, cosine_threshold=0.5),
        )
        self.assertEqual(result.selected_memory_ids, ("boundary",))
        diagnostics = {item.memory_id: item for item in result.memory_diagnostics}
        self.assertTrue(diagnostics["boundary"].threshold_passed)
        self.assertFalse(diagnostics["below"].threshold_passed)

    def test_ties_use_original_order(self) -> None:
        first = cosine_memory("first", 0.7, 1)
        second = cosine_memory("second", 0.7, 2)
        result = build_cosine_memory_baseline(
            Q, [second, first], config=BaselineConfig(memory_top_k=2)
        )
        self.assertEqual(result.selected_memory_ids, ("second", "first"))

    def test_negative_cosines_never_become_negative_weights(self) -> None:
        positive = cosine_memory("positive", 0.2, 1)
        negative = cosine_memory("negative", -0.8, 2)
        result = build_cosine_memory_baseline(
            Q, [negative, positive], config=BaselineConfig(memory_top_k=2)
        )
        weights = {item.memory_id: item.aggregation_weight for item in result.memory_diagnostics}
        self.assertEqual(weights, {"negative": 0.0, "positive": 1.0})
        self.assertTrue(all(weight >= 0.0 for weight in weights.values()))
        all_negative = build_cosine_memory_baseline(
            Q, [negative], config=BaselineConfig(memory_top_k=1)
        )
        np.testing.assert_array_equal(all_negative.aggregate_delta, np.zeros(8))

    def test_high_cosine_redundant_memory_has_tiny_steering(self) -> None:
        redundant = memory("redundant", normalize(Q + 1e-10 * np.eye(8)[1]))
        directional = cosine_memory("directional", 0.8, 2)
        result = build_cosine_memory_baseline(
            Q, [directional, redundant], config=BaselineConfig(memory_top_k=1)
        )
        self.assertEqual(result.selected_memory_ids, ("redundant",))
        self.assertLess(float(np.linalg.norm(result.aggregate_delta)), 1e-9)
        self.assertGreater(float(np.linalg.norm(result.aggregate_delta)), 0.0)

    def test_zero_structurally_eligible_memory(self) -> None:
        result = build_cosine_memory_baseline(
            Q,
            [memory("negative", np.eye(8)[1], polarity=MemoryPolarity.NEGATIVE)],
        )
        self.assertEqual(result.selected_memory_ids, ())
        np.testing.assert_array_equal(result.aggregate_delta, np.zeros(8))

    def test_raw_residual_keeps_unsupported_axis(self) -> None:
        mixed = memory("mixed", 0.6 * np.eye(8)[2] + 0.8 * np.eye(8)[5])
        result = build_cosine_memory_baseline(
            Q, [mixed], config=BaselineConfig(memory_top_k=1)
        )
        # Orthogonal cosine is zero, so clamped-cosine weighting deliberately
        # produces no steering unless a positive-cosine memory is selected.
        self.assertEqual(result.memory_diagnostics[0].aggregation_weight, 0.0)
        positive_mixed = memory("positive-mixed", normalize(Q + mixed.embedding))
        result = build_cosine_memory_baseline(Q, [positive_mixed])
        self.assertGreater(result.aggregate_delta[5], 0.0)

    def test_config_and_input_validation(self) -> None:
        for top_k in (0, -1, True, 1.5):
            with self.subTest(top_k=top_k), self.assertRaises(ValueError):
                BaselineConfig(memory_top_k=top_k)  # type: ignore[arg-type]
        for threshold in (-1.1, 1.1, np.nan, np.inf):
            with self.subTest(threshold=threshold), self.assertRaises(ValueError):
                BaselineConfig(cosine_threshold=threshold)
        with self.assertRaises(ValueError):
            BaselineConfig(allow_unscoped_memory=1)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            BaselineConfig(epsilon=0.0)
        with self.assertRaises(ValueError):
            build_naive_memory_baseline(Q, [object()])  # type: ignore[list-item]


if __name__ == "__main__":
    unittest.main()
