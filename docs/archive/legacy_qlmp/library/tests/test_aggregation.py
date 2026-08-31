from __future__ import annotations

import unittest

import numpy as np

from nickolas.memory.qlmp import (
    aggregate_projected_residuals,
    aggregate_raw_residuals,
    aggregate_residuals,
    build_local_subspace,
    memory_tangent_residual,
    normalize_nonnegative_weights,
    project_memory_residual,
)
from nickolas.memory.qlmp.synthetic import LOCAL_PRODUCT_VECTORS, QUERY_VECTOR


class WeightUtilityTests(unittest.TestCase):
    def test_normalization_and_float64_copy(self) -> None:
        original = np.asarray([1, 3], dtype=np.int32)
        before = original.copy()
        result = normalize_nonnegative_weights(original)
        np.testing.assert_allclose(result, [0.25, 0.75])
        np.testing.assert_array_equal(original, before)
        self.assertEqual(result.dtype, np.float64)
        self.assertFalse(np.shares_memory(result, original))

    def test_unnormalized_and_zero_total(self) -> None:
        np.testing.assert_array_equal(
            normalize_nonnegative_weights([2.0, 4.0], normalize=False),
            [2.0, 4.0],
        )
        np.testing.assert_array_equal(normalize_nonnegative_weights([0.0, 0.0]), 0.0)
        self.assertEqual(normalize_nonnegative_weights([]).shape, (0,))

    def test_invalid_weights_are_rejected(self) -> None:
        invalid = ([[1.0]], [-1.0], [np.nan], [np.inf])
        for weights in invalid:
            with self.subTest(weights=weights), self.assertRaises(ValueError):
                normalize_nonnegative_weights(weights)
        with self.assertRaises(ValueError):
            normalize_nonnegative_weights([1.0], normalize=1)  # type: ignore[arg-type]


class ResidualAggregationTests(unittest.TestCase):
    def test_memory_residual_reuses_phase_1_geometry(self) -> None:
        q = np.asarray([1.0, 0.0, 0.0])
        memory = np.asarray([0.6, 0.8, 0.0])
        np.testing.assert_allclose(memory_tangent_residual(q, memory), [0.0, 0.8, 0.0])

    def test_one_and_multiple_weighted_results(self) -> None:
        residuals = np.asarray([[0.0, 2.0, 0.0], [0.0, 0.0, 4.0]])
        np.testing.assert_allclose(
            aggregate_residuals(residuals[:1], [1.0]), [0.0, 2.0, 0.0]
        )
        np.testing.assert_allclose(
            aggregate_raw_residuals(residuals, [1.0, 3.0]),
            [0.0, 0.5, 3.0],
        )
        np.testing.assert_allclose(
            aggregate_residuals(residuals, [1.0, 3.0], normalize_weights=False),
            [0.0, 2.0, 12.0],
        )

    def test_zero_weights_and_empty_matrix_are_safe(self) -> None:
        np.testing.assert_array_equal(
            aggregate_residuals(np.eye(3), [0.0, 0.0, 0.0]), np.zeros(3)
        )
        empty = aggregate_residuals(np.empty((0, 4)), [])
        np.testing.assert_array_equal(empty, np.zeros(4))
        self.assertEqual(empty.dtype, np.float64)

    def test_invalid_residual_shapes_and_weight_count_are_rejected(self) -> None:
        invalid = ([1.0, 2.0], np.empty((2, 0)), [[1.0, np.nan]])
        for residuals in invalid:
            with self.subTest(shape=np.asarray(residuals).shape), self.assertRaises(ValueError):
                aggregate_residuals(residuals, [1.0])
        with self.assertRaises(ValueError):
            aggregate_residuals(np.eye(3), [1.0, 1.0])

    def test_inputs_are_unchanged_and_output_is_float64(self) -> None:
        residuals = np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
        weights = np.asarray([1.0, 2.0], dtype=np.float32)
        residuals_before, weights_before = residuals.copy(), weights.copy()
        result = aggregate_raw_residuals(residuals, weights)
        np.testing.assert_array_equal(residuals, residuals_before)
        np.testing.assert_array_equal(weights, weights_before)
        self.assertEqual(result.dtype, np.float64)

    def test_projected_aggregation_removes_unsupported_component(self) -> None:
        subspace = build_local_subspace(QUERY_VECTOR, LOCAL_PRODUCT_VECTORS, rank=2)
        e2, e3, e6 = np.eye(8)[1], np.eye(8)[2], np.eye(8)[5]
        first = project_memory_residual(QUERY_VECTOR, e2, subspace.basis)
        second = project_memory_residual(
            QUERY_VECTOR, 0.6 * e3 + 0.8 * e6, subspace.basis
        )
        result = aggregate_projected_residuals(
            np.vstack([first.projected_residual, second.projected_residual]),
            [0.25, 0.75],
        )
        expected = 0.25 * e2 + 0.45 * e3
        np.testing.assert_allclose(result, expected, atol=1e-14)
        self.assertEqual(result[5], 0.0)


if __name__ == "__main__":
    unittest.main()
