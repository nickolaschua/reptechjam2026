from __future__ import annotations

import unittest

import numpy as np

from nickolas.memory.qlmp import (
    LocalSubspace,
    ProjectionConfig,
    build_local_subspace,
    build_tangent_matrix,
    cosine_similarity,
    normalize,
    tangent_residual,
)
from nickolas.memory.qlmp.geometry import _numerical_rank_tolerance
from nickolas.memory.qlmp.synthetic import LOCAL_PRODUCT_VECTORS, QUERY_VECTOR


class NormalizeAndCosineTests(unittest.TestCase):
    def test_normalize_returns_float64_copy_without_mutating_input(self) -> None:
        original = np.asarray([3, 4], dtype=np.int32)
        before = original.copy()
        result = normalize(original)
        np.testing.assert_array_equal(original, before)
        self.assertEqual(result.dtype, np.float64)
        self.assertFalse(np.shares_memory(result, original))
        np.testing.assert_allclose(result, [0.6, 0.8])

    def test_normalize_rejects_empty_shape_nonfinite_and_small_vectors(self) -> None:
        invalid = (
            [],
            [[1.0, 0.0]],
            [np.nan, 0.0],
            [np.inf, 0.0],
            [0.0, 0.0],
            [1e-8, 0.0],
        )
        for vector in invalid:
            with self.subTest(vector=vector), self.assertRaises(ValueError):
                normalize(vector)
        np.testing.assert_allclose(normalize([1.0001e-8, 0.0]), [1.0, 0.0])

    def test_cosine_examples_and_nonmutation(self) -> None:
        left = np.asarray([2.0, 0.0])
        right = np.asarray([1.0, 1.0])
        left_before, right_before = left.copy(), right.copy()
        self.assertAlmostEqual(cosine_similarity(left, right), 1.0 / np.sqrt(2.0))
        self.assertAlmostEqual(cosine_similarity(left, [-1.0, 0.0]), -1.0)
        self.assertAlmostEqual(cosine_similarity(left, [0.0, 4.0]), 0.0)
        np.testing.assert_array_equal(left, left_before)
        np.testing.assert_array_equal(right, right_before)

    def test_cosine_rejects_bad_dimensions_and_values(self) -> None:
        cases = (
            ([1.0], [1.0, 0.0]),
            ([[1.0]], [1.0]),
            ([1.0], [np.nan]),
            ([0.0], [1.0]),
            ([1e-9], [1.0]),
        )
        for left, right in cases:
            with self.subTest(left=left, right=right), self.assertRaises(ValueError):
                cosine_similarity(left, right)

    def test_projection_config_validation_and_not_tuned_defaults(self) -> None:
        config = ProjectionConfig()
        self.assertEqual(config.rank, 16)
        self.assertEqual(config.epsilon, 1e-8)
        for rank in (0, -1, True, 1.5):
            with self.subTest(rank=rank), self.assertRaises(ValueError):
                ProjectionConfig(rank=rank)  # type: ignore[arg-type]
        for epsilon in (0.0, -1.0, np.inf, np.nan, True):
            with self.subTest(epsilon=epsilon), self.assertRaises(ValueError):
                ProjectionConfig(epsilon=epsilon)


class TangentGeometryTests(unittest.TestCase):
    def test_tangent_residual_is_orthogonal(self) -> None:
        q = normalize([1.0, 2.0, -1.0])
        vector = normalize([2.0, -3.0, 0.5])
        residual = tangent_residual(q, vector)
        self.assertAlmostEqual(float(q @ residual), 0.0, places=15)

    def test_near_parallel_residual_is_stable(self) -> None:
        q = np.asarray([1.0, 0.0, 0.0])
        vector = normalize([1.0, 1e-10, 0.0])
        residual = tangent_residual(q, vector)
        self.assertAlmostEqual(float(q @ residual), 0.0, places=15)
        self.assertGreater(float(np.linalg.norm(residual)), 0.0)
        np.testing.assert_allclose(residual[1:], [1e-10, 0.0], rtol=1e-12, atol=0.0)

    def test_tangent_operations_require_normalized_inputs(self) -> None:
        with self.assertRaisesRegex(ValueError, "q must be normalized"):
            tangent_residual([2.0, 0.0], [1.0, 0.0])
        with self.assertRaisesRegex(ValueError, "vector must be normalized"):
            tangent_residual([1.0, 0.0], [1.0, 1.0])
        with self.assertRaises(ValueError):
            tangent_residual([1.0, 0.0], [1.0, 0.0, 0.0])
        with self.assertRaises(ValueError):
            tangent_residual([1.0, 0.0], [np.nan, 0.0])

    def test_tangent_matrix_shape_rows_and_nonmutation(self) -> None:
        products = LOCAL_PRODUCT_VECTORS.copy()
        before = products.copy()
        matrix = build_tangent_matrix(QUERY_VECTOR, products)
        self.assertEqual(matrix.shape, products.shape)
        np.testing.assert_allclose(matrix @ QUERY_VECTOR, 0.0, atol=1e-15)
        np.testing.assert_array_equal(products, before)

    def test_tangent_matrix_validates_shape_dimension_finiteness_and_norms(self) -> None:
        bad_products = (
            [1.0, 0.0],
            np.ones((2, 7)),
            np.vstack([QUERY_VECTOR, np.full(8, np.nan)]),
            np.vstack([QUERY_VECTOR, 2.0 * QUERY_VECTOR]),
        )
        for products in bad_products:
            with self.subTest(shape=np.asarray(products).shape), self.assertRaises(ValueError):
                build_tangent_matrix(QUERY_VECTOR, products)

    def test_float32_inputs_produce_float64_tangent_outputs(self) -> None:
        q = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
        candidate = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
        products = np.asarray(
            [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32
        )
        normalized = normalize(np.asarray([3.0, 4.0], dtype=np.float32))
        residual = tangent_residual(q, candidate)
        matrix = build_tangent_matrix(q, products)
        subspace = build_local_subspace(q, products, rank=2)
        self.assertEqual(normalized.dtype, np.float64)
        self.assertEqual(residual.dtype, np.float64)
        self.assertEqual(matrix.dtype, np.float64)
        self.assertEqual(subspace.basis.dtype, np.float64)
        self.assertEqual(subspace.singular_values.dtype, np.float64)
        self.assertFalse(subspace.basis.flags.writeable)
        self.assertFalse(subspace.singular_values.flags.writeable)


class LocalSubspaceTests(unittest.TestCase):
    def test_numerical_rank_tolerance_formula(self) -> None:
        matrix = np.empty((3, 7), dtype=np.float64)
        sigma_max = 2.5
        expected = sigma_max * 7 * np.finfo(np.float64).eps
        self.assertEqual(_numerical_rank_tolerance(matrix, sigma_max), expected)

    def test_real_svd_recovers_e2_e3_projector(self) -> None:
        subspace = build_local_subspace(QUERY_VECTOR, LOCAL_PRODUCT_VECTORS, rank=2)
        self.assertEqual(subspace.basis.shape, (8, 2))
        self.assertEqual(subspace.effective_rank, 2)
        self.assertEqual(subspace.requested_rank, 2)
        self.assertEqual(subspace.embedding_dimension, 8)
        self.assertEqual(subspace.local_product_count, 4)
        np.testing.assert_allclose(subspace.basis.T @ subspace.basis, np.eye(2), atol=1e-14)
        np.testing.assert_allclose(QUERY_VECTOR @ subspace.basis, 0.0, atol=1e-14)
        expected_projector = np.zeros((8, 8))
        expected_projector[1, 1] = 1.0
        expected_projector[2, 2] = 1.0
        np.testing.assert_allclose(
            subspace.basis @ subspace.basis.T, expected_projector, atol=1e-14
        )

    def test_rank_above_product_count_and_dimension_is_capped(self) -> None:
        subspace = build_local_subspace(QUERY_VECTOR, LOCAL_PRODUCT_VECTORS[:2], rank=99)
        self.assertEqual(subspace.requested_rank, 99)
        self.assertEqual(subspace.effective_rank, 2)
        self.assertEqual(subspace.basis.shape, (8, 2))
        self.assertEqual(subspace.singular_values.shape, (2,))

    def test_duplicates_and_rank_deficiency_reduce_effective_rank(self) -> None:
        duplicate = LOCAL_PRODUCT_VECTORS[0]
        subspace = build_local_subspace(
            QUERY_VECTOR, np.vstack([duplicate, duplicate, duplicate]), rank=3
        )
        self.assertEqual(subspace.effective_rank, 1)
        self.assertEqual(subspace.basis.shape, (8, 1))

    def test_audited_small_real_direction_above_rank_tolerance_is_retained(self) -> None:
        e2 = np.eye(8)[1]
        e3 = np.eye(8)[2]
        products = np.vstack(
            [normalize(QUERY_VECTOR + e2), normalize(QUERY_VECTOR + e2 + 1e-8 * e3)]
        )
        tangent_matrix = build_tangent_matrix(QUERY_VECTOR, products)
        measured = np.linalg.svd(tangent_matrix, compute_uv=False)
        rank_tolerance = _numerical_rank_tolerance(tangent_matrix, measured[0])
        self.assertGreaterEqual(measured[1], 100.0 * rank_tolerance)
        self.assertGreater(measured[1], 1e-9)
        self.assertLess(measured[1], 1e-8)
        subspace = build_local_subspace(QUERY_VECTOR, products, rank=2, epsilon=1e-8)
        self.assertEqual(subspace.effective_rank, 2)

    def test_direction_well_below_rank_tolerance_is_discarded(self) -> None:
        e2 = np.eye(8)[1]
        e3 = np.eye(8)[2]
        products = np.vstack(
            [normalize(QUERY_VECTOR + e2), normalize(QUERY_VECTOR + e2 + 1e-17 * e3)]
        )
        tangent_matrix = build_tangent_matrix(QUERY_VECTOR, products)
        measured = np.linalg.svd(tangent_matrix, compute_uv=False)
        rank_tolerance = _numerical_rank_tolerance(tangent_matrix, measured[0])
        self.assertLessEqual(measured[1], 0.01 * rank_tolerance)
        subspace = build_local_subspace(QUERY_VECTOR, products, rank=2)
        self.assertEqual(subspace.effective_rank, 1)

    def test_local_subspace_rejects_scaled_nonorthonormal_basis(self) -> None:
        with self.assertRaisesRegex(ValueError, "orthonormal"):
            LocalSubspace(
                basis=np.eye(3)[:, :1] * (1.0 + 1e-8),
                singular_values=np.asarray([1.0]),
                requested_rank=1,
                effective_rank=1,
                embedding_dimension=3,
                local_product_count=1,
            )

    def test_empty_and_all_parallel_neighbourhoods_have_zero_rank(self) -> None:
        empty = build_local_subspace(QUERY_VECTOR, [], rank=4)
        self.assertEqual(empty.basis.shape, (8, 0))
        self.assertEqual(empty.singular_values.shape, (0,))
        self.assertEqual(empty.effective_rank, 0)
        parallel = build_local_subspace(
            QUERY_VECTOR, np.vstack([QUERY_VECTOR, QUERY_VECTOR, QUERY_VECTOR]), rank=3
        )
        self.assertEqual(parallel.effective_rank, 0)
        self.assertEqual(parallel.basis.shape, (8, 0))
        np.testing.assert_allclose(parallel.singular_values, 0.0)

    def test_invalid_rank_and_query_fail_clearly(self) -> None:
        for rank in (0, -1, True, 1.5):
            with self.subTest(rank=rank), self.assertRaises(ValueError):
                build_local_subspace(QUERY_VECTOR, LOCAL_PRODUCT_VECTORS, rank=rank)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            build_local_subspace(2.0 * QUERY_VECTOR, LOCAL_PRODUCT_VECTORS, rank=2)

    def test_fixed_seed_randomized_basis_invariants(self) -> None:
        rng = np.random.default_rng(2601)
        for _ in range(20):
            q = normalize(rng.normal(size=8))
            products = np.vstack([normalize(q + 0.4 * rng.normal(size=8)) for _ in range(12)])
            subspace = build_local_subspace(q, products, rank=5)
            basis = subspace.basis
            np.testing.assert_allclose(basis.T @ basis, np.eye(5), atol=2e-14)
            np.testing.assert_allclose(q @ basis, 0.0, atol=2e-14)


if __name__ == "__main__":
    unittest.main()
