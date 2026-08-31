from __future__ import annotations

import unittest

import numpy as np

from nickolas.memory.qlmp import build_local_subspace, normalize, project_memory_residual
from nickolas.memory.qlmp.models import _basis_contract_tolerance
from nickolas.memory.qlmp.projection import _roundoff_clamp_fraction
from nickolas.memory.qlmp.synthetic import (
    LOCAL_PRODUCT_VECTORS,
    MIXED_MEMORY_VECTOR,
    QUERY_VECTOR,
    REDUNDANT_MEMORY_VECTOR,
    SUPPORTED_COMBINATION_MEMORY_VECTOR,
    SUPPORTED_MEMORY_VECTOR,
    UNSUPPORTED_MEMORY_VECTOR,
)


class ControlledProjectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.subspace = build_local_subspace(QUERY_VECTOR, LOCAL_PRODUCT_VECTORS, rank=2)

    def test_supported_axis_is_fully_retained(self) -> None:
        result = project_memory_residual(
            QUERY_VECTOR, SUPPORTED_MEMORY_VECTOR, self.subspace.basis
        )
        self.assertAlmostEqual(result.raw_query_memory_cosine, 0.0)
        self.assertAlmostEqual(result.tangent_norm, 1.0)
        self.assertAlmostEqual(result.projected_norm, 1.0)
        self.assertAlmostEqual(result.projection_fraction, 0.99999999, places=8)
        np.testing.assert_allclose(result.projected_residual, SUPPORTED_MEMORY_VECTOR, atol=1e-14)

    def test_supported_combination_is_fully_retained(self) -> None:
        result = project_memory_residual(
            QUERY_VECTOR, SUPPORTED_COMBINATION_MEMORY_VECTOR, self.subspace.basis
        )
        self.assertAlmostEqual(result.tangent_norm, 1.0)
        self.assertAlmostEqual(result.projected_norm, 1.0)
        self.assertAlmostEqual(result.projection_fraction, 0.99999999, places=8)
        np.testing.assert_allclose(
            result.projected_residual, SUPPORTED_COMBINATION_MEMORY_VECTOR, atol=1e-14
        )

    def test_unsupported_axis_is_removed(self) -> None:
        result = project_memory_residual(
            QUERY_VECTOR, UNSUPPORTED_MEMORY_VECTOR, self.subspace.basis
        )
        self.assertAlmostEqual(result.tangent_norm, 1.0)
        self.assertAlmostEqual(result.projected_norm, 0.0, places=15)
        self.assertAlmostEqual(result.projection_fraction, 0.0, places=15)
        np.testing.assert_allclose(result.projected_residual, 0.0, atol=1e-15)

    def test_mixed_memory_keeps_supported_component_only(self) -> None:
        result = project_memory_residual(
            QUERY_VECTOR, MIXED_MEMORY_VECTOR, self.subspace.basis
        )
        expected = np.zeros(8)
        expected[1] = 0.6
        self.assertAlmostEqual(result.tangent_norm, 1.0)
        self.assertAlmostEqual(result.projected_norm, 0.6)
        self.assertAlmostEqual(result.projection_fraction, 0.3599999964, places=10)
        np.testing.assert_allclose(result.projected_residual, expected, atol=1e-14)

    def test_query_parallel_memory_is_stable(self) -> None:
        result = project_memory_residual(
            QUERY_VECTOR, REDUNDANT_MEMORY_VECTOR, self.subspace.basis
        )
        self.assertAlmostEqual(result.raw_cosine, 1.0)
        self.assertEqual(result.tangent_norm, 0.0)
        self.assertEqual(result.projected_norm, 0.0)
        self.assertEqual(result.projection_fraction, 0.0)
        np.testing.assert_array_equal(result.residual, np.zeros(8))

    def test_zero_column_basis_is_supported(self) -> None:
        result = project_memory_residual(
            QUERY_VECTOR, SUPPORTED_MEMORY_VECTOR, np.empty((8, 0))
        )
        self.assertEqual(result.coefficients.shape, (0,))
        np.testing.assert_array_equal(result.projected_residual, np.zeros(8))
        self.assertEqual(result.projection_fraction, 0.0)

    def test_basis_validation(self) -> None:
        invalid = (
            np.eye(8)[:, :2] * 2.0,
            np.eye(8)[:, :2],
            np.ones((7, 1)),
            np.ones(8),
            np.full((8, 1), np.nan),
        )
        for basis in invalid:
            with self.subTest(shape=basis.shape), self.assertRaises(ValueError):
                project_memory_residual(QUERY_VECTOR, SUPPORTED_MEMORY_VECTOR, basis)

    def test_scaled_basis_regression_is_rejected(self) -> None:
        scaled = self.subspace.basis * (1.0 + 1e-8)
        with self.assertRaisesRegex(ValueError, "orthonormal"):
            project_memory_residual(QUERY_VECTOR, SUPPORTED_MEMORY_VECTOR, scaled)

    def test_machine_error_basis_is_accepted_and_fraction_overshoot_is_clamped(self) -> None:
        machine_epsilon = np.finfo(np.float64).eps
        q = np.asarray([1.0, 0.0, 0.0])
        memory = np.asarray([0.0, 1.0, 0.0])
        basis = np.asarray([[0.0], [1.0 + 2.0 * machine_epsilon], [0.0]])
        result = project_memory_residual(
            q, memory, basis, epsilon=np.finfo(np.float64).tiny
        )
        self.assertEqual(result.projection_fraction, 1.0)
        self.assertGreater(result.projected_norm, result.tangent_norm)

    def test_material_fraction_overshoot_is_never_silently_accepted(self) -> None:
        tau = _basis_contract_tolerance(8, 2)
        allowance = 2.0 * tau + tau**2 + 32.0 * np.finfo(np.float64).eps
        with self.assertRaisesRegex(RuntimeError, "escaped"):
            _roundoff_clamp_fraction(1.0 + 2.0 * allowance, tau)

    def test_inputs_must_be_normalized_and_dimensionally_equal(self) -> None:
        with self.assertRaises(ValueError):
            project_memory_residual(2.0 * QUERY_VECTOR, SUPPORTED_MEMORY_VECTOR, self.subspace.basis)
        with self.assertRaises(ValueError):
            project_memory_residual(QUERY_VECTOR, 2.0 * SUPPORTED_MEMORY_VECTOR, self.subspace.basis)
        with self.assertRaises(ValueError):
            project_memory_residual(QUERY_VECTOR, [1.0, 0.0], self.subspace.basis)

    def test_projection_has_pythagorean_decomposition_and_is_idempotent(self) -> None:
        result = project_memory_residual(
            QUERY_VECTOR, MIXED_MEMORY_VECTOR, self.subspace.basis
        )
        rejected = result.residual - result.projected_residual
        self.assertAlmostEqual(
            result.tangent_norm**2,
            result.projected_norm**2 + float(np.linalg.norm(rejected)) ** 2,
            places=14,
        )
        projector = self.subspace.basis @ self.subspace.basis.T
        np.testing.assert_allclose(
            projector @ result.projected_residual,
            result.projected_residual,
            atol=1e-15,
        )

    def test_near_parallel_memory_has_conservative_epsilon_fraction(self) -> None:
        q = np.asarray([1.0, 0.0, 0.0])
        memory = normalize([1.0, 1e-10, 0.0])
        basis = np.asarray([[0.0], [1.0], [0.0]])
        result = project_memory_residual(q, memory, basis)
        self.assertGreater(result.tangent_norm, 0.0)
        self.assertAlmostEqual(result.projected_norm, result.tangent_norm, places=20)
        self.assertAlmostEqual(result.projection_fraction, 1e-12, places=20)

    def test_float32_inputs_produce_owned_readonly_float64_diagnostics(self) -> None:
        q = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
        memory = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
        basis = np.asarray([[0.0], [1.0], [0.0]], dtype=np.float32)
        result = project_memory_residual(q, memory, basis)
        for diagnostic in (
            result.residual,
            result.coefficients,
            result.projected_residual,
        ):
            self.assertEqual(diagnostic.dtype, np.float64)
            self.assertFalse(diagnostic.flags.writeable)
            self.assertFalse(np.shares_memory(diagnostic, memory))
            self.assertFalse(np.shares_memory(diagnostic, basis))
        for scalar in (
            result.raw_query_memory_cosine,
            result.tangent_norm,
            result.projected_norm,
            result.projection_fraction,
        ):
            self.assertIs(type(scalar), float)


class RandomProjectionInvariantTests(unittest.TestCase):
    def test_fixed_seed_randomized_projection_invariants(self) -> None:
        rng = np.random.default_rng(2602)
        for _ in range(30):
            q = normalize(rng.normal(size=8))
            products = np.vstack([normalize(q + rng.normal(size=8)) for _ in range(10)])
            subspace = build_local_subspace(q, products, rank=4)
            memory = normalize(rng.normal(size=8))
            result = project_memory_residual(q, memory, subspace.basis)

            self.assertAlmostEqual(float(q @ result.residual), 0.0, places=14)
            self.assertAlmostEqual(float(q @ result.projected_residual), 0.0, places=14)
            self.assertLessEqual(result.projected_norm, result.tangent_norm + 1e-12)
            self.assertGreaterEqual(result.projection_fraction, 0.0)
            self.assertLessEqual(result.projection_fraction, 1.0)
            projector = subspace.basis @ subspace.basis.T
            np.testing.assert_allclose(
                projector @ result.projected_residual,
                result.projected_residual,
                atol=2e-14,
            )
            np.testing.assert_allclose(
                subspace.basis.T @ subspace.basis,
                np.eye(subspace.effective_rank),
                atol=2e-14,
            )


if __name__ == "__main__":
    unittest.main()
