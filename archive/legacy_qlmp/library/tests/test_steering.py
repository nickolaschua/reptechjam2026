from __future__ import annotations

import math
import unittest

import numpy as np

from nickolas.memory.qlmp import SteeringConfig, bound_query_shift, normalize


class AngularSteeringTests(unittest.TestCase):
    def setUp(self) -> None:
        self.q = np.asarray([1.0, 0.0, 0.0])
        self.delta = np.asarray([0.0, 1.0, 0.0])

    def test_untuned_steering_defaults_are_explicit(self) -> None:
        config = SteeringConfig()
        self.assertEqual(config.beta, 1.0)
        self.assertEqual(config.max_shift_deg, 10.0)
        self.assertEqual(config.epsilon, 1e-8)

    def test_below_limit_is_not_clipped(self) -> None:
        beta = math.tan(math.radians(5.0))
        result = bound_query_shift(
            self.q,
            self.delta,
            config=SteeringConfig(beta=beta, max_shift_deg=10.0),
        )
        self.assertFalse(result.diagnostics.clipped)
        self.assertAlmostEqual(result.diagnostics.applied_beta, beta)
        self.assertAlmostEqual(result.diagnostics.unclipped_angle_deg, 5.0, places=12)
        self.assertAlmostEqual(result.diagnostics.actual_shift_deg, 5.0, places=12)

    def test_above_limit_is_clipped_to_maximum(self) -> None:
        result = bound_query_shift(
            self.q,
            self.delta,
            config=SteeringConfig(
                beta=math.tan(math.radians(20.0)), max_shift_deg=10.0
            ),
        )
        self.assertTrue(result.diagnostics.clipped)
        self.assertAlmostEqual(result.diagnostics.actual_shift_deg, 10.0, places=12)
        self.assertAlmostEqual(
            result.diagnostics.applied_tangent_norm,
            math.tan(math.radians(10.0)),
            places=15,
        )

    def test_exact_boundary_is_stable_and_not_clipped(self) -> None:
        beta = math.tan(math.radians(10.0))
        result = bound_query_shift(
            self.q,
            self.delta,
            config=SteeringConfig(beta=beta, max_shift_deg=10.0),
        )
        self.assertFalse(result.diagnostics.clipped)
        self.assertAlmostEqual(result.diagnostics.actual_shift_deg, 10.0, places=12)

    def test_zero_delta_returns_query_unchanged(self) -> None:
        result = bound_query_shift(self.q, np.zeros(3))
        np.testing.assert_array_equal(result.q_star, self.q)
        self.assertTrue(result.diagnostics.delta_zero)
        self.assertFalse(result.diagnostics.clipped)
        self.assertEqual(result.diagnostics.actual_shift_deg, 0.0)
        self.assertEqual(result.diagnostics.applied_beta, 0.0)

    def test_non_tangent_delta_is_corrected_before_bounding(self) -> None:
        result = bound_query_shift(
            self.q,
            np.asarray([1.0, 1.0, 0.0]),
            config=SteeringConfig(
                beta=math.tan(math.radians(20.0)), max_shift_deg=10.0
            ),
        )
        self.assertTrue(result.diagnostics.tangency_corrected)
        self.assertAlmostEqual(result.diagnostics.corrected_tangent_norm, 1.0)
        self.assertAlmostEqual(result.diagnostics.actual_shift_deg, 10.0, places=12)

    def test_parallel_only_delta_becomes_effective_zero(self) -> None:
        result = bound_query_shift(self.q, np.asarray([4.0, 0.0, 0.0]))
        np.testing.assert_array_equal(result.q_star, self.q)
        self.assertTrue(result.diagnostics.tangency_corrected)
        self.assertTrue(result.diagnostics.delta_zero)
        self.assertEqual(result.diagnostics.original_delta_norm, 4.0)

    def test_inputs_unchanged_and_output_owned_float64_readonly(self) -> None:
        q = self.q.astype(np.float32)
        delta = self.delta.astype(np.float32)
        q_before, delta_before = q.copy(), delta.copy()
        result = bound_query_shift(q, delta)
        np.testing.assert_array_equal(q, q_before)
        np.testing.assert_array_equal(delta, delta_before)
        self.assertEqual(result.q_star.dtype, np.float64)
        self.assertFalse(result.q_star.flags.writeable)
        self.assertFalse(np.shares_memory(result.q_star, q))
        self.assertFalse(np.shares_memory(result.q_star, delta))

    def test_invalid_configuration_and_inputs_fail(self) -> None:
        for beta in (-1.0, np.nan, np.inf, True):
            with self.subTest(beta=beta), self.assertRaises(ValueError):
                SteeringConfig(beta=beta)
        for angle in (-1.0, 90.0, 100.0, np.nan, np.inf):
            with self.subTest(angle=angle), self.assertRaises(ValueError):
                SteeringConfig(max_shift_deg=angle)
        with self.assertRaises(ValueError):
            bound_query_shift([2.0, 0.0], [0.0, 1.0])
        with self.assertRaises(ValueError):
            bound_query_shift([1.0, 0.0], [0.0, 1.0, 0.0])
        with self.assertRaises(ValueError):
            bound_query_shift([1.0, 0.0], [0.0, np.nan])


class RandomSteeringInvariantTests(unittest.TestCase):
    def test_fixed_seed_randomized_bounds_and_norms(self) -> None:
        rng = np.random.default_rng(2603)
        for _ in range(200):
            q = normalize(rng.normal(size=12))
            delta = rng.normal(size=12)
            q_before, delta_before = q.copy(), delta.copy()
            beta = float(rng.uniform(0.0, 8.0))
            max_shift = float(rng.uniform(0.0, 25.0))
            result = bound_query_shift(
                q,
                delta,
                config=SteeringConfig(beta=beta, max_shift_deg=max_shift),
            )
            self.assertTrue(np.all(np.isfinite(result.q_star)))
            self.assertAlmostEqual(float(np.linalg.norm(result.q_star)), 1.0, places=14)
            self.assertLessEqual(
                result.diagnostics.actual_shift_deg, max_shift + 2e-12
            )
            np.testing.assert_array_equal(q, q_before)
            np.testing.assert_array_equal(delta, delta_before)


if __name__ == "__main__":
    unittest.main()
