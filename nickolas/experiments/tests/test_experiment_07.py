from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import fields
from pathlib import Path

import numpy as np

from nickolas.experiments.config import MAX_TURNS, default_public_set, default_results
from nickolas.experiments.experiment_07_residual_failure_analysis import (
    CASCADE_COMPONENTS,
    EXPECTED_EXACT_BASELINE,
    METHODS,
    RetrievalInput,
    assert_exact_baseline_identity,
    corrected_active_evidence,
    deterministic_rrf,
    failure_category_counts,
    fallback_reasons,
    is_normalization_problem,
    load_frozen_split,
    robust_normalize,
    validate_artifact_payloads,
)
from nickolas.experiments.harness import rank_metrics


class Experiment07Tests(unittest.TestCase):
    def test_exact_baseline_identity(self) -> None:
        row = {
            "sample_id": "public_0001",
            "scenario_type": "buying",
            "hit": True,
            "first_hit_turn": 1,
            "best_rank": 2,
            "reciprocal_rank": 0.5,
        }
        assert_exact_baseline_identity([row], [{**row, "ignored": "oracle-only"}])
        with self.assertRaises(RuntimeError):
            assert_exact_baseline_identity([row], [{**row, "best_rank": 3}])
        self.assertEqual(EXPECTED_EXACT_BASELINE["technical_score"], 0.816917)

    def test_ranker_input_has_no_leakage_fields(self) -> None:
        self.assertEqual([field.name for field in fields(RetrievalInput)], ["category", "active_constraints"])
        item = RetrievalInput("Running Shoes", ("blue", "wide fit"))
        self.assertEqual(item.query, "Running Shoes blue wide fit")
        forbidden = {"target_asin", "scenario_type", "sample_id", "oracle_card", "user_profile"}
        self.assertFalse(forbidden & set(vars(item)))

    def test_fallback_activation(self) -> None:
        self.assertIn("no_active_constraint", fallback_reasons((), 20, 20))
        self.assertIn("no_catalog_product_matches_all_active_phrases", fallback_reasons(("blue",), 0, 4))
        self.assertIn("highest_exact_match_tier_exceeds_top_10", fallback_reasons(("blue",), 11, 11))
        self.assertEqual(fallback_reasons(("blue",), 3, 3), ())

    def test_deterministic_equal_weight_rrf(self) -> None:
        ids = np.asarray(["B", "A", "C"])
        first, first_scores = deterministic_rrf([np.asarray([0, 1]), np.asarray([1, 0])], ids)
        second, second_scores = deterministic_rrf([np.asarray([1, 0]), np.asarray([0, 1])], ids)
        self.assertEqual(list(ids[first]), ["A", "B"])
        self.assertEqual(list(first), list(second))
        np.testing.assert_allclose(first_scores, second_scores)

    def test_normalization_taxonomy(self) -> None:
        self.assertNotIn(normalize_current := "blue—cotton", "blue cotton shirt")
        self.assertIn(robust_normalize(normalize_current), robust_normalize("Blue cotton shirt"))
        self.assertTrue(is_normalization_problem(False, True))
        self.assertFalse(is_normalization_problem(True, True))

    def test_override_invalidation(self) -> None:
        corrected = corrected_active_evidence(
            ("old preference", "new requirement"),
            ("old preference", "new requirement", "machine wash"),
            ("old preference",),
        )
        self.assertEqual(corrected, ("new requirement", "machine wash"))

    def test_multi_label_counts_are_non_exclusive(self) -> None:
        rows = [
            {
                "diagnostic_residual_type": "hard_failure",
                "diagnostic_failure_tags": ["no_exact_match", "ranking_failure"],
            },
            {
                "diagnostic_residual_type": "weak_success",
                "diagnostic_failure_tags": ["ranking_failure"],
            },
        ]
        counts = {row["failure_category"]: row for row in failure_category_counts(rows)}
        self.assertEqual(counts["no_exact_match"]["session_count"], 1)
        self.assertEqual(counts["ranking_failure"]["session_count"], 2)
        self.assertTrue(all(row["non_exclusive"] for row in counts.values()))

    def test_experiment_06_split_is_reused_and_partitions_public_set(self) -> None:
        ids = {
            json.loads(line)["sample_id"]
            for line in default_public_set().read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        split_path = default_results() / "experiment_06_slate_width_counterfactuals" / "metrics.json"
        calibration, evaluation, metadata = load_frozen_split(split_path, ids)
        self.assertEqual((len(calibration), len(evaluation)), (60, 140))
        self.assertEqual(calibration | evaluation, ids)
        self.assertTrue(metadata["validated_partition"])

    def test_metric_formula(self) -> None:
        rows = [
            {"hit": True, "first_hit_turn": 2, "reciprocal_rank": 0.5},
            {"hit": False, "first_hit_turn": None, "reciprocal_rank": 0.0},
        ]
        self.assertEqual(
            rank_metrics(rows),
            {
                "sample_count": 2,
                "hit_rate_at_10": 0.5,
                "mrr": 0.25,
                "mttc": 6.5,
                "efficiency": 0.45,
                "technical_score": 0.415,
            },
        )

    def test_artifact_schema_validator(self) -> None:
        metrics = {"experiment": 7, "method_metrics": {method: {} for method in METHODS}}
        rows = [{} for _ in range(200 * MAX_TURNS)]
        sessions = {method: [{} for _ in range(200)] for method in METHODS}
        hard = [
            {
                "oracle_target_asin": "T",
                "diagnostic_failure_tags": ["ranking_failure"],
            }
        ]
        weak = [
            {
                "oracle_target_asin": "U",
                "diagnostic_failure_tags": ["ranking_failure"],
            }
        ]
        residual = [{} for _ in range((len(hard) + len(weak)) * MAX_TURNS)]
        validate_artifact_payloads(metrics, rows, sessions, residual, hard, weak)
        with self.assertRaises(RuntimeError):
            validate_artifact_payloads(metrics, rows[:-1], sessions, residual, hard, weak)


if __name__ == "__main__":
    unittest.main()
