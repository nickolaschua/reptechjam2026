from __future__ import annotations

import unittest
from dataclasses import fields
from types import SimpleNamespace

import numpy as np

from nickolas.experiments.experiment_08_intent_routed_dense_browsing import (
    ObservableRetrievalInput,
    ObservableStateParser,
    ROBUSTNESS_TRANSFORMS,
    detect_initial_intent,
    routed_rank,
    transform_constraint,
)
from nickolas.experiments.experiment_09_adaptive_hybrid_architecture import (
    AdaptiveHybridRanker,
    TypedConstraint,
    TypedObservableState,
    TypedStateParser,
    classify_typed_constraint,
    deterministic_reranker_score,
    entropy_question,
    rank_aware_question,
    weighted_rrf,
)
from nickolas.experiments.harness import normalize


class Experiment08Tests(unittest.TestCase):
    def test_detector_routes_templates_boundary_override_paraphrase_and_unknown(self) -> None:
        self.assertEqual(detect_initial_intent("I'm looking for Shoes, but I'm still exploring."), "browsing")
        self.assertEqual(detect_initial_intent("I'm looking for Shoes, but I'm weighing my options."), "browsing")
        # Boundary shares the exploratory first-message template and therefore
        # takes the browsing route without observing its scenario label.
        self.assertEqual(detect_initial_intent("I'm looking for Shirts, but I'm still exploring."), "browsing")
        self.assertEqual(detect_initial_intent("I'm looking for Shoes. A key requirement is: wide fit."), "buying")
        self.assertEqual(detect_initial_intent("I'm looking for Shoes. I prefer trail shoes"), "buying")
        self.assertEqual(detect_initial_intent("surprise me"), "buying")

    def test_route_lock_and_override_removal_before_query(self) -> None:
        parser = ObservableStateParser()
        first = parser.update("I'm looking for Shoes. I prefer red", 1)
        self.assertEqual(first.locked_intent, "buying")
        self.assertEqual(first.active_constraints, ("I prefer red",))
        second = parser.update("For that, what matters is: waterproof.", 2)
        self.assertEqual(second.locked_intent, "buying")
        third = parser.update("Actually, ignore my earlier preference. What I need is: blue.", 3)
        self.assertEqual(third.locked_intent, "buying")
        self.assertNotIn("I prefer red", third.active_constraints)
        self.assertEqual(third.active_constraints, ("waterproof", "blue"))
        self.assertNotIn("I prefer red", third.query)

    def test_ranker_input_has_no_oracle_fields(self) -> None:
        self.assertEqual(
            [item.name for item in fields(ObservableRetrievalInput)],
            ["category", "active_constraints", "locked_intent"],
        )
        self.assertFalse({"target_asin", "sample_id", "scenario_type"} & set(vars(ObservableRetrievalInput("Shoes", (), "browsing"))))

    def test_browsing_route_calls_dense_only_and_ties_by_id(self) -> None:
        class DenseOnly:
            @staticmethod
            def ranked(query, depth):
                ids = np.asarray(["B", "A", "C"])
                scores = np.asarray([0.8, 0.8, 0.2], dtype=np.float32)
                order = np.lexsort((ids, -scores))
                return order[:depth], scores[order][:depth]

        fake = SimpleNamespace(dense=DenseOnly())
        order, _, diagnostic = routed_rank(fake, ObservableRetrievalInput("Shoes", ("wide",), "browsing"))
        self.assertEqual(list(order[:2]), [1, 0])
        self.assertFalse(diagnostic["exact_called"])
        self.assertFalse(diagnostic["bm25_called"])
        self.assertTrue(diagnostic["dense_called"])

    def test_robustness_constraints_drop_original_phrase_except_punctuation(self) -> None:
        values = ("cotton", "color: blue", "Machine Wash Cold", "Material:alloy")
        for transform in ROBUSTNESS_TRANSFORMS:
            for value in values:
                changed = transform_constraint(value, transform)
                if transform != "punctuation_changes":
                    self.assertNotIn(normalize(value), normalize(changed))
                else:
                    self.assertIn(normalize(value), normalize(changed))


class Experiment09Tests(unittest.TestCase):
    def test_typed_slots_strength_and_specificity(self) -> None:
        parser = TypedStateParser(("comfort", "fit"))
        first = parser.update("I'm looking for Shoes. A key requirement is: cotton.", 1)
        self.assertEqual(first.specificity, "specific")
        self.assertEqual(first.active_constraints[0].kind, "material")
        self.assertTrue(first.active_constraints[0].hard)
        self.assertEqual(first.active_constraints[0].strength, 1.0)
        second = parser.update("For that, what matters is: color: blue.", 2)
        self.assertEqual(second.active_constraints[-1].kind, "color")
        self.assertFalse(second.active_constraints[-1].hard)
        self.assertEqual(second.active_constraints[-1].strength, 0.7)
        self.assertEqual(second.profile_tags, ("comfort", "fit"))

        browsing = TypedStateParser()
        self.assertEqual(browsing.update("I'm looking for Shoes, but I'm still exploring.", 1).specificity, "exploratory")
        self.assertEqual(browsing.update("For that, what matters is: waterproof.", 2).specificity, "mixed")
        self.assertEqual(browsing.update("For that, what matters is: wide fit.", 3).specificity, "specific")

    def test_typed_negation_and_replacement(self) -> None:
        parser = TypedStateParser()
        parser.update("I'm looking for Shoes. I prefer red", 1)
        parser.update("Actually, ignore my earlier preference. What I need is: blue.", 2)
        values = [item.value for item in parser.state.active_constraints]
        self.assertEqual(values, ["blue"])
        self.assertEqual([item.value for item in parser.state.negations], ["I prefer red"])
        self.assertTrue(parser.state.active_constraints[0].hard)

    def test_classifier_order_and_weighted_rrf(self) -> None:
        self.assertEqual(classify_typed_constraint("blue cotton under $20"), "budget")
        ids = np.asarray(["B", "A", "C"])
        order, _ = weighted_rrf(
            {"exact": np.asarray([0, 1]), "dense": np.asarray([1, 0])},
            {"exact": 1.0, "dense": 1.0},
            ids,
        )
        self.assertEqual(list(ids[order]), ["A", "B"])
        dense_first, _ = weighted_rrf(
            {"exact": np.asarray([0, 1]), "dense": np.asarray([1, 0])},
            {"exact": 0.1, "dense": 1.0},
            ids,
        )
        self.assertEqual(ids[dense_first[0]], "A")

    def test_budget_violation_evidence(self) -> None:
        ranker = AdaptiveHybridRanker.__new__(AdaptiveHybridRanker)
        ranker.ids = np.asarray(["A", "B", "C"])
        ranker.prices = np.asarray([20.0, 50.0, np.nan])
        ranker._evidence_cache = {}
        np.testing.assert_array_equal(ranker.evidence("budget", "budget under $30"), [1.0, 0.0, 0.0])

    def test_recency_decay_and_explicit_over_profile_precedence(self) -> None:
        ranker = AdaptiveHybridRanker.__new__(AdaptiveHybridRanker)
        ranker.evidence = lambda kind, value: {
            "old": np.asarray([1.0, 0.0]),
            "new": np.asarray([0.0, 1.0]),
        }[value]
        state = TypedObservableState(
            turn=3,
            constraints=[
                TypedConstraint("old", "feature", False, 1, 1.0, False, False, "ambiguous_disclosure"),
                TypedConstraint("new", "feature", False, 3, 1.0, False, False, "ambiguous_disclosure"),
            ],
        )
        _, no_decay, _ = ranker._constraint_features(state, np.asarray([0, 1]), decay=False)
        _, decayed, _ = ranker._constraint_features(state, np.asarray([0, 1]), decay=True)
        np.testing.assert_allclose(no_decay, [0.5, 0.5])
        self.assertLess(decayed[0], decayed[1])

        # A profile match cannot compensate for an otherwise identical explicit
        # violation because +0.05 profile is dominated by the -0.30 penalty.
        scores = deterministic_reranker_score(
            *[np.asarray([1.0, 1.0]) for _ in range(5)],
            violations=np.asarray([0.0, 1.0]),
            profile=np.asarray([0.0, 1.0]),
        )
        self.assertGreater(scores[0], scores[1])

    def test_deterministic_reranker_ordering(self) -> None:
        scores = deterministic_reranker_score(
            hybrid=np.asarray([1.0, 0.8]),
            dense=np.asarray([0.5, 1.0]),
            hard=np.asarray([1.0, 0.0]),
            soft=np.asarray([0.0, 1.0]),
            category=np.asarray([1.0, 1.0]),
            violations=np.asarray([0.0, 1.0]),
        )
        self.assertGreater(scores[0], scores[1])

    def test_entropy_and_rank_aware_question_choice(self) -> None:
        products = [
            {"title": "red cotton running shirt", "store": "A", "price": 20, "features": ["washable"]},
            {"title": "blue wool hiking shirt", "store": "B", "price": 70, "features": ["warm"]},
            {"title": "red wool work shirt", "store": "C", "price": 35, "features": ["durable"]},
        ]
        fake = SimpleNamespace(products=products)
        entropy_attribute, _ = entropy_question(fake, [0, 1, 2], [])
        self.assertIn(entropy_attribute, {"material", "color", "brand", "budget", "feature", "use_case"})
        rank_attribute, diagnostic = rank_aware_question(fake, [0, 1, 2], [])
        self.assertNotEqual(rank_attribute, "other")
        self.assertGreater(diagnostic["expected_gain"], 0)

    def test_question_fallback_without_meaningful_values(self) -> None:
        fake = SimpleNamespace(products=[{"title": "plain item"}, {"title": "plain item"}])
        self.assertEqual(entropy_question(fake, [0, 1], [])[0], "other")
        self.assertEqual(rank_aware_question(fake, [0, 1], [])[0], "other")


if __name__ == "__main__":
    unittest.main()
