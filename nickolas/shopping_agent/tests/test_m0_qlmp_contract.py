from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SHOPPING_AGENT_DIR = Path(__file__).resolve().parents[1]
for path in (PROJECT_ROOT, SHOPPING_AGENT_DIR):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)

import agent as agent_module
from embedding_backends import OPENAI_EMBEDDING_SPACE_ID
from memory_adapter import FastMemoryQLMPAdapter, MemoryDraft
from nickolas.memory.qlmp import (
    BaselineConfig,
    MemoryItem,
    MemoryPolarity,
    MemorySource,
    ProjectionConfig,
    SteeringConfig,
    bound_query_shift,
    build_cosine_memory_baseline,
    build_naive_memory_baseline,
)
from qlmp_integration import (
    CandidateMemoryBatch,
    CandidateUniverse,
    MemoryMode,
    ProjectionSteeringDeferredError,
    QLMPIntegrationConfig,
    QLMPIntegrationError,
    promote_local_product_rows,
    promote_q_work,
    run_projector_isolation,
    run_qlmp_integration,
)
from longitudinal_eval import qlmp_component_eval as component


DIMENSION = 3072


def normalize64(value: np.ndarray) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    return result / np.linalg.norm(result)


def axis(index: int, *, dtype=np.float32) -> np.ndarray:
    value = np.zeros(DIMENSION, dtype=dtype)
    value[index] = 1.0
    return value


def mixed(*pairs: tuple[int, float], dtype=np.float64) -> np.ndarray:
    value = np.zeros(DIMENSION, dtype=np.float64)
    for index, weight in pairs:
        value[index] = weight
    value = normalize64(value)
    return value.astype(dtype)


class CountingBackend:
    embedding_space_id = OPENAI_EMBEDDING_SPACE_ID

    def __init__(self, query: np.ndarray) -> None:
        self.query = np.asarray(query, dtype=np.float32)
        self.calls: list[str] = []

    def embed_query(self, text: str) -> np.ndarray:
        self.calls.append(str(text))
        return self.query.copy()

    def usage_snapshot(self) -> dict:
        return {"request_count": len(self.calls)}


def make_agent() -> agent_module.Agent:
    instance = agent_module.Agent.__new__(agent_module.Agent)
    rows = np.vstack(
        [
            axis(0),
            mixed((0, 0.98), (1, 0.2), dtype=np.float32),
            mixed((0, 0.96), (2, 0.28), dtype=np.float32),
            mixed((0, 0.94), (3, 0.34), dtype=np.float32),
            mixed((0, 0.92), (4, 0.39), dtype=np.float32),
            axis(1),
            axis(2),
            axis(3),
        ]
    ).astype(np.float32)
    instance.catalog_embeddings = rows
    instance.catalog_ids = [f"p{index}" for index in range(len(rows))]
    instance.embedding_space_id = OPENAI_EMBEDDING_SPACE_ID
    instance.embedding_backend = CountingBackend(axis(0))
    instance.instrumentation = {"semantic_queries": []}
    return instance


def memory(
    identifier: str,
    embedding: np.ndarray,
    *,
    scope: str | None = "footwear",
    polarity: MemoryPolarity = MemoryPolarity.POSITIVE,
) -> MemoryItem:
    return MemoryItem(
        id=identifier,
        text=f"memory text {identifier}",
        embedding=normalize64(embedding),
        source=MemorySource.USER,
        polarity=polarity,
        scope=scope,
        confidence=0.8,
    )


def batch(*items: MemoryItem, space: str = OPENAI_EMBEDDING_SPACE_ID) -> CandidateMemoryBatch:
    return CandidateMemoryBatch(items=tuple(items), embedding_space_id=space)


def config(
    mode: MemoryMode | str,
    *,
    baseline: BaselineConfig | None = None,
    steering: SteeringConfig | None = None,
    local_k: int = 5,
    rank: int = 3,
    universe: CandidateUniverse = CandidateUniverse.M0_FULL_CATALOGUE,
) -> QLMPIntegrationConfig:
    return QLMPIntegrationConfig(
        memory_mode=mode,
        embedding_space_id=OPENAI_EMBEDDING_SPACE_ID,
        embedding_dimension=DIMENSION,
        baseline=baseline or BaselineConfig(),
        steering=steering or SteeringConfig(),
        projection=ProjectionConfig(rank=rank),
        local_k=local_k,
        candidate_universe=universe,
    )


def assert_dense_equal(
    case: unittest.TestCase, left: agent_module.DenseRetrievalResult, right: agent_module.DenseRetrievalResult
) -> None:
    np.testing.assert_array_equal(left.query_embedding, right.query_embedding)
    np.testing.assert_array_equal(left.row_indices, right.row_indices)
    case.assertEqual(left.product_ids, right.product_ids)
    np.testing.assert_array_equal(left.scores, right.scores)
    np.testing.assert_array_equal(left.product_embeddings, right.product_embeddings)


class NumericBoundaryTests(unittest.TestCase):
    def test_float32_promotion_normalizes_owned_copies_without_mutation(self) -> None:
        q = mixed((0, 1.0), (1, 1.0), dtype=np.float32)
        products = np.vstack(
            [
                mixed((0, 1.0), (2, 1.0), dtype=np.float32),
                mixed((0, 1.0), (3, 1.0), dtype=np.float32),
            ]
        )
        q_before = q.copy()
        products_before = products.copy()

        q_work = promote_q_work(q, dimension=DIMENSION)
        product_work = promote_local_product_rows(products, dimension=DIMENSION)

        self.assertEqual(q_work.dtype, np.float64)
        self.assertEqual(product_work.dtype, np.float64)
        self.assertAlmostEqual(float(np.linalg.norm(q_work)), 1.0, places=14)
        np.testing.assert_allclose(np.linalg.norm(product_work, axis=1), 1.0, atol=1e-14)
        self.assertFalse(np.shares_memory(q_work, q))
        self.assertFalse(np.shares_memory(product_work, products))
        np.testing.assert_array_equal(q, q_before)
        np.testing.assert_array_equal(products, products_before)

    def test_memory_creation_renormalizes_float32_drift_for_qlmp(self) -> None:
        drifted = mixed((0, 1.0), (1, 1.0), dtype=np.float32)
        self.assertGreater(abs(float(np.linalg.norm(drifted.astype(np.float64))) - 1.0), 1e-8)
        backend = CountingBackend(drifted)
        adapter = FastMemoryQLMPAdapter(backend, OPENAI_EMBEDDING_SPACE_ID)
        draft = MemoryDraft(
            id="memory-drift",
            text="material: mesh",
            source=MemorySource.USER,
            polarity=MemoryPolarity.POSITIVE,
            scope="footwear",
        )

        item = adapter.embed_drafts([draft])[0]

        self.assertEqual(item.embedding.dtype, np.float64)
        self.assertAlmostEqual(float(np.linalg.norm(item.embedding)), 1.0, places=15)
        np.testing.assert_array_equal(drifted, backend.query)

    def test_q_star_float32_is_accepted_and_angle_remains_bounded(self) -> None:
        agent = make_agent()
        useful = memory("useful", mixed((0, 0.5), (1, 0.866)))
        settings = config(
            MemoryMode.NAIVE,
            steering=SteeringConfig(beta=20.0, max_shift_deg=7.0),
        )

        result = run_qlmp_integration(
            agent,
            q_m0=axis(0),
            candidate_memories=batch(useful),
            query_scope="footwear",
            top_n=5,
            config=settings,
        )

        self.assertEqual(result.q_final.dtype, np.float32)
        agent._validate_dense_query_embedding(result.q_final)
        self.assertLessEqual(result.angle64_deg, 7.0 + 1e-10)
        self.assertLessEqual(result.angle32_deg, 7.0 + 1e-4)


class IntegrationParityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.agent = make_agent()
        self.q = axis(0)
        self.useful = memory("useful", mixed((0, 0.8), (1, 0.6)))
        self.second = memory("second", mixed((0, 0.6), (2, 0.8)))

    def test_none_is_bitwise_direct_m0_and_does_not_inspect_memory(self) -> None:
        class Exploding:
            def __getattr__(self, name):
                raise AssertionError(f"none inspected {name}")

        direct = self.agent.dense_retrieve_vector(self.q, top_n=8)
        integrated = run_qlmp_integration(
            self.agent,
            q_m0=self.q,
            candidate_memories=Exploding(),  # type: ignore[arg-type]
            top_n=8,
            config=config(MemoryMode.NONE),
        )

        self.assertIs(integrated.q_m0, self.q)
        self.assertIs(integrated.q_final, self.q)
        self.assertIsNone(integrated.fallback_reason)
        assert_dense_equal(self, direct, integrated.final_dense_result)

    def test_no_usable_memory_cases_fall_back_to_exact_none(self) -> None:
        none = run_qlmp_integration(
            self.agent, q_m0=self.q, top_n=8, config=config(MemoryMode.NONE)
        )
        cases = [
            (MemoryMode.NAIVE, None, "no_memories"),
            (MemoryMode.COSINE, batch(), "no_memories"),
            (
                MemoryMode.NAIVE,
                batch(memory("negative", axis(1), polarity=MemoryPolarity.NEGATIVE)),
                "no_eligible_memories",
            ),
            (
                MemoryMode.NAIVE,
                batch(memory("wrong-scope", axis(1), scope="electronics")),
                "no_eligible_memories",
            ),
            (MemoryMode.COSINE, batch(memory("zero-cosine", axis(1))), "all_weights_zero"),
        ]
        for mode, memories, expected_reason in cases:
            with self.subTest(mode=mode, reason=expected_reason):
                result = run_qlmp_integration(
                    self.agent,
                    q_m0=self.q,
                    candidate_memories=memories,
                    query_scope="footwear",
                    top_n=8,
                    config=config(mode),
                )
                self.assertIs(result.q_final, self.q)
                self.assertEqual(result.fallback_reason, expected_reason)
                assert_dense_equal(self, none.final_dense_result, result.final_dense_result)

    def test_b1_matches_existing_standalone_baseline_and_steering(self) -> None:
        memories = batch(self.useful, self.second)
        settings = config(MemoryMode.NAIVE)
        q_work = promote_q_work(self.q, dimension=DIMENSION)
        expected_baseline = build_naive_memory_baseline(
            q_work,
            memories.items,
            query_scope="footwear",
            config=settings.baseline,
        )
        expected_steering = bound_query_shift(
            q_work, expected_baseline.aggregate_delta, config=settings.steering
        )

        actual = run_qlmp_integration(
            self.agent,
            q_m0=self.q,
            candidate_memories=memories,
            query_scope="footwear",
            top_n=8,
            config=settings,
        )

        self.assertEqual(actual.baseline_result.selected_memory_ids, ("useful", "second"))
        np.testing.assert_array_equal(
            actual.baseline_result.aggregate_delta, expected_baseline.aggregate_delta
        )
        np.testing.assert_array_equal(actual.steering_result.q_star, expected_steering.q_star)
        np.testing.assert_array_equal(
            actual.q_final, np.asarray(expected_steering.q_star, dtype=np.float32)
        )

    def test_b2_matches_existing_selection_weighting_and_steering(self) -> None:
        low = memory("low", mixed((0, 0.2), (3, np.sqrt(0.96))))
        memories = batch(low, self.second, self.useful)
        settings = config(
            MemoryMode.COSINE,
            baseline=BaselineConfig(memory_top_k=2, cosine_threshold=0.3),
        )
        q_work = promote_q_work(self.q, dimension=DIMENSION)
        expected_baseline = build_cosine_memory_baseline(
            q_work,
            memories.items,
            query_scope="footwear",
            config=settings.baseline,
        )
        expected_steering = bound_query_shift(
            q_work, expected_baseline.aggregate_delta, config=settings.steering
        )

        actual = run_qlmp_integration(
            self.agent,
            q_m0=self.q,
            candidate_memories=memories,
            query_scope="footwear",
            top_n=8,
            config=settings,
        )

        self.assertEqual(actual.baseline_result.selected_memory_ids, ("useful", "second"))
        np.testing.assert_array_equal(
            actual.baseline_result.aggregate_delta, expected_baseline.aggregate_delta
        )
        self.assertEqual(
            [value.aggregation_weight for value in actual.baseline_result.memory_diagnostics],
            [value.aggregation_weight for value in expected_baseline.memory_diagnostics],
        )
        np.testing.assert_array_equal(actual.steering_result.q_star, expected_steering.q_star)

    def test_b1_and_b2_both_call_the_shared_qlmp_steering_function(self) -> None:
        import qlmp_integration as integration_module

        with patch.object(
            integration_module,
            "bound_query_shift",
            wraps=bound_query_shift,
        ) as shared:
            for mode in (MemoryMode.NAIVE, MemoryMode.COSINE):
                run_qlmp_integration(
                    self.agent,
                    q_m0=self.q,
                    candidate_memories=batch(self.useful),
                    query_scope="footwear",
                    top_n=3,
                    config=config(mode),
                )
            self.assertEqual(shared.call_count, 2)

    def test_one_frozen_q_is_reused_across_every_mode_without_reembedding(self) -> None:
        snapshot = self.agent.freeze_dense_query(
            example_id="fixture-1",
            raw_user_message="comfortable footwear",
            effective_query_text="footwear comfortable",
            current_scope="footwear",
            current_category="shoes",
            user_id="u1",
            session_id="u1_s4",
        )
        memories = batch(self.useful)
        for mode in (MemoryMode.NONE, MemoryMode.NAIVE, MemoryMode.COSINE):
            run_qlmp_integration(
                self.agent,
                q_m0=snapshot.q_m0,
                candidate_memories=memories,
                query_scope=snapshot.query_scope,
                top_n=3,
                config=config(mode),
            )
        run_projector_isolation(
            self.agent,
            q_m0=snapshot.q_m0,
            candidate_memories=memories,
            config=config(MemoryMode.PROJECTION),
        )

        self.assertEqual(self.agent.embedding_backend.calls, ["footwear comfortable"])
        self.assertFalse(snapshot.q_m0.flags.writeable)
        self.assertEqual(snapshot.embedding_space_id, OPENAI_EMBEDDING_SPACE_ID)
        self.assertEqual((snapshot.user_id, snapshot.session_id), ("u1", "u1_s4"))

    def test_embedding_space_and_dimension_mismatch_are_invalid(self) -> None:
        wrong_space = batch(self.useful, space="other-space")
        with self.assertRaisesRegex(QLMPIntegrationError, "embedding_space_mismatch"):
            run_qlmp_integration(
                self.agent,
                q_m0=self.q,
                candidate_memories=wrong_space,
                config=config(MemoryMode.NAIVE),
            )
        with self.assertRaisesRegex(QLMPIntegrationError, "dimension_mismatch"):
            promote_q_work(np.asarray([1.0, 0.0], dtype=np.float32), dimension=DIMENSION)


class ProjectorIsolationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.agent = make_agent()
        self.q = axis(0)
        self.memories = batch(
            memory("supported", mixed((0, 0.7), (1, np.sqrt(0.51)))),
            memory(
                "other",
                mixed((0, 0.7), (7, np.sqrt(0.51))),
                scope="other",
            ),
        )

    def fixture(self, *, target: str = "target-a", labels=None) -> component.ProjectorFixture:
        snapshot = agent_module.DenseQuerySnapshot(
            example_id="fixture-projector",
            raw_user_message="current request",
            effective_query_text="frozen dense query",
            query_embedding=self.q,
            target_product_id=target,
            current_scope="footwear",
            current_category="shoes",
            user_id="u1",
            session_id="u1_s9",
            embedding_space_id=OPENAI_EMBEDDING_SPACE_ID,
        )
        return component.ProjectorFixture(
            snapshot=snapshot,
            candidate_memories=self.memories,
            labels_by_memory_id=labels
            or {
                "supported": component.ProjectorLabel.USEFUL_ADDITIONAL_STEERING,
                "other": component.ProjectorLabel.IRRELEVANT,
            },
            sequence_index=9,
            turn_index=2,
            scenario_type="buying",
        )

    def fixture_set(self, fixture: component.ProjectorFixture) -> component.ProjectorFixtureSet:
        return component.ProjectorFixtureSet(
            fixture_version=component.FIXTURE_VERSION,
            embedding_space_id=OPENAI_EMBEDDING_SPACE_ID,
            candidate_universe=CandidateUniverse.M0_FULL_CATALOGUE,
            fixtures=(fixture,),
        )

    def test_real_local_result_is_aligned_recorded_and_has_no_q_star(self) -> None:
        catalogue_before = self.agent.catalog_embeddings.copy()
        memory_before = tuple(item.embedding.copy() for item in self.memories.items)

        result = run_projector_isolation(
            self.agent,
            q_m0=self.q,
            candidate_memories=self.memories,
            config=config(MemoryMode.PROJECTION, local_k=5, rank=3),
        )

        self.assertEqual(result.requested_local_k, 5)
        self.assertEqual(result.local_subspace.local_product_count, 5)
        self.assertEqual(result.candidate_universe.value, "m0_full_catalogue")
        self.assertEqual(len(result.initial_dense_result.product_ids), 5)
        self.assertEqual(len(result.memory_projections), 2)
        self.assertFalse(hasattr(result, "q_final"))
        self.assertFalse(hasattr(result, "q_star"))
        np.testing.assert_array_equal(self.agent.catalog_embeddings, catalogue_before)
        for item, before in zip(self.memories.items, memory_before):
            np.testing.assert_array_equal(item.embedding, before)

    def test_projection_mode_cannot_be_used_for_retrieval_steering(self) -> None:
        with self.assertRaises(ProjectionSteeringDeferredError):
            run_qlmp_integration(
                self.agent,
                q_m0=self.q,
                candidate_memories=self.memories,
                config=config(MemoryMode.PROJECTION),
            )

    def test_unsupported_candidate_universe_fails_explicitly(self) -> None:
        with self.assertRaisesRegex(QLMPIntegrationError, "candidate_universe_unsupported"):
            run_projector_isolation(
                self.agent,
                q_m0=self.q,
                candidate_memories=self.memories,
                config=config(
                    MemoryMode.PROJECTION,
                    universe=CandidateUniverse.POST_CURRENT_HARD_FILTER,
                ),
            )

    def test_private_label_changes_do_not_change_any_projection_value(self) -> None:
        first = component.evaluate_projector_fixtures(
            self.agent,
            self.fixture_set(self.fixture()),
            config=config(MemoryMode.PROJECTION),
            bootstrap_samples=0,
        )
        changed = self.fixture(
            labels={
                "supported": component.ProjectorLabel.CROSS_DOMAIN_DISTRACTOR,
                "other": component.ProjectorLabel.USEFUL_ADDITIONAL_STEERING,
            }
        )
        second = component.evaluate_projector_fixtures(
            self.agent,
            self.fixture_set(changed),
            config=config(MemoryMode.PROJECTION),
            bootstrap_samples=0,
        )
        fields = ("memory_id", "raw_cosine", "tangent_norm", "rho", "projected_norm")
        self.assertEqual(
            [{key: row[key] for key in fields} for row in first.pair_records],
            [{key: row[key] for key in fields} for row in second.pair_records],
        )
        self.assertNotEqual(
            [row["label"] for row in first.pair_records],
            [row["label"] for row in second.pair_records],
        )

    def test_target_change_does_not_change_retrieval_or_projection(self) -> None:
        first = component.evaluate_projector_fixtures(
            self.agent,
            self.fixture_set(self.fixture(target="target-a")),
            config=config(MemoryMode.PROJECTION),
            bootstrap_samples=0,
        )
        second = component.evaluate_projector_fixtures(
            self.agent,
            self.fixture_set(self.fixture(target="target-b")),
            config=config(MemoryMode.PROJECTION),
            bootstrap_samples=0,
        )
        self.assertEqual(first.pair_records, second.pair_records)
        self.assertEqual(first.query_diagnostics, second.query_diagnostics)

    def test_labels_targets_and_private_profile_never_enter_qlmp_call(self) -> None:
        fixture = self.fixture()
        fixture_set = self.fixture_set(fixture)
        with patch.object(
            component,
            "run_projector_isolation",
            wraps=run_projector_isolation,
        ) as isolated:
            component.evaluate_projector_fixtures(
                self.agent,
                fixture_set,
                config=config(MemoryMode.PROJECTION),
                bootstrap_samples=0,
            )
        kwargs = isolated.call_args.kwargs
        self.assertEqual(
            set(kwargs), {"q_m0", "candidate_memories", "config"}
        )
        rendered = repr(kwargs)
        self.assertNotIn("target-a", rendered)
        self.assertNotIn("useful_additional_steering", rendered)
        self.assertNotIn("shopper_private_persona", rendered)
        self.assertNotIn("constant_profile", rendered)

    def test_candidate_universe_is_present_in_every_scientific_record(self) -> None:
        evaluation = component.evaluate_projector_fixtures(
            self.agent,
            self.fixture_set(self.fixture()),
            config=config(MemoryMode.PROJECTION),
            bootstrap_samples=0,
        )
        self.assertTrue(evaluation.pair_records)
        self.assertEqual(
            {value["candidate_universe"] for value in evaluation.pair_records},
            {"m0_full_catalogue"},
        )
        self.assertEqual(
            {value["candidate_universe"] for value in evaluation.query_diagnostics},
            {"m0_full_catalogue"},
        )


class ProjectorHarnessTests(unittest.TestCase):
    @staticmethod
    def record(
        fixture_id: str,
        label: str,
        cosine: float,
        rho: float,
        projected: float,
        *,
        user: str = "u1",
        category: str = "shoes",
    ) -> dict:
        return {
            "fixture_id": fixture_id,
            "user_id": user,
            "current_category": category,
            "label": label,
            "raw_cosine": cosine,
            "rho": rho,
            "projected_norm": projected,
        }

    def test_metrics_keep_redundant_separate_and_projection_can_beat_cosine(self) -> None:
        rows = [
            self.record("q1", "useful_additional_steering", 0.3, 0.9, 0.8),
            self.record("q1", "irrelevant", 0.8, 0.1, 0.1),
            self.record("q1", "same_category_hard_negative", 0.7, 0.2, 0.2),
            self.record("q2", "useful_additional_steering", 0.4, 0.8, 0.7),
            self.record("q2", "cross_domain_distractor", 0.1, 0.0, 0.05),
            self.record("q2", "relevant_but_redundant", 0.99, 0.01, 0.001),
        ]

        summary = component.summarize_projector_records(
            rows, bootstrap_samples=20, bootstrap_seed=7
        )

        self.assertEqual(summary["primary_binary"]["counts"], {"positive": 2, "negative": 3})
        self.assertEqual(summary["label_counts"]["relevant_but_redundant"], 1)
        self.assertEqual(summary["primary_binary"]["scores"]["rho"]["auroc"], 1.0)
        self.assertLess(
            summary["primary_binary"]["scores"]["raw_cosine"]["auroc"], 1.0
        )
        self.assertTrue(summary["query_bootstrap_95_ci"]["available"])
        self.assertEqual(summary["empirical_null"]["rho"]["count"], 3)

    def test_artifacts_are_run_specific_machine_readable_and_not_overwritten(self) -> None:
        evaluation = component.ProjectorEvaluation(
            pair_records=(
                self.record("q1", "irrelevant", 0.1, 0.2, 0.3),
            ),
            query_diagnostics=(),
            summary={"decision": {"verdict": "PROJECTOR INCONCLUSIVE"}},
        )
        manifest = {
            "candidate_universe": "m0_full_catalogue",
            "b3_retrieval_steering_enabled": False,
        }
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = component.write_projector_artifacts(
                temporary, "run-1", evaluation, manifest
            )
            self.assertTrue((run_dir / "projector_pairs.jsonl").is_file())
            self.assertTrue((run_dir / "projector_pairs.csv").is_file())
            self.assertTrue((run_dir / "summary.json").is_file())
            loaded_manifest = json.loads((run_dir / "run_manifest.json").read_text())
            self.assertFalse(loaded_manifest["b3_retrieval_steering_enabled"])
            with self.assertRaises(FileExistsError):
                component.write_projector_artifacts(
                    temporary, "run-1", evaluation, manifest
                )

    def test_loader_refuses_missing_q_or_memory_embeddings(self) -> None:
        base = {
            "fixture_version": component.FIXTURE_VERSION,
            "embedding_space_id": OPENAI_EMBEDDING_SPACE_ID,
            "candidate_universe": "m0_full_catalogue",
            "fixtures": [
                {
                    "fixture_id": "q1",
                    "current_message": "request",
                    "effective_query_text": "query",
                    "q_m0": axis(0).tolist(),
                    "memories": [
                        {
                            "id": "m1",
                            "text": "memory",
                            "embedding": axis(1, dtype=np.float64).tolist(),
                            "label": "irrelevant",
                        }
                    ],
                }
            ],
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "fixture.json"
            missing_q = json.loads(json.dumps(base))
            del missing_q["fixtures"][0]["q_m0"]
            path.write_text(json.dumps(missing_q), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "persist exact q_m0"):
                component.load_projector_fixture(path)

            missing_memory = json.loads(json.dumps(base))
            del missing_memory["fixtures"][0]["memories"][0]["embedding"]
            path.write_text(json.dumps(missing_memory), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "persist an embedding"):
                component.load_projector_fixture(path)


if __name__ == "__main__":
    unittest.main()
