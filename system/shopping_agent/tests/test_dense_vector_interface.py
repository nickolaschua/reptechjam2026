from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


SHOPPING_AGENT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SHOPPING_AGENT_DIR))

import agent as agent_module


class FixedQueryBackend:
    backend_id = "synthetic"
    model_id = "synthetic"
    embedding_space_id = "synthetic:3d"
    vector_dimension = 3

    def __init__(self, query: np.ndarray) -> None:
        self.query = np.asarray(query, dtype=np.float32)
        self.calls: list[str] = []

    def embed_query(self, text: str) -> np.ndarray:
        self.calls.append(str(text))
        return self.query.copy()

    def usage_snapshot(self) -> dict:
        return {"request_count": 0, "input_tokens": 0, "request_latencies_seconds": []}


class ExplodingMemory:
    def __getattr__(self, name: str):
        raise AssertionError(f"dense scorer accessed longitudinal memory attribute {name!r}")


class DenseVectorInterfaceTests(unittest.TestCase):
    def make_agent(self, query: np.ndarray | None = None) -> agent_module.Agent:
        instance = agent_module.Agent.__new__(agent_module.Agent)
        instance.catalog_embeddings = np.asarray(
            [
                [1.0, 0.0, 0.0],
                [0.8, 0.6, 0.0],
                [0.0, 1.0, 0.0],
                [-1.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
        instance.catalog_ids = ["p0", "p1", "p2", "p3"]
        instance.embedding_backend = FixedQueryBackend(
            np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
            if query is None
            else query
        )
        instance.embedding_space_id = instance.embedding_backend.embedding_space_id
        instance.instrumentation = {"semantic_queries": []}
        instance.memory_store = ExplodingMemory()
        instance.memory_adapter = ExplodingMemory()
        return instance

    def test_valid_q_returns_expected_ranking_and_explicit_alignment(self) -> None:
        instance = self.make_agent()

        result = instance.dense_retrieve_vector(
            np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
            top_n=3,
        )

        np.testing.assert_array_equal(result.row_indices, [0, 1, 2])
        self.assertEqual(result.product_ids, ("p0", "p1", "p2"))
        np.testing.assert_allclose(result.scores, [1.0, 0.8, 0.0])
        for position, row in enumerate(result.row_indices):
            self.assertEqual(result.product_ids[position], instance.catalog_ids[row])
            self.assertEqual(
                result.scores[position],
                np.dot(instance.catalog_embeddings[row], result.query_embedding),
            )
            np.testing.assert_array_equal(
                result.product_embeddings[position],
                instance.catalog_embeddings[row],
            )

    def test_query_shape_and_dimension_are_rejected(self) -> None:
        instance = self.make_agent()
        for invalid in (
            np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32),
            np.asarray([1.0, 0.0], dtype=np.float32),
        ):
            with self.subTest(shape=invalid.shape):
                with self.assertRaises(agent_module.DenseRetrievalError):
                    instance.dense_retrieve_vector(invalid)

    def test_non_finite_and_non_normalized_q_are_rejected(self) -> None:
        instance = self.make_agent()
        invalid_vectors = (
            np.asarray([np.nan, 0.0, 0.0], dtype=np.float32),
            np.asarray([np.inf, 0.0, 0.0], dtype=np.float32),
            np.asarray([2.0, 0.0, 0.0], dtype=np.float32),
            np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
        )
        for invalid in invalid_vectors:
            with self.subTest(vector=invalid):
                with self.assertRaises(agent_module.DenseRetrievalError):
                    instance.dense_retrieve_vector(invalid)

    def test_top_n_count_and_validation(self) -> None:
        instance = self.make_agent()
        query = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
        self.assertEqual(len(instance.dense_retrieve_vector(query, top_n=0).row_indices), 0)
        self.assertEqual(len(instance.dense_retrieve_vector(query, top_n=2).row_indices), 2)
        self.assertEqual(len(instance.dense_retrieve_vector(query, top_n=20).row_indices), 4)
        for invalid in (-1, 1.5, True):
            with self.subTest(top_n=invalid):
                with self.assertRaises(agent_module.DenseRetrievalError):
                    instance.dense_retrieve_vector(query, top_n=invalid)

    def test_boundary_dtype_and_inputs_and_catalogue_are_not_mutated(self) -> None:
        instance = self.make_agent()
        query = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
        query_before = query.copy()
        catalogue_before = instance.catalog_embeddings.copy()

        result = instance.dense_retrieve_vector(query, top_n=2)

        self.assertEqual(result.query_embedding.dtype, np.dtype(np.float32))
        self.assertEqual(result.product_embeddings.dtype, np.dtype(np.float32))
        np.testing.assert_array_equal(query, query_before)
        np.testing.assert_array_equal(instance.catalog_embeddings, catalogue_before)
        result.product_embeddings[0, 0] = 123.0
        np.testing.assert_array_equal(instance.catalog_embeddings, catalogue_before)

    def test_text_route_and_vector_route_have_exact_parity(self) -> None:
        instance = self.make_agent()
        query = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)

        text_result = instance.dense_retrieve_text("walking boots", top_n=4)
        vector_result = instance.dense_retrieve_vector(query, top_n=4)

        self.assertEqual(instance.embedding_backend.calls, ["walking boots"])
        np.testing.assert_array_equal(text_result.row_indices, vector_result.row_indices)
        self.assertEqual(text_result.product_ids, vector_result.product_ids)
        np.testing.assert_array_equal(text_result.scores, vector_result.scores)
        np.testing.assert_array_equal(
            text_result.product_embeddings,
            vector_result.product_embeddings,
        )
        np.testing.assert_array_equal(text_result.query_embedding, vector_result.query_embedding)
        self.assertEqual(len(instance.instrumentation["semantic_queries"]), 1)

    def test_legacy_text_wrapper_uses_the_vector_scorer(self) -> None:
        instance = self.make_agent()
        expected = instance.dense_retrieve_vector(
            np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
            top_n=2,
        ).row_indices

        actual = instance._dense_retrieve("same query", top_n=2)

        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(instance.embedding_backend.calls, ["same query"])

    def test_normalized_alternative_q_uses_the_same_scorer(self) -> None:
        instance = self.make_agent()
        q_alt = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)

        result = instance.dense_retrieve_vector(q_alt, top_n=3)

        # Rows 0 and 3 tie at zero; retain the current reversed-np.argsort order.
        np.testing.assert_array_equal(result.row_indices, [2, 1, 3])
        self.assertEqual(result.product_ids, ("p2", "p1", "p3"))
        self.assertEqual(instance.embedding_backend.calls, [])

    def test_one_frozen_q_is_reused_without_reembedding(self) -> None:
        instance = self.make_agent()
        query = instance.embed_dense_query("one current query")

        first = instance.dense_retrieve_vector(query, top_n=4)
        second = instance.dense_retrieve_vector(query, top_n=4)

        self.assertEqual(instance.embedding_backend.calls, ["one current query"])
        self.assertIs(first.query_embedding, query)
        self.assertIs(second.query_embedding, query)
        np.testing.assert_array_equal(first.row_indices, second.row_indices)
        np.testing.assert_array_equal(first.scores, second.scores)

    def test_snapshot_captures_replay_fields_without_target_leakage(self) -> None:
        instance = self.make_agent()

        snapshot = instance.freeze_dense_query(
            example_id="session-2-probe",
            raw_user_message="Something comfortable for wet weather",
            effective_query_text="boots shoes waterproof comfortable",
            target_product_id="evaluation-only-target",
            current_scope="footwear",
            current_category="boots",
        )
        result = instance.dense_retrieve_vector(snapshot.query_embedding, top_n=2)

        self.assertEqual(
            instance.embedding_backend.calls,
            ["boots shoes waterproof comfortable"],
        )
        self.assertEqual(snapshot.example_id, "session-2-probe")
        self.assertEqual(snapshot.target_product_id, "evaluation-only-target")
        self.assertEqual(snapshot.current_scope, "footwear")
        self.assertEqual(snapshot.current_category, "boots")
        self.assertIs(result.query_embedding, snapshot.query_embedding)

    def test_vector_scorer_is_memory_agnostic(self) -> None:
        instance = self.make_agent()
        result = instance.dense_retrieve_vector(
            np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
            top_n=1,
        )
        self.assertEqual(result.product_ids, ("p0",))


if __name__ == "__main__":
    unittest.main()
