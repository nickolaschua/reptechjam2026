from __future__ import annotations

import importlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np


SHOPPING_AGENT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SHOPPING_AGENT_DIR))

import agent as agent_module
import agent_bge
import agent_openai
from compare_embeddings import calculate_retrieval_metrics
from embedding_backends import (
    BGE_EMBEDDING_SPACE_ID,
    BGE_QUERY_PREFIX,
    BGEEmbeddingBackend,
    CacheExpectation,
    CacheValidationError,
    OPENAI_EMBEDDING_SPACE_ID,
    OpenAIEmbeddingBackend,
    load_embedding_cache,
    save_embedding_cache,
)


class FakeSentenceModel:
    def __init__(self) -> None:
        self.calls = []

    def encode(self, inputs, **kwargs):
        self.calls.append((inputs, kwargs))
        if isinstance(inputs, list):
            return np.asarray([[3.0, 4.0] for _ in inputs], dtype=np.float32)
        return np.asarray([0.0, 5.0], dtype=np.float32)


class FakeEmbeddingsEndpoint:
    def __init__(self) -> None:
        self.calls = []

    def create(self, *, model, input):
        ordered_inputs = list(input)
        self.calls.append({"model": model, "input": ordered_inputs})
        data = [
            SimpleNamespace(index=index, embedding=[float(index + 1), 1.0])
            for index, _text in enumerate(ordered_inputs)
        ]
        # Reverse transport order; the backend must restore API index order.
        data.reverse()
        return SimpleNamespace(
            data=data,
            usage=SimpleNamespace(prompt_tokens=len(ordered_inputs) * 3, total_tokens=len(ordered_inputs) * 3),
        )


class ConstantBackend:
    def __init__(self, backend_id: str) -> None:
        self.backend_id = backend_id
        self.model_id = backend_id
        self.embedding_space_id = f"test:{backend_id}"

    def embed_query(self, text):
        return np.asarray([1.0, 0.0], dtype=np.float32)

    def usage_snapshot(self):
        return {"request_count": 0, "input_tokens": 0, "request_latencies_seconds": []}


class EmbeddingBakeoffTests(unittest.TestCase):
    def expectation(self, backend_id="bge-base-en-v1.5", ids=None):
        return CacheExpectation(
            backend_id=backend_id,
            model_id=(
                "BAAI/bge-base-en-v1.5"
                if backend_id == "bge-base-en-v1.5"
                else "text-embedding-3-large"
            ),
            embedding_space_id=(
                BGE_EMBEDDING_SPACE_ID
                if backend_id == "bge-base-en-v1.5"
                else OPENAI_EMBEDDING_SPACE_ID
            ),
            catalog_ids=ids or ["a", "b"],
            product_text_fingerprint="text-fingerprint",
            catalog_fingerprint="catalog-fingerprint",
        )

    def test_variant_modules_both_expose_agent(self):
        self.assertTrue(issubclass(agent_bge.Agent, agent_module.Agent))
        self.assertTrue(issubclass(agent_openai.Agent, agent_module.Agent))

    def test_variants_share_routing_state_query_and_ranking_implementation(self):
        self.assertIs(agent_bge.Agent.respond, agent_module.Agent.respond)
        self.assertIs(agent_openai.Agent.respond, agent_module.Agent.respond)
        self.assertIs(agent_bge.Agent._new_session_state, agent_module.Agent._new_session_state)
        self.assertIs(agent_openai.Agent._dense_retrieve, agent_module.Agent._dense_retrieve)
        state = agent_module.Agent._new_session_state()
        state.update({"category": "boots", "department": "shoes"})
        state["disclosed_slots"] = {"color": {"white"}}
        self.assertEqual(agent_module._state_to_retrieval_query(state), "boots shoes white")

    def test_bge_backend_normalizes_catalog_and_query_vectors(self):
        model = FakeSentenceModel()
        backend = BGEEmbeddingBackend(model=model, batch_size=2)
        catalog = backend.embed_catalog(["one", "two"])
        query = backend.embed_query("boots")
        np.testing.assert_allclose(np.linalg.norm(catalog, axis=1), np.ones(2))
        self.assertAlmostEqual(float(np.linalg.norm(query)), 1.0)
        self.assertEqual(model.calls[-1][0], BGE_QUERY_PREFIX + "boots")

    def test_openai_batches_preserves_order_normalizes_and_records_usage(self):
        endpoint = FakeEmbeddingsEndpoint()
        backend = OpenAIEmbeddingBackend(
            client=SimpleNamespace(embeddings=endpoint),
            batch_size=2,
        )
        vectors = backend.embed_catalog(["first", "second", "third"])
        self.assertEqual([call["input"] for call in endpoint.calls], [["first", "second"], ["third"]])
        np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), np.ones(3))
        # Within the first batch, API index zero must still be the first row.
        np.testing.assert_allclose(vectors[0], np.asarray([1.0, 1.0]) / np.sqrt(2.0))
        usage = backend.usage_snapshot()
        self.assertEqual(usage["request_count"], 2)
        self.assertEqual(usage["input_tokens"], 9)
        self.assertEqual(len(usage["request_latencies_seconds"]), 2)

    def test_bge_prefix_is_not_sent_to_openai(self):
        endpoint = FakeEmbeddingsEndpoint()
        backend = OpenAIEmbeddingBackend(client=SimpleNamespace(embeddings=endpoint))
        backend.embed_query("boots shoes white")
        self.assertEqual(endpoint.calls[0]["input"], ["boots shoes white"])
        self.assertNotIn("Represent this sentence", endpoint.calls[0]["input"][0])

    def test_cache_spaces_are_bidirectionally_separated(self):
        vectors = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.npz"
            bge = self.expectation()
            openai = self.expectation("openai-text-embedding-3-large")
            save_embedding_cache(path, vectors, bge)
            with self.assertRaises(CacheValidationError):
                load_embedding_cache(path, openai)
            save_embedding_cache(path, vectors, openai)
            with self.assertRaises(CacheValidationError):
                load_embedding_cache(path, bge)

    def test_cache_catalog_row_order_mismatch_is_rejected(self):
        vectors = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.npz"
            save_embedding_cache(path, vectors, self.expectation(ids=["a", "b"]))
            with self.assertRaisesRegex(CacheValidationError, "catalog_ids_exact_row_order"):
                load_embedding_cache(path, self.expectation(ids=["b", "a"]))

    def test_explicit_embedding_space_mismatch_is_rejected(self):
        vectors = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.npz"
            expectation = self.expectation("openai-text-embedding-3-large")
            save_embedding_cache(path, vectors, expectation)
            incompatible = replace(expectation, embedding_space_id="openai:other-space")
            with self.assertRaisesRegex(CacheValidationError, "embedding_space_id"):
                load_embedding_cache(path, incompatible)

    def test_valid_phase3_cache_without_space_id_is_reused(self):
        vectors = np.zeros((2, 3072), dtype=np.float32)
        vectors[0, 0] = 1.0
        vectors[1, 1] = 1.0
        expectation = self.expectation("openai-text-embedding-3-large")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.npz"
            save_embedding_cache(path, vectors, expectation)
            with np.load(path, allow_pickle=False) as data:
                metadata = json.loads(str(data["metadata_json"].item()))
                ids = np.asarray(data["ids"])
                cached_vectors = np.asarray(data["embeddings"])
            metadata["schema_version"] = 1
            metadata.pop("embedding_space_id")
            np.savez_compressed(
                path,
                embeddings=cached_vectors,
                ids=ids,
                metadata_json=np.asarray(json.dumps(metadata, sort_keys=True), dtype=np.str_),
            )
            loaded = load_embedding_cache(path, expectation)
            np.testing.assert_array_equal(loaded, vectors)

    def test_retrieval_metric_calculation_for_synthetic_ranks(self):
        metrics = calculate_retrieval_metrics([1, 10, 11, 50, 51, 150, 151, None])
        self.assertEqual(metrics["recall_at_10"], 2 / 8)
        self.assertEqual(metrics["recall_at_50"], 4 / 8)
        self.assertEqual(metrics["recall_at_150"], 6 / 8)
        self.assertAlmostEqual(
            metrics["mrr"],
            sum(1.0 / rank for rank in [1, 10, 11, 50, 51, 150, 151]) / 8,
        )

    def test_dense_instrumentation_does_not_change_ordering(self):
        matrix = np.asarray(
            [[0.1, 0.0], [0.9, 0.0], [0.5, 0.0]],
            dtype=np.float32,
        )
        expected = np.argsort(np.dot(matrix, np.asarray([1.0, 0.0])))[::-1][:3]
        for backend_id in ("bge", "openai"):
            instance = agent_module.Agent.__new__(agent_module.Agent)
            instance.embedding_backend = ConstantBackend(backend_id)
            instance.embedding_space_id = instance.embedding_backend.embedding_space_id
            instance.catalog_embeddings = matrix
            instance.catalog_ids = ["low", "high", "middle"]
            instance.instrumentation = {"semantic_queries": []}
            actual = instance._dense_retrieve("same canonical query", top_n=3)
            np.testing.assert_array_equal(actual, expected)
            self.assertEqual(len(instance.instrumentation["semantic_queries"]), 1)

    def test_openai_module_import_does_not_create_client_or_call_api(self):
        with patch("openai.OpenAI") as client_constructor:
            importlib.reload(agent_openai)
        client_constructor.assert_not_called()


if __name__ == "__main__":
    unittest.main()
    OPENAI_EMBEDDING_SPACE_ID,
