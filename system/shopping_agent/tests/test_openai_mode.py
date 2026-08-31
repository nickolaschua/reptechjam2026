from __future__ import annotations

import json
import numpy as np
import pytest

from system.shopping_agent.config import RuntimeConfig, memory_store_path
from system.shopping_agent.agent import Agent
from system.shopping_agent.embedding_backends import (
    BGEEmbeddingBackend, CatalogCacheMissError, OpenAIEmbeddingBackend, OpenAIEmbeddingResponseError,
    cache_filename,
)
from system.shopping_agent.ollama_client import OllamaClient
from system.shopping_agent.openai_client import OpenAIClient, OpenAIResponseError
from system.shopping_agent.runtime import create_runtime_providers


def config(test_mode: bool) -> RuntimeConfig:
    return RuntimeConfig(test_mode, "openai" if test_mode else "ollama", "test-key" if test_mode else None,
        "gpt-4o-mini", "text-embedding-3-small", 1536, 7,
        "http://localhost:11434", "llama3.1:8b", 8, False)


def test_factory_selects_both_providers_without_network():
    local = create_runtime_providers(config(False))
    hosted = create_runtime_providers(config(True))
    assert isinstance(local.llm_client, OllamaClient)
    assert isinstance(local.embedding_backend, BGEEmbeddingBackend)
    assert isinstance(hosted.llm_client, OpenAIClient)
    assert isinstance(hosted.embedding_backend, OpenAIEmbeddingBackend)
    assert local.llm_client.instrumentation() == hosted.llm_client.instrumentation() == []


def test_openai_response_payload_structured_output_retry_and_usage():
    payloads = []
    def transport(_url, body, _timeout, headers):
        payloads.append(json.loads(body))
        assert headers["Authorization"] == "Bearer test-key"
        if len(payloads) == 1: return {"output": []}
        return {"model": "gpt-4o-mini", "output_text": '{"ok":true}',
                "usage": {"input_tokens": 2, "output_tokens": 3, "total_tokens": 5}}
    client = OpenAIClient(api_key="test-key", transport=transport)
    call = client.chat_result([{"role": "user", "content": "hello"}], format="json",
                              options={"temperature": 0, "num_predict": 99}, role="assistant")
    assert call.retry_count == 1 and call.usage["total_tokens"] == 5
    assert payloads[0]["max_output_tokens"] == 99
    assert payloads[0]["text"]["format"]["type"] == "json_object"


def test_openai_invalid_output_is_typed():
    client = OpenAIClient(api_key="test-key", transport=lambda *_: {"output": []})
    with pytest.raises(OpenAIResponseError):
        client.chat([{"role": "user", "content": "hello"}])


def test_openai_embeddings_batch_order_normalization_usage_and_identity():
    calls = []
    def transport(_url, body, _timeout, _headers):
        request = json.loads(body); calls.append(request)
        data = [{"index": i, "embedding": [1.0] + [0.0] * 1535}
                for i in range(len(request["input"]))]
        return {"data": data, "usage": {"total_tokens": len(data)}}
    backend = OpenAIEmbeddingBackend(api_key="test-key", batch_size=2, transport=transport)
    matrix = backend.embed_catalog(["a", "b", "c"])
    assert matrix.shape == (3, 1536) and np.allclose(np.linalg.norm(matrix, axis=1), 1)
    assert [len(call["input"]) for call in calls] == [2, 1]
    assert backend.usage_snapshot()["input_tokens"] == 3
    assert cache_filename(backend.backend_id, backend.model_id, backend.vector_dimension) == (
        "catalog_cache_openai-text-embedding-3-small-d1536.npz")


def test_openai_embeddings_reject_reordered_rows_and_memory_isolation():
    def transport(*_):
        return {"data": [{"index": 1, "embedding": [0.0] * 1536}]}
    backend = OpenAIEmbeddingBackend(api_key="test-key", transport=transport)
    with pytest.raises(OpenAIEmbeddingResponseError, match="row order"):
        backend.embed_query("x")
    assert memory_store_path(config(False)).name == "vector_memory.json"
    assert memory_store_path(config(True)).name == (
        "vector_memory_openai-text-embedding-3-small-d1536.json")


def test_normal_openai_startup_never_builds_missing_cache(tmp_path):
    catalog = tmp_path / "catalog.jsonl"; catalog.write_text("{}\n", encoding="utf-8")
    backend = OpenAIEmbeddingBackend(api_key="test-key", transport=lambda *_: pytest.fail("network"))
    agent = Agent.__new__(Agent)
    agent.catalog_path = catalog; agent.catalog_products = [{"parent_asin": "a"}]
    agent.catalog_ids = ["a"]; agent.embedding_backend = backend
    agent.embedding_backend_id = backend.backend_id; agent.embedding_model_id = backend.model_id
    agent.embedding_space_id = backend.embedding_space_id; agent.embedding_cache_dir = tmp_path / "cache"
    agent.allow_catalog_embedding = True; agent.explicit_cache_build = False
    agent.instrumentation = {"initialization": {"product_text_build_seconds": 0,
        "cache_status": "not-attempted", "embedding_cache_load_seconds": 0,
        "catalog_embedding_generation_seconds": 0}}
    with pytest.raises(CatalogCacheMissError, match="build_embedding_cache"):
        agent._build_vector_index()
