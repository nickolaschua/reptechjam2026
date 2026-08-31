"""Fast offline end-to-end check of the real 50,000-row demo path."""

from __future__ import annotations

import hashlib

import numpy as np

from system.shopping_agent.agent import Agent
from system.shopping_agent.config import (
    EMBEDDING_CACHE_DIR,
    EXPECTED_CATALOG_ROWS,
    OPENAI_EMBEDDING_DIMENSIONS,
    OPENAI_EMBEDDING_MODEL,
)
from system.shopping_agent.demo import DemoApplication
from system.shopping_agent.embedding_backends import OPENAI_EMBEDDING_SPACE_ID
from system.shopping_agent.memory_store import JsonFileVectorMemoryStore


class OfflineOpenAIQueryBackend:
    backend_id = "openai-text-embedding-3-large"
    model_id = OPENAI_EMBEDDING_MODEL
    embedding_space_id = OPENAI_EMBEDDING_SPACE_ID
    vector_dimension = OPENAI_EMBEDDING_DIMENSIONS

    def embed_catalog(self, texts):
        raise AssertionError("the smoke test must use the cached catalogue")

    def embed_query(self, text: str) -> np.ndarray:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        vector = np.zeros(self.vector_dimension, dtype=np.float32)
        for offset, value in enumerate(digest):
            vector[(offset * 97 + value) % self.vector_dimension] += (value + 1) / 256.0
        return vector / np.linalg.norm(vector)

    def usage_snapshot(self):
        return {"request_count": 0, "input_tokens": 0, "request_latencies_seconds": []}


def test_complete_demo_flow_uses_cache_persistence_isolation_and_reset(tmp_path):
    memory_path = tmp_path / "demo-memory.json"
    store = JsonFileVectorMemoryStore(memory_path)
    agent = Agent(
        embedding_backend=OfflineOpenAIQueryBackend(),
        embedding_cache_dir=EMBEDDING_CACHE_DIR,
        allow_catalog_embedding=False,
        memory_store=store,
    )
    agent._call_llm = lambda *args, **kwargs: "Here are the strongest catalogue matches."
    app = DemoApplication(agent=agent, store=store, top_k=3)

    assert len(agent.catalog_ids) == EXPECTED_CATALOG_ROWS
    assert agent.catalog_embeddings.shape == (
        EXPECTED_CATALOG_ROWS,
        OPENAI_EMBEDDING_DIMENSIONS,
    )

    app.start_session("user-a", mode="browsing")
    cold = app.send("I'm looking for dresses.")
    cold_trace = cold["debug"]["memory_trace"]
    assert cold["recommendations"]
    assert cold_trace["catalog_rows_scored"] == EXPECTED_CATALOG_ROWS
    assert cold_trace["buyer_mode"] == "browsing"
    assert cold_trace["prior_ltm_exists"] is False
    assert (cold_trace["a"], cold_trace["b"]) == (1.0, 0.0)
    app.send("For that, what matters is: floral.")
    committed = app.finish_session()
    assert committed["ltm_updated_after_turn"] is True
    assert committed["post_update_memory"]["update_count"] == 1

    reloaded = JsonFileVectorMemoryStore(memory_path)
    assert reloaded.get_state("user-a") is not None
    assert reloaded.get_state("user-b") is None

    app.start_session("user-a", mode="browsing")
    longitudinal = app.send("I'm looking for dresses.")
    trace = longitudinal["debug"]["memory_trace"]
    assert trace["prior_ltm_exists"] is True
    assert trace["memory_update_count"] == 1
    assert trace["gate_cosine"] is not None
    assert isinstance(trace["gate_passed"], bool)
    assert trace["eligible_count"] > 0
    assert longitudinal["recommendations"]
    app.finish_session()

    app.start_session("user-b", mode="buying")
    isolated = app.send("I'm looking for boots.")["debug"]["memory_trace"]
    assert isolated["prior_ltm_exists"] is False
    assert (isolated["a"], isolated["b"]) == (1.0, 0.0)
    app.finish_session()

    app.reset_user("user-a")
    assert app.inspect_user("user-a")["exists"] is False
    assert app.inspect_user("user-b")["user_id"] == "user-b"
