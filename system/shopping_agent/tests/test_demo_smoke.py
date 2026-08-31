"""Fast offline end-to-end check of the real 50,000-row demo path."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from system.shopping_agent.agent import Agent
from system.shopping_agent.config import (
    EXPECTED_CATALOG_ROWS,
)
from system.shopping_agent.demo import DemoApplication
from system.shopping_agent.embedding_backends import (
    BGE_EMBEDDING_SPACE_ID,
    BGE_MODEL,
)
from system.shopping_agent.memory_store import JsonFileVectorMemoryStore


class OfflineBGEBackend:
    backend_id = "bge-base-en-v1.5"
    model_id = BGE_MODEL
    embedding_space_id = BGE_EMBEDDING_SPACE_ID
    vector_dimension = 768

    def embed_catalog(self, texts):
        matrix = np.zeros((len(texts), self.vector_dimension), dtype=np.float32)
        for row, text in enumerate(texts):
            normalized = text.lower()
            if "dress" in normalized:
                matrix[row, 0] = 1.0
                continue
            if "boot" in normalized:
                matrix[row, 1] = 1.0
                continue
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            matrix[row, int.from_bytes(digest[:2], "big") % self.vector_dimension] = 1.0
        return matrix

    def embed_query(self, text: str) -> np.ndarray:
        normalized = text.lower()
        if "dress" in normalized:
            vector = np.zeros(self.vector_dimension, dtype=np.float32)
            vector[0] = 1.0
            return vector
        if "boot" in normalized:
            vector = np.zeros(self.vector_dimension, dtype=np.float32)
            vector[1] = 1.0
            return vector
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
        embedding_backend=OfflineBGEBackend(),
        embedding_cache_dir=tmp_path / "cache",
        allow_catalog_embedding=True,
        memory_store=store,
    )
    agent._call_llm = lambda *args, **kwargs: "Here are the strongest catalogue matches."
    app = DemoApplication(agent=agent, store=store, top_k=3)

    assert len(agent.catalog_ids) == EXPECTED_CATALOG_ROWS
    assert agent.catalog_embeddings.shape == (
        EXPECTED_CATALOG_ROWS,
        768,
    )
    query = agent.embedding_backend.embed_query(agent.catalog_texts[0])
    cosine = float(agent.catalog_embeddings[0] @ query)
    assert np.isfinite(cosine) and -1.00001 <= cosine <= 1.00001

    cache_digest = hashlib.sha256(agent.embedding_cache_path.read_bytes()).hexdigest()
    manifest_path = tmp_path / "bge_artifact_manifest.json"
    manifest_path.write_text(
        json.dumps({"files": {agent.embedding_cache_path.name: cache_digest}}),
        encoding="utf-8",
    )
    repository_root = Path(__file__).resolve().parents[3]
    verified = subprocess.run(
        [
            sys.executable,
            str(repository_root / "colab" / "verify_bge_artifact.py"),
            str(agent.embedding_cache_path),
            "--repo-root",
            str(repository_root),
            "--manifest",
            str(manifest_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(verified.stdout)["status"] == "valid"

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
