from __future__ import annotations

import numpy as np
import pytest

from system.shopping_agent.memory_store import (
    InMemoryVectorMemoryStore,
    JsonFileVectorMemoryStore,
    MemoryStoreSnapshot,
)
from system.shopping_agent.vector_memory import (
    BuyerMode,
    VectorMemoryConfig,
    positive_slot_text,
    score_catalog,
)


def unit(values):
    vector = np.asarray(values, dtype=np.float32)
    return vector / np.linalg.norm(vector)


@pytest.mark.parametrize(
    ("cosine", "passed"), [(0.199, False), (0.20, True), (0.201, True)]
)
def test_relevance_gate_boundary(cosine, passed):
    v1 = np.array([1.0, 0.0], dtype=np.float32)
    v2 = unit([cosine, np.sqrt(1.0 - cosine**2)])
    matrix = np.eye(2, dtype=np.float32)
    _, _, _, actual, gate_passed, _, _ = score_catalog(
        matrix, v1, v2, BuyerMode.BUYING
    )
    assert actual == pytest.approx(cosine, abs=1e-6)
    assert gate_passed is passed


def test_cold_start_and_exact_mode_equations():
    matrix = np.asarray([[1, 0], [0, 1], [2**-0.5, 2**-0.5]], dtype=np.float32)
    v1, v2 = np.asarray([1, 0], dtype=np.float32), np.asarray([0.6, 0.8], dtype=np.float32)
    s1, s2, cold, gate, passed, a, b = score_catalog(matrix, v1, None, None)
    assert s2 is None and gate is None and not passed and (a, b) == (1.0, 0.0)
    np.testing.assert_allclose(cold, s1)
    for mode, weights in ((BuyerMode.BUYING, (0.8, 0.2)), (BuyerMode.BROWSING, (0.2, 0.8))):
        s1, s2, s3, _, passed, a, b = score_catalog(matrix, v1, v2, mode)
        assert passed and (a, b) == weights
        np.testing.assert_allclose(s3, a * s1 + b * s2)


def test_gate_failure_ignores_mode_weights():
    matrix = np.eye(2, dtype=np.float32)
    v1, v2 = np.array([1, 0], dtype=np.float32), np.array([0, 1], dtype=np.float32)
    s1, s2, s3, _, passed, a, b = score_catalog(matrix, v1, v2, BuyerMode.BROWSING)
    assert s2 is not None and not passed and (a, b) == (1.0, 0.0)
    np.testing.assert_array_equal(s3, s1)


def test_positive_slot_serialization_is_sorted_and_exclusive():
    state = {
        "style": {"casual"}, "color": {"yes"}, "feature": {"no", "waterproof"}
    }
    assert positive_slot_text(state) == "color: color; feature: waterproof; style: casual"


def test_store_cold_start_ewma_empty_commit_and_isolation():
    store = InMemoryVectorMemoryStore()
    first = store.commit(user_id="u1", session_id="s1", sequence_index=1,
                         embedding_space_id="space", new_preferences=np.array([1, 0], np.float32))
    assert first.vector_changed and first.state.update_count == 1
    second = store.commit(user_id="u1", session_id="s2", sequence_index=2,
                          embedding_space_id="space", new_preferences=np.array([0, 1], np.float32))
    np.testing.assert_allclose(second.state.vector, unit([0.7, 0.3]), atol=1e-6)
    empty = store.commit(user_id="u1", session_id="s3", sequence_index=3,
                         embedding_space_id="space", new_preferences=None)
    assert not empty.vector_changed and empty.state.last_committed_sequence == 3
    np.testing.assert_allclose(empty.state.vector, second.state.vector)
    assert store.get_state("u2") is None


def test_store_chronology_space_and_snapshot_contracts():
    store = InMemoryVectorMemoryStore()
    store.commit(user_id="u", session_id="s", sequence_index=4,
                 embedding_space_id="space", new_preferences=np.array([1, 0], np.float32))
    with pytest.raises(ValueError, match="out-of-order"):
        store.validate_new_session("u", "later", 4)
    with pytest.raises(ValueError, match="different embedding space"):
        store.commit(user_id="u", session_id="later", sequence_index=5,
                     embedding_space_id="other", new_preferences=np.array([1, 0], np.float32))
    payload = store.export_snapshot().to_payload()
    restored_snapshot = MemoryStoreSnapshot.from_payload(payload)
    restored = InMemoryVectorMemoryStore()
    restored.import_snapshot(restored_snapshot, expected_embedding_space_id="space")
    np.testing.assert_array_equal(restored.get_state("u").vector, store.get_state("u").vector)
    with pytest.raises(ValueError, match="QLMP snapshot version 1"):
        MemoryStoreSnapshot.from_payload({"version": 1, "records": [], "commits": []})


def test_snapshot_and_state_vectors_are_immutable():
    source = np.array([1, 0], np.float32)
    store = InMemoryVectorMemoryStore()
    commit = store.commit(user_id="u", session_id="s", sequence_index=1,
                          embedding_space_id="space", new_preferences=source)
    source[:] = [0, 1]
    np.testing.assert_array_equal(commit.state.vector, [1, 0])
    with pytest.raises(ValueError):
        commit.state.vector[0] = 0


def test_filtered_snapshot_restores_prefix_vector():
    store = InMemoryVectorMemoryStore()
    store.commit(user_id="u", session_id="s1", sequence_index=1,
                 embedding_space_id="space", new_preferences=np.array([1, 0], np.float32))
    store.commit(user_id="u", session_id="s2", sequence_index=2,
                 embedding_space_id="space", new_preferences=np.array([0, 1], np.float32))
    prefix = store.export_snapshot().filtered(user_id="u", sequence_indices=[1], before_sequence_index=2)
    restored = InMemoryVectorMemoryStore()
    restored.import_snapshot(prefix)
    np.testing.assert_array_equal(restored.get_state("u").vector, [1, 0])


def test_failed_snapshot_import_leaves_store_empty():
    store = InMemoryVectorMemoryStore()
    broken = MemoryStoreSnapshot.from_payload({
        "version": 2, "kind": "gated-vector-memory",
        "states": [{"user_id": "u", "vector": [1, 0], "embedding_space_id": "space",
                    "last_committed_sequence": 2, "update_count": 1}],
        "commits": [{"user_id": "u", "session_id": "s", "sequence_index": 1,
                     "embedding_space_id": "space", "vector_changed": True}],
    })
    with pytest.raises(ValueError, match="chronology"):
        store.import_snapshot(broken)
    assert store.export_snapshot().states == ()
    assert store.export_snapshot().commits == ()


def test_config_defaults_are_frozen_contract():
    config = VectorMemoryConfig()
    assert (config.relevance_threshold, config.buying_current_weight,
            config.buying_memory_weight, config.browsing_current_weight,
            config.browsing_memory_weight, config.ewma_alpha) == (0.20, 0.80, 0.20, 0.20, 0.80, 0.30)


def test_json_store_persists_isolates_and_resets_users(tmp_path):
    path = tmp_path / "memory.json"
    store = JsonFileVectorMemoryStore(path)
    store.commit(user_id="a", session_id="a1", sequence_index=0,
                 embedding_space_id="space", new_preferences=np.array([1, 0], np.float32))
    store.commit(user_id="b", session_id="b1", sequence_index=0,
                 embedding_space_id="space", new_preferences=np.array([0, 1], np.float32))

    restored = JsonFileVectorMemoryStore(path)
    np.testing.assert_array_equal(restored.get_state("a").vector, [1, 0])
    np.testing.assert_array_equal(restored.get_state("b").vector, [0, 1])
    assert restored.describe_user("a")["update_count"] == 1
    assert restored.next_sequence_index("a") == 1

    restored.clear_user("a")
    reloaded = JsonFileVectorMemoryStore(path)
    assert reloaded.get_state("a") is None
    assert reloaded.get_state("b") is not None
    reloaded.clear()
    assert JsonFileVectorMemoryStore(path).export_snapshot().states == ()
