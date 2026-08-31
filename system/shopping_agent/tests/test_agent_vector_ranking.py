from __future__ import annotations

import numpy as np
import pytest

from system.shopping_agent.agent import Agent
from system.shopping_agent.memory_store import InMemoryVectorMemoryStore
from system.shopping_agent.vector_memory import DEFAULT_VECTOR_MEMORY_CONFIG
from system.shopping_agent.vector_memory import BuyerMode


def ranking_agent():
    agent = Agent.__new__(Agent)
    agent._sessions = {"s": Agent._new_session_state()}
    agent._active_lifecycle = {"s": {"user_id": None, "sequence_index": None, "visible_state": None}}
    agent._ended_lifecycle = {}
    agent._forensic_capture_sessions = set()
    agent._forensic_ranking_snapshots = {}
    agent._forensic_update_sessions = set()
    agent._forensic_update_vectors = {}
    agent.memory_store = InMemoryVectorMemoryStore()
    agent.embedding_space_id = "space"
    agent.vector_memory_config = DEFAULT_VECTOR_MEMORY_CONFIG
    agent.catalog_ids = ["b", "a", "c", "d"]
    agent.catalog_embeddings = np.asarray([[1, 0], [1, 0], [0, 1], [-1, 0]], dtype=np.float32)
    agent.catalog_prices = np.asarray([10, 10, 20, 100], dtype=np.float32)
    agent.catalog_metadata = {
        "b": {"title": "Blue boot", "brand": "x", "searchable_bag": "blue boot", "rating_number": 0},
        "a": {"title": "Red boot", "brand": "x", "searchable_bag": "red boot", "rating_number": 99999},
        "c": {"title": "Green shoe", "brand": "x", "searchable_bag": "green shoe", "rating_number": 0},
        "d": {"title": "Black coat", "brand": "x", "searchable_bag": "black coat", "rating_number": 0},
    }
    agent.embed_dense_query = lambda text: np.array([1.0, 0.0], dtype=np.float32)
    agent._update_state_via_llm = lambda session_id, message: None
    agent._call_llm = lambda *args, **kwargs: "Here are matches."
    return agent


def test_full_matrix_scoring_ties_and_no_popularity_or_seen_effects():
    agent = ranking_agent()
    agent._sessions["s"]["seen_asins"] = {"a", "b"}
    result = agent._respond_custom("s", "show options", 1, 3)
    assert [item["parent_asin"] for item in result["recommendations"]] == ["a", "b", "c"]
    trace = result["debug"]["memory_trace"]
    assert trace["catalog_rows_scored"] == 4
    assert trace["final_asins"] == ["a", "b", "c"]


def test_budget_and_negative_masks_may_return_fewer_than_top_k():
    agent = ranking_agent()
    state = agent._sessions["s"]
    state["price_max"] = 15
    state["negated_terms"] = {"red"}
    result = agent._respond_custom("s", "show options", 1, 10)
    assert [item["parent_asin"] for item in result["recommendations"]] == ["b"]
    trace = result["debug"]["memory_trace"]
    assert trace["price_filtered_count"] == 2
    assert trace["negative_filtered_count"] == 1


def test_compatibility_negative_mask_uses_catalogue_generic_exceptions():
    agent = ranking_agent()
    state = agent._sessions["s"]
    state["negated_terms"] = {"clothing", "shoes", "jewelry"}
    generic = agent._respond_custom("s", "show options", 1, 10)
    assert len(generic["recommendations"]) == 4
    assert generic["debug"]["memory_trace"]["negative_filtered_count"] == 0

    agent._sessions["s"] = Agent._new_session_state()
    agent._sessions["s"]["negated_terms"] = {"red"}
    specific = agent._respond_custom("s", "show options", 1, 10)
    assert [item["parent_asin"] for item in specific["recommendations"]] == ["b", "c", "d"]
    assert specific["debug"]["memory_trace"]["negative_filtered_count"] == 1


def test_zero_eligibility_never_returns_known_ineligible_products():
    agent = ranking_agent()
    agent._sessions["s"]["price_max"] = 0.0
    result = agent._respond_custom("s", "show options", 1, 10)
    assert result["recommendations"] == []
    assert result["debug"]["memory_trace"]["retrieval_route"] == "no_eligible"
    assert "relax" in result["message"].lower()


def test_buyer_mode_remains_validated_but_is_optional_with_prior_memory():
    agent = ranking_agent()
    agent.instrumentation = {"semantic_queries": [], "agent_errors": [], "turns": []}
    agent.memory_store.commit(user_id="u", session_id="prior", sequence_index=1,
                              embedding_space_id="space", new_preferences=np.array([1, 0], np.float32))
    agent._sessions = {}
    agent._active_lifecycle = {}
    agent.reset("s", {}, user_id="u", sequence_index=2)
    agent._respond_custom = lambda *args, **kwargs: {}
    assert agent.respond("s", "hello", 1, 1) == {}
    with pytest.raises(ValueError, match="exactly"):
        agent.respond("s", "hello", 1, 1, buyer_mode="BUYING")
    assert agent.respond("s", "hello", 1, 1, buyer_mode="buying") == {}


def test_live_intent_is_authoritative_and_exposes_mode_thresholds():
    agent = ranking_agent()
    browsing = agent._respond_custom("s", "show me options", 1, 2, buyer_mode=None)
    browsing_trace = browsing["debug"]["memory_trace"]
    assert browsing_trace["intent_mode"] == "browsing"
    assert (browsing_trace["fts_or_threshold"], browsing_trace["keyword_route_threshold"]) == (30, 15)

    agent._sessions["s"] = Agent._new_session_state()
    buying = agent._respond_custom("s", "I need blue boots", 1, 2, buyer_mode=None)
    buying_trace = buying["debug"]["memory_trace"]
    assert buying_trace["intent_mode"] == "buying"
    assert (buying_trace["fts_or_threshold"], buying_trace["keyword_route_threshold"]) == (15, 10)


def test_concrete_live_evidence_overrides_browsing_llm_and_caller_fallback():
    agent = ranking_agent()
    def detect_browsing(session_id, message):
        state = agent._sessions[session_id]
        state["intent_mode"] = "browsing"
        state["intent_source"] = "llm"
        state["_intent_detection_succeeded"] = True
    agent._update_state_via_llm = detect_browsing
    result = agent._respond_custom("s", "I need a specific boot", 1, 2, buyer_mode=BuyerMode.BUYING)
    trace = result["debug"]["memory_trace"]
    assert trace["intent_mode"] == "buying"
    assert trace["intent_source"] == "deterministic_precedence"
    assert trace["caller_buyer_mode"] == "buying"


def test_reset_freezes_prior_vector_snapshot_and_blocks_overlap():
    agent = ranking_agent()
    store = agent.memory_store
    store.commit(user_id="u", session_id="one", sequence_index=1,
                 embedding_space_id="space", new_preferences=np.array([1, 0], np.float32))
    agent._sessions = {}
    agent._active_lifecycle = {}
    agent.reset("two", {}, user_id="u", sequence_index=2)
    frozen = agent._active_lifecycle["two"]["visible_state"]
    with pytest.raises(ValueError, match="active session"):
        store.begin_session("u", "external", 2)
    np.testing.assert_array_equal(frozen.vector, [1, 0])


def test_opt_in_forensic_snapshot_has_both_arms_and_is_not_in_response_debug():
    agent = ranking_agent()
    agent._active_lifecycle["s"]["visible_state"] = None
    agent.enable_forensic_ranking("s")
    result = agent._respond_custom("s", "show options", 1, 2)
    snapshot = agent.get_forensic_ranking_snapshots("s")[-1]
    assert snapshot.s1.shape == snapshot.s3.shape == (4,)
    assert snapshot.m0_ranked_rows.tolist() == snapshot.m3_ranked_rows.tolist()
    assert snapshot.eligibility_mask.tolist() == [True] * 4
    with pytest.raises(ValueError):
        snapshot.s1[0] = 0
    assert "v1" not in result["debug"]["memory_trace"]
    assert "s1" not in result["debug"]["memory_trace"]
