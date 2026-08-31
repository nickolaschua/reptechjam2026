from __future__ import annotations

import numpy as np
import pytest
from types import SimpleNamespace

import system.shopping_agent.agent as agent_module
from system.shopping_agent.agent import (
    Agent,
    ExperimentConfig,
    _apply_confidence_gate,
    _keyword_state_score,
)
from system.shopping_agent.memory_store import InMemoryVectorMemoryStore
from system.shopping_agent.vector_memory import DEFAULT_VECTOR_MEMORY_CONFIG
from system.shopping_agent.vector_memory import BuyerMode


def ranking_agent():
    agent = Agent.__new__(Agent)
    agent.experiment_config = ExperimentConfig()
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


def install_keyword_catalogue(agent, fts_rows):
    count = len(agent.catalog_ids)

    class KeywordCatalogue:
        row_by_asin = {}

        def eligibility(self, state):
            mask = np.ones(count, dtype=bool)
            return SimpleNamespace(mask=mask, hard_mask=mask, negative_mask=mask,
                                   hard_eligible_count=count, negative_filtered_count=0)

        def fts_route(self, terms, or_threshold):
            return SimpleNamespace(row_indices=tuple(fts_rows), and_count=count, or_count=count)

    agent.catalogue = KeywordCatalogue()


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"fts_position": 3}, -0.003),
        ({"department_category_match": True}, 20.0),
        ({"category_match": True}, 15.0),
        ({"brand": "other"}, -10.0),
        ({"accumulated_terms": ["rugged", "missing", "trail"]}, 0.6),
        ({"constraint": "all terrain"}, 10.0),
        ({"constraint": "rugged terrain trail"}, 5.0),
        ({"category": "trail boot"}, 2.0),
        ({"rating_number": 1}, 0.02),
    ],
)
def test_keyword_state_score_components(kwargs, expected):
    state = Agent._new_session_state()
    state["category"] = kwargs.get("category", "")
    state["accumulated_terms"] = kwargs.get("accumulated_terms", [])
    state["disclosed_slots"] = {}
    if "brand" in kwargs:
        state["disclosed_slots"]["brand"] = kwargs["brand"]
    if "constraint" in kwargs:
        state["disclosed_slots"]["style"] = kwargs["constraint"]
    score = _keyword_state_score(
        fts_position=kwargs.get("fts_position", 0),
        metadata={
            "brand": "target",
            "searchable_bag": "rugged all terrain trail boot",
            "rating_number": kwargs.get("rating_number", 0),
        },
        state=state,
        department_category_match=kwargs.get("department_category_match", False),
        category_match=kwargs.get("category_match", False),
    )
    assert score == pytest.approx(expected)


def test_keyword_state_score_exact_composite_formula():
    state = Agent._new_session_state()
    state.update({
        "category": "hiking boots",
        "accumulated_terms": ["waterproof", "missing"],
        "disclosed_slots": {
            "brand": "nike",
            "feature": ["waterproof", "all terrain"],
        },
    })
    score = _keyword_state_score(
        fts_position=7,
        metadata={
            "brand": "adidas",
            "searchable_bag": "waterproof all rugged terrain hiking boots",
            "rating_number": 1024,
        },
        state=state,
        department_category_match=True,
        category_match=True,
    )
    assert score == pytest.approx(
        -0.007 + 20.0 + 15.0 - 10.0 + 0.3 + 10.0 + 5.0 + 2.0 + 0.04
    )


def test_full_matrix_scoring_ties_excludes_previously_seen_results():
    agent = ranking_agent()
    agent.catalog_embeddings = np.asarray(
        [[1, 0], [1, 0], [1, 0], [1, 0]], dtype=np.float32
    )
    agent._sessions["s"]["seen_asins"] = {"a", "b"}
    result = agent._respond_custom("s", "show options", 1, 3)
    assert [item["parent_asin"] for item in result["recommendations"]] == ["c", "d"]
    trace = result["debug"]["memory_trace"]
    assert trace["catalog_rows_scored"] == 4
    assert trace["final_asins"] == ["c", "d"]
    assert trace["confidence_gate"]["seen_filter"] == {
        "previously_seen_count": 2,
        "ranked_rows_removed": 2,
        "unseen_ranked_count": 2,
    }


def test_repeated_unchanged_query_returns_a_fresh_page():
    agent = ranking_agent()
    agent.catalog_embeddings = np.asarray(
        [[1, 0], [1, 0], [1, 0], [1, 0]], dtype=np.float32
    )
    first = agent._respond_custom("s", "show options", 1, 2)
    second = agent._respond_custom("s", "show options", 2, 2)
    first_ids = [item["parent_asin"] for item in first["recommendations"]]
    second_ids = [item["parent_asin"] for item in second["recommendations"]]
    assert first_ids == ["a", "b"]
    assert second_ids == ["c", "d"]
    assert set(first_ids).isdisjoint(second_ids)


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
    assert [item["parent_asin"] for item in generic["recommendations"]] == ["a", "b"]
    assert generic["debug"]["memory_trace"]["negative_filtered_count"] == 0

    agent._sessions["s"] = Agent._new_session_state()
    agent._sessions["s"]["negated_terms"] = {"red"}
    specific = agent._respond_custom("s", "show options", 1, 10)
    assert [item["parent_asin"] for item in specific["recommendations"]] == ["b"]
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


def test_public_debug_flag_controls_terminal_trace_emission_independently():
    agent = ranking_agent()
    agent.instrumentation = {"semantic_queries": [], "agent_errors": [], "turns": []}
    observed = []

    def respond_custom(*args, emit_debug, **kwargs):
        observed.append(emit_debug)
        return {"debug": {}}

    agent._respond_custom = respond_custom
    agent.respond("s", "quiet", 1, 1)
    agent.respond("s", "visible", 2, 1, debug=True)
    result = agent.respond("s", "structured-only", 3, 1, debug=True, emit_trace=False)

    assert observed == [False, True, False]
    assert result["debug"] == {}


def test_terminal_telemetry_contains_legacy_state_and_modern_routing(capsys):
    agent = ranking_agent()
    agent._respond_custom("s", "show options", 1, 2, emit_debug=True)

    output = capsys.readouterr().out
    assert "[AGENT BRAIN TELEMETRY - CUSTOM HYBRID CASCADE ROUTE]" in output
    assert "Active LLM Model:" in output
    assert "FTS5 Matches:" in output
    assert "Category State:" in output
    assert "Disclosed Slots:" in output
    assert "Retrieval Route:" in output
    assert "Long-Term Memory:" in output
    assert "Confidence Gate:" in output
    assert "Final Products:" in output
    assert "[DEMO TRACE]" not in output


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


def _browsing_parser(agent):
    def detect_browsing(session_id, message):
        state = agent._sessions[session_id]
        state["intent_mode"] = "browsing"
        state["intent_source"] = "llm"
        state["_intent_detection_succeeded"] = True
    agent._update_state_via_llm = detect_browsing


def test_parser_intent_stands_and_no_second_detector_overrides_it():
    """One detector decides per turn: the parser when it ran, the local read when it did not."""

    agent = ranking_agent()
    _browsing_parser(agent)
    result = agent._respond_custom("s", "I need a specific boot", 1, 2, buyer_mode=BuyerMode.BUYING)
    trace = result["debug"]["memory_trace"]
    assert trace["intent_mode"] == "browsing"
    assert trace["intent_source"] == "winston_parser"
    assert trace["caller_buyer_mode"] == "buying"


def test_session_reset_outranks_the_parser():
    agent = ranking_agent()
    _browsing_parser(agent)
    agent._sessions["s"] = Agent._new_session_state()
    agent._sessions["s"]["intent_mode"] = "buying"
    result = agent._respond_custom("s", "start over", 1, 2, buyer_mode=BuyerMode.BUYING)
    trace = result["debug"]["memory_trace"]
    assert trace["intent_mode"] == "browsing"
    assert trace["intent_source"] == "session_reset"


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


def test_confidence_gate_boundary_order_pool_limit_and_no_backfill():
    ids = [f"p{i}" for i in range(12)]
    scores = np.asarray([0.39, 0.40, 0.75, 0.1, 0.2, 0.3, 0.1, 0.2, 0.3, 0.1, 0.99, 1.0])
    pool, survivors, trace = _apply_confidence_gate(list(range(12)), scores, ids)
    assert pool == list(range(10))
    assert survivors == [1, 2]
    assert [item["parent_asin"] for item in trace["evaluated_products"]] == ids[:10]
    assert [item["passed"] for item in trace["evaluated_products"][:3]] == [False, True, True]
    assert trace == {
        "threshold": 0.40,
        "pool_limit": 10,
        "evaluated_products": trace["evaluated_products"],
        "pass_count": 2,
        "reject_count": 8,
        "empty_result": False,
    }


def test_top_k_is_applied_after_confidence_filtering():
    ids = ["first", "rejected", "second", "third"]
    scores = np.asarray([0.9, 0.1, 0.8, 0.7])
    _, survivors, _ = _apply_confidence_gate([0, 1, 2, 3], scores, ids)
    assert [ids[row] for row in survivors[:2]] == ["first", "second"]


def test_empty_confidence_gate_calls_response_llm_and_does_not_mark_products_seen():
    agent = ranking_agent()
    agent.embed_dense_query = lambda text: np.array([0.3, 0.3], dtype=np.float32)
    prompts = []
    agent._call_llm = lambda prompt, system, **kwargs: prompts.append(system) or "Which style do you prefer?"
    result = agent._respond_custom("s", "show options", 1, 3)
    trace = result["debug"]["memory_trace"]
    assert result["recommendations"] == []
    assert trace["confidence_gate"]["empty_result"] is True
    assert trace["returned"] == trace["final_asins"] == []
    assert agent._sessions["s"]["seen_asins"] == set()
    assert prompts and "No product cleared" in prompts[0]


def test_fewer_than_ten_vector_fallback_candidates_are_all_evaluated():
    result = ranking_agent()._respond_custom("s", "show options", 1, 10)
    gate = result["debug"]["memory_trace"]["confidence_gate"]
    assert result["debug"]["memory_trace"]["retrieval_route"] == "vector_fallback"
    assert len(gate["evaluated_products"]) == 4
    assert (gate["pass_count"], gate["reject_count"]) == (2, 2)


def test_keyword_route_uses_gate_survivors_for_entropy_and_response(monkeypatch):
    agent = ranking_agent()
    count = 15
    agent.catalog_ids = [f"p{i:02d}" for i in range(count)]
    similarities = [1.0, 0.9, 0.9, 0.9, 0.9] + [0.1] * 6 + [0.9] + [0.1] * 3
    agent.catalog_embeddings = np.asarray(
        [[similarity, np.sqrt(1.0 - similarity**2)] for similarity in similarities],
        dtype=np.float32,
    )
    agent.catalog_prices = np.asarray([10.0] * count, dtype=np.float32)
    agent.catalog_metadata = {
        asin: {
            "title": asin,
            "brand": "x" if row < 4 else f"brand-{row}",
            "searchable_bag": asin,
            "rating_number": 0,
        }
        for row, asin in enumerate(agent.catalog_ids)
    }
    agent._sessions["s"]["seen_asins"] = {"p00"}
    install_keyword_catalogue(agent, range(count))
    entropy_inputs = []
    monkeypatch.setattr(
        agent_module,
        "select_best_attributes",
        lambda catalogue, products, remaining, **kwargs: entropy_inputs.append(list(products)) or ["style", "color"],
    )
    result = agent._respond_custom("s", "show options", 1, 3)
    trace = result["debug"]["memory_trace"]
    assert trace["retrieval_route"] == "keyword"
    assert trace["ranking_method"] == "keyword_state_score"
    assert len(trace["confidence_gate"]["evaluated_products"]) == 10
    assert [item["parent_asin"] for item in trace["confidence_gate"]["evaluated_products"]] == [
        f"p{row:02d}" for row in range(1, 11)
    ]
    assert trace["final_asins"] == ["p01", "p02", "p04"]
    assert "p11" not in trace["final_asins"]
    assert entropy_inputs == [["p01", "p02", "p04"]]
    assert set(trace["keyword_state_scores"]) == set(trace["final_asins"])
    assert all("keyword_state_score" in item for item in trace["returned"])
    assert all(asin in result["debug"]["system_prompt"] for asin in trace["final_asins"])


def test_keyword_state_evidence_outranks_higher_s3_and_ltm_cannot_rerank():
    agent = ranking_agent()
    count = 15
    agent.catalog_ids = ["weak-high-s3", "strong-state"] + [f"filler-{row}" for row in range(13)]
    agent.catalog_embeddings = np.asarray(
        [[0.8, 0.6], [1.0, 0.0]] + [[0.4, -np.sqrt(0.84)]] * 13,
        dtype=np.float32,
    )
    agent.catalog_prices = np.asarray([10.0] * count, dtype=np.float32)
    agent.catalog_metadata = {
        asin: {
            "title": asin,
            "brand": f"brand-{row}",
            "searchable_bag": "waterproof boot" if asin == "strong-state" else "plain boot",
            "rating_number": 0,
        }
        for row, asin in enumerate(agent.catalog_ids)
    }
    state = agent._sessions["s"]
    state["disclosed_slots"] = {"feature": "waterproof"}
    state["accumulated_terms"] = ["waterproof"]
    agent._active_lifecycle["s"]["visible_state"] = SimpleNamespace(
        vector=np.asarray([0.3, np.sqrt(0.91)], dtype=np.float32),
        update_count=1,
        embedding_space_id="space",
    )
    install_keyword_catalogue(agent, range(count))

    result = agent._respond_custom("s", "show options", 1, 2, buyer_mode=BuyerMode.BROWSING)
    trace = result["debug"]["memory_trace"]
    returned = {item["parent_asin"]: item for item in trace["returned"]}
    assert trace["gate_passed"] is True
    assert returned["weak-high-s3"]["s3"] > returned["strong-state"]["s3"]
    assert returned["strong-state"]["keyword_state_score"] > returned["weak-high-s3"]["keyword_state_score"]
    assert trace["final_asins"][:2] == ["strong-state", "weak-high-s3"]


def test_keyword_exact_score_ties_preserve_fts_order(monkeypatch):
    agent = ranking_agent()
    count = 15
    agent.catalog_ids = [f"p{row:02d}" for row in range(count)]
    agent.catalog_embeddings = np.asarray([[1.0, 0.0]] * count, dtype=np.float32)
    agent.catalog_prices = np.asarray([10.0] * count, dtype=np.float32)
    agent.catalog_metadata = {
        asin: {"title": asin, "brand": f"b{row}", "searchable_bag": asin, "rating_number": 0}
        for row, asin in enumerate(agent.catalog_ids)
    }
    fts_order = [7, 2, 10, 1, 0, 3, 4, 5, 6, 8, 9, 11, 12, 13, 14]
    install_keyword_catalogue(agent, fts_order)
    monkeypatch.setattr(agent_module, "_keyword_state_score", lambda **kwargs: 1.0)

    result = agent._respond_custom("s", "show options", 1, 3)
    trace = result["debug"]["memory_trace"]
    assert trace["final_asins"] == ["p07", "p02", "p10"]


def test_vector_fallback_keeps_descending_s3_with_asin_tie_breaking():
    result = ranking_agent()._respond_custom("s", "show options", 1, 2)
    trace = result["debug"]["memory_trace"]
    assert trace["retrieval_route"] == "vector_fallback"
    assert trace["ranking_method"] == "s3"
    assert trace["final_asins"] == ["a", "b"]
    assert trace["keyword_state_scores"] == {}
    assert all("keyword_state_score" not in item for item in trace["returned"])


def test_empty_gate_uses_rejected_pre_gate_pool_for_entropy(monkeypatch):
    agent = ranking_agent()
    agent.embed_dense_query = lambda text: np.array([0.3, 0.3], dtype=np.float32)
    count = len(agent.catalog_ids)

    class VectorCatalogue:
        row_by_asin = {}

        def eligibility(self, state):
            mask = np.ones(count, dtype=bool)
            return SimpleNamespace(mask=mask, hard_mask=mask, negative_mask=mask,
                                   hard_eligible_count=count, negative_filtered_count=0)

        def fts_route(self, terms, or_threshold):
            return SimpleNamespace(row_indices=(), and_count=0, or_count=0)

    agent.catalogue = VectorCatalogue()
    entropy_inputs = []
    monkeypatch.setattr(
        agent_module,
        "select_best_attributes",
        lambda catalogue, products, remaining, **kwargs: entropy_inputs.append(list(products)) or ["style", "color"],
    )
    result = agent._respond_custom("s", "show options", 1, 3)
    evaluated = result["debug"]["memory_trace"]["confidence_gate"]["evaluated_products"]
    assert result["recommendations"] == []
    assert entropy_inputs == [[item["parent_asin"] for item in evaluated]]
    assert "Input Candidate Products:\n\n" in result["debug"]["system_prompt"]


def test_long_term_memory_cannot_promote_a_low_s1_product_through_gate():
    agent = ranking_agent()
    agent.catalog_ids = ["current-match", "memory-promoted"]
    agent.catalog_embeddings = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    agent.catalog_prices = np.asarray([10.0, 10.0], dtype=np.float32)
    agent.catalog_metadata = {
        asin: {"title": asin, "brand": "x", "searchable_bag": asin, "rating_number": 0}
        for asin in agent.catalog_ids
    }
    memory = np.asarray([0.3, np.sqrt(0.91)], dtype=np.float32)
    agent._active_lifecycle["s"]["visible_state"] = SimpleNamespace(
        vector=memory, update_count=1, embedding_space_id="space"
    )
    result = agent._respond_custom("s", "show options", 1, 2, buyer_mode=BuyerMode.BROWSING)
    trace = result["debug"]["memory_trace"]
    assert trace["gate_passed"] is True
    assert trace["confidence_gate"]["evaluated_products"][0]["parent_asin"] == "memory-promoted"
    assert trace["confidence_gate"]["evaluated_products"][0]["passed"] is False
    assert trace["final_asins"] == ["current-match"]


@pytest.mark.parametrize("message", ["WA1200", "compatible with model ABC-123"])
def test_code_like_and_compatibility_text_receive_the_same_gate(message):
    result = ranking_agent()._respond_custom("s", message, 1, 10)
    gate = result["debug"]["memory_trace"]["confidence_gate"]
    assert (gate["pass_count"], gate["reject_count"]) == (2, 2)
