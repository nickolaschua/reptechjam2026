from copy import deepcopy
import json

import pytest

from system.shopping_agent.agent import Agent, _state_to_retrieval_query
from system.shopping_agent.memory_store import InMemoryVectorMemoryStore


def bare_agent():
    agent = Agent.__new__(Agent)
    agent._sessions = {}
    agent._active_lifecycle = {}
    agent._ended_lifecycle = {}
    agent.memory_store = InMemoryVectorMemoryStore()
    agent.embedding_space_id = "test-space"
    return agent


def test_new_session_state_is_not_shared():
    agent = bare_agent()
    agent.reset("a", {})
    agent.reset("b", {})
    agent._sessions["a"]["disclosed_slots"]["color"] = {"black"}
    agent._sessions["a"]["history"].append({"role": "user", "content": "x"})
    assert agent._sessions["b"]["disclosed_slots"] == {}
    assert agent._sessions["b"]["history"] == []
    assert agent._sessions["a"]["intent_mode"] == "browsing"


def test_local_live_intent_reads_each_message_and_never_latches():
    """The fast-path fallback reads this message only; the parser covers general turns."""

    state = Agent._new_session_state()
    assert Agent._detect_intent_locally(state, "show me options").value == "browsing"
    assert Agent._detect_intent_locally(state, "I need waterproof boots, show me options").value == "buying"
    assert Agent._detect_intent_locally(state, "anything under $40").value == "buying"
    assert Agent._detect_intent_locally(state, "not just looking, I want one today").value == "buying"

    # Session state must not colour the read.  Intent drives retrieval thresholds, the
    # memory blend, and clarification order -- all soft -- so a latch quietly mis-steers
    # every later turn, while a wrong read costs one turn and self-corrects on the next.
    state["intent_mode"] = "buying"
    state["disclosed_slots"]["material"] = {"cotton"}
    assert Agent._detect_intent_locally(state, "maybe shoes").value == "browsing"
    assert Agent._detect_intent_locally(state, "show me options").value == "browsing"
    assert Agent._detect_intent_locally(state, "actually, show me other styles").value == "browsing"
    assert Agent._detect_intent_locally(state, "start over").value == "browsing"


def _run_structured_state(agent, payload, message="Find something for me"):
    prompts = []
    agent._call_llm = lambda prompt, system_prompt, **kwargs: (
        prompts.append((prompt, system_prompt)) or json.dumps(payload)
    )
    agent._update_state_via_llm("s", message, turn=1)
    return prompts[0], agent._sessions["s"]


def test_state_editor_prompt_includes_complete_prior_state_and_message():
    agent = bare_agent()
    agent.reset("s", {})
    agent._sessions["s"]["disclosed_slots"] = {"color": {"blue"}}
    (prompt, system_prompt), _ = _run_structured_state(agent, {})
    assert '"color"' in prompt and '"blue"' in prompt
    assert "Find something for me" in prompt
    assert 'exclusively in "target_department"' in system_prompt
    assert 'never put demographics in "department", "use_case", or any "disclosed_slots" key' in system_prompt
    assert "NOT confined to this list" in system_prompt


def test_state_editor_provider_failure_propagates_without_local_fallback():
    agent = bare_agent()
    agent.reset("s", {})
    before = deepcopy(agent._sessions["s"])
    agent._call_llm = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        RuntimeError("provider down")
    )
    with pytest.raises(RuntimeError, match="provider down"):
        agent._update_state_via_llm("s", "I'm looking for boots", turn=1)
    assert agent._sessions["s"] == before


def test_invalid_numeric_fields_keep_historical_field_local_tolerance():
    agent = bare_agent()
    agent.reset("s", {})
    _, state = _run_structured_state(agent, {
        "price_max": "invalid",
        "min_avg_rating": "invalid",
        "min_rating_number": "invalid",
        "disclosed_slots": {"closure": ["zipper"]},
    })
    assert state["price_max"] == 9999.0
    assert state["min_avg_rating"] == 0.0
    assert state["min_rating_number"] == 0
    assert state["disclosed_slots"]["closure"] == {"zipper"}


def test_structured_target_department_activates_session_only_gender():
    agent = bare_agent()
    agent.reset("s", {})
    _, state = _run_structured_state(agent, {"category": "dress", "department": "women", "target_department": "women"})
    assert state["department"] == "clothing"
    assert state["target_department"] == "women"
    assert state["disclosed_slots"]["gender"] == {"women"}


def test_structured_mens_department_is_normalized():
    agent = bare_agent()
    agent.reset("s", {})
    _, state = _run_structured_state(agent, {"category": "boots", "department": "men"})
    assert state["department"] == "shoes"
    assert state["target_department"] == "men"
    assert state["disclosed_slots"]["gender"] == {"men"}
    assert "use_case" not in state["disclosed_slots"]


def test_structured_custom_slots_are_accepted():
    agent = bare_agent()
    agent.reset("s", {})
    _, state = _run_structured_state(agent, {"disclosed_slots": {"closure": ["zipper"]}})
    assert state["disclosed_slots"]["closure"] == {"zipper"}


def test_structured_demographic_is_absent_from_committed_ltm_text():
    import numpy as np

    agent = bare_agent()
    agent.vector_memory_config = type("Config", (), {"ewma_alpha": 0.30})()
    agent.reset("s", {}, user_id="u", sequence_index=0)
    _run_structured_state(agent, {
        "category": "dress", "department": "women", "target_department": "women",
        "disclosed_slots": {"color": ["blue"]},
    })
    embedded = []
    agent.embed_dense_query = lambda text: embedded.append(text) or np.array([1.0, 0.0], dtype=np.float32)
    agent.end_session("s")
    trace = agent.get_memory_debug("s")
    assert embedded == ["color: blue"]
    assert trace["preference_text"] == "color: blue"


def test_end_session_exposes_vector_free_adaptive_update_diagnostics():
    import numpy as np

    agent = bare_agent()
    from system.shopping_agent.vector_memory import DEFAULT_VECTOR_MEMORY_CONFIG
    agent.vector_memory_config = DEFAULT_VECTOR_MEMORY_CONFIG
    embedded = []
    agent.embed_dense_query = lambda text: embedded.append(text) or np.array([1.0, 0.0], dtype=np.float32)

    agent.reset("first", {}, user_id="u", sequence_index=0)
    agent._sessions["first"]["disclosed_slots"] = {"color": {"blue"}, "budget": {"under 50"}}
    agent.end_session("first")

    agent.reset("second", {}, user_id="u", sequence_index=1)
    agent._sessions["second"]["disclosed_slots"] = {"color": {"blue"}, "brand": {"example"}}
    agent.end_session("second")
    trace = agent.get_memory_debug("second")

    assert embedded == ["color: blue", "color: blue"]
    assert trace["update_mode"] == "adaptive"
    assert trace["raw_update_similarity"] == pytest.approx(1.0)
    assert trace["bounded_update_similarity"] == pytest.approx(1.0)
    assert trace["effective_alpha"] == pytest.approx(0.0)
    assert trace["update_fallback_reason"] is None
    assert trace["vector_changed"] is False
    assert trace["post_update_memory"]["update_count"] == 2
    assert not any(isinstance(value, np.ndarray) for value in trace.values())
    assert "vector" not in trace
    assert "vector" not in trace["post_update_memory"]


def test_fast_parser_builds_canonical_active_query():
    agent = bare_agent()
    agent.reset("s", {})
    agent._parse_message_locally("s", "I'm looking for boots. A key requirement is: waterproof.")
    query = _state_to_retrieval_query(agent._sessions["s"])
    assert "boots" in query
    assert "waterproof" in query


def test_override_replaces_stale_preferences_and_stashes_terms():
    agent = bare_agent()
    agent.reset("s", {})
    agent._parse_message_locally("s", "I'm looking for black boots.")
    agent._parse_message_locally(
        "s", "Actually, ignore my earlier preference. What I need is: white sandals."
    )
    state = agent._sessions["s"]
    assert "black" not in _state_to_retrieval_query(state)
    assert "white" in _state_to_retrieval_query(state)


def test_no_preference_erases_slot_without_leaking_stash_into_query():
    agent = bare_agent()
    agent.reset("s", {})
    state = agent._sessions["s"]
    state["disclosed_slots"]["color"] = {"black"}
    state["accumulated_terms"].append("black")
    agent._parse_message_locally("s", "I don't have a preference for color.")
    assert "color" not in state["disclosed_slots"]
    assert "black" in state["stashed_terms"]
    assert "black" not in _state_to_retrieval_query(state)


def test_query_serialization_is_deterministic_for_sets():
    state = Agent._new_session_state()
    state["category"] = "boots"
    state["department"] = "shoes"
    state["disclosed_slots"] = {"style": {"western", "casual"}, "color": {"black"}}
    first = _state_to_retrieval_query(state)
    second = _state_to_retrieval_query(deepcopy(state))
    assert first == second == "boots shoes black casual western"


def test_session_state_has_exact_pre_winston_keys():
    state = Agent._new_session_state()
    assert set(state) == {
        "disclosed_slots", "constraint_provenance", "accumulated_terms",
        "stashed_terms", "search_epoch", "seen_asins", "seen_asins_by_epoch",
        "history", "negated_terms", "asked_attributes", "intent_mode",
        "intent_source", "category", "department", "price_max",
        "target_department", "min_avg_rating", "min_rating_number", "store",
        "debug_info",
    }


def test_malformed_state_json_rolls_back_then_uses_local_parser():
    agent = bare_agent()
    agent.reset("s", {})
    before = deepcopy(agent._sessions["s"])
    agent._call_llm = lambda *_args, **_kwargs: "not-json"
    agent._update_state_via_llm("s", "I'm looking for waterproof boots.", turn=1)
    state = agent._sessions["s"]
    assert before["disclosed_slots"] == {}
    assert state["category"] == "waterproof boots"
    assert "price_min" not in state


def test_longitudinal_identity_and_sequence_validation():
    agent = bare_agent()
    agent.reset("s1", {}, user_id="u", sequence_index=1)
    try:
        agent.reset("s2", {}, user_id="u", sequence_index=2)
    except ValueError as exc:
        assert "active session" in str(exc)
    else:
        raise AssertionError("overlapping same-user sessions must be rejected")


def test_anonymous_reset_keeps_four_argument_api_eligible():
    agent = bare_agent()
    agent.reset("anonymous", {})
    assert agent._active_lifecycle["anonymous"]["visible_state"] is None


def test_end_session_embedding_failure_is_retryable():
    agent = bare_agent()
    agent.vector_memory_config = type("Config", (), {"ewma_alpha": 0.30})()
    agent.reset("s", {}, user_id="u", sequence_index=1)
    agent._sessions["s"]["disclosed_slots"] = {"color": {"black"}}
    calls = 0
    def embed(_):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("provider unavailable")
        import numpy as np
        return np.array([1.0, 0.0], dtype=np.float32)
    agent.embed_dense_query = embed
    try:
        agent.end_session("s")
    except RuntimeError:
        pass
    assert "s" in agent._active_lifecycle
    commit = agent.end_session("s")
    assert commit.vector_changed
    assert "s" not in agent._active_lifecycle


def test_session_only_hard_conditions_are_excluded_from_ltm_text():
    import numpy as np

    agent = bare_agent()
    agent.vector_memory_config = type("Config", (), {"ewma_alpha": 0.30})()
    agent.reset("hard", {}, user_id="u", sequence_index=0)
    agent._sessions["hard"]["disclosed_slots"] = {
        "budget": {"under 50"}, "brand": {"example"}, "gender": {"women"},
        "rating": {"4 stars"}, "reviews": {"100 reviews"}, "color": {"blue"},
    }
    embedded = []
    agent.embed_dense_query = lambda text: embedded.append(text) or np.array([1.0, 0.0], dtype=np.float32)
    agent.end_session("hard")
    trace = agent.get_memory_debug("hard")
    assert embedded == ["color: blue"]
    assert trace["preference_text"] == "color: blue"
