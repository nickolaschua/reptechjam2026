from copy import deepcopy

from system.shopping_agent.agent import Agent, _state_to_retrieval_query
from system.shopping_agent.memory_store import InMemoryVectorMemoryStore
from system.shopping_agent.turn_parser import PROMPT, SCHEMA, ParsedSlot, ParsedTurn


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


def test_local_live_intent_upgrades_and_only_explicitly_resets():
    state = Agent._new_session_state()
    assert Agent._detect_intent_locally(state, "show me options").value == "browsing"
    assert Agent._detect_intent_locally(state, "I need waterproof boots, show me options").value == "buying"
    state["disclosed_slots"]["material"] = {"cotton"}
    assert Agent._detect_intent_locally(state, "cotton please").value == "buying"
    state["intent_mode"] = "buying"
    state["disclosed_slots"].clear()
    assert Agent._detect_intent_locally(state, "maybe shoes").value == "buying"
    assert Agent._detect_intent_locally(state, "show me options").value == "buying"
    assert Agent._detect_intent_locally(state, "actually, show me other styles").value == "browsing"
    assert Agent._detect_intent_locally(state, "start over").value == "browsing"


def _turn(*, category=None, department=None, slots=()):
    return ParsedTurn(
        category=category,
        positive_slots=tuple(slots),
        negatives=(),
        declined_attributes=(),
        price_min=None,
        price_max=None,
        department=department,
        specificity="type_with_wishes",
        intent="browsing",
        message_type="feature",
        model_code=None,
        resolver_candidates=(),
        resolver_confidence=0.5,
        raw_parse={},
    )


def _run_structured_state(agent, parsed):
    agent._apply_parsed_turn(agent._sessions["s"], parsed, "Find something for me", 1)
    return agent._sessions["s"]


def test_constrained_schema_has_one_demographic_destination():
    assert "department" in SCHEMA["properties"]
    assert "target_department" not in SCHEMA["properties"]
    assert "department: only if stated" in PROMPT


def test_structured_target_department_activates_session_only_gender():
    agent = bare_agent()
    agent.reset("s", {})
    state = _run_structured_state(agent, _turn(category="dress", department="womens"))
    assert state["department"] == "clothing"
    assert state["target_department"] == "women"
    assert state["disclosed_slots"]["gender"] == {"women"}


def test_structured_mens_department_is_normalized():
    agent = bare_agent()
    agent.reset("s", {})
    state = _run_structured_state(agent, _turn(category="boots", department="mens"))
    assert state["department"] == "shoes"
    assert state["target_department"] == "men"
    assert state["disclosed_slots"]["gender"] == {"men"}
    assert "use_case" not in state["disclosed_slots"]


def test_structured_unisex_adult_does_not_invent_a_binary_target():
    agent = bare_agent()
    agent.reset("s", {})
    state = _run_structured_state(agent, _turn(category="shirt", department="unisex-adult"))
    assert state["target_department"] == ""
    assert "gender" not in state["disclosed_slots"]


def test_structured_demographic_is_absent_from_committed_ltm_text():
    import numpy as np

    agent = bare_agent()
    agent.vector_memory_config = type("Config", (), {"ewma_alpha": 0.30})()
    agent.reset("s", {}, user_id="u", sequence_index=0)
    _run_structured_state(
        agent,
        _turn(
            category="dress",
            department="womens",
            slots=(ParsedSlot("color", "blue", "soft"),),
        ),
    )
    embedded = []
    agent.embed_dense_query = lambda text: embedded.append(text) or np.array([1.0, 0.0], dtype=np.float32)
    agent.end_session("s")
    trace = agent.get_memory_debug("s")
    assert embedded == ["color: blue"]
    assert trace["preference_text"] == "color: blue"


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
