from __future__ import annotations

from copy import deepcopy
import inspect
import json
from threading import Event, Thread

import numpy as np
import pytest

from system.shopping_agent.agent import Agent
from system.shopping_agent.catalogue import Catalogue, contains_phrase
from system.shopping_agent.memory_store import InMemoryVectorMemoryStore, JsonFileVectorMemoryStore
from system.shopping_agent.visualizer.server import (
    BrowserApplication, _catalog_family, _safe_image_url,
)


def bare_agent() -> Agent:
    agent = Agent.__new__(Agent)
    agent._sessions = {}
    agent._active_lifecycle = {}
    agent._ended_lifecycle = {}
    agent._forensic_capture_sessions = set()
    agent._forensic_ranking_snapshots = {}
    agent._forensic_update_sessions = set()
    agent._forensic_update_vectors = {}
    agent.memory_store = InMemoryVectorMemoryStore()
    agent.embedding_space_id = "test-space"
    agent.instrumentation = {"semantic_queries": [], "turns": [], "agent_errors": []}
    return agent


def structured(agent: Agent, payload: dict, message: str = "update", turn: int = 2) -> dict:
    agent._call_llm = lambda *args, **kwargs: json.dumps(payload)
    agent._update_state_via_llm("s", message, turn=turn)
    return agent._sessions["s"]


def test_structured_state_defensively_merges_slots_and_asked_attributes():
    agent = bare_agent()
    agent.reset("s", {})
    state = agent._sessions["s"]
    state["disclosed_slots"] = {"color": {"black"}}
    state["asked_attributes"] = {"style"}
    updated = structured(agent, {
        "disclosed_slots": {"material": ["cotton"]},
        "asked_attributes": ["budget", "not-a-real-attribute"],
    })
    assert updated["disclosed_slots"] == {"color": {"black"}, "material": {"cotton"}}
    assert updated["asked_attributes"] == {"style", "budget"}


def test_category_change_clears_slots_and_asked_attributes():
    agent = bare_agent()
    agent.reset("s", {})
    state = agent._sessions["s"]
    state["disclosed_slots"] = {"color": {"black"}}
    state["asked_attributes"] = {"style", "budget"}
    updated = structured(agent, {"category": "boots", "disclosed_slots": {}})
    assert updated["disclosed_slots"] == {}
    assert updated["asked_attributes"] == set()


@pytest.mark.parametrize("message", [
    "I am not just looking; I need waterproof boots",
    "Just looking, but it must be under $50",
    "Just looking for a specific Nike boot under $100",
])
def test_concrete_buying_signals_beat_just_looking(message):
    assert Agent._detect_intent_locally(Agent._new_session_state(), message).value == "buying"


@pytest.mark.parametrize("message", ["start over", "Actually, show me other styles"])
def test_true_reset_messages_remain_browsing(message):
    state = Agent._new_session_state()
    state["intent_mode"] = "buying"
    assert Agent._detect_intent_locally(state, message).value == "browsing"


def test_local_negation_and_brand_parsing_are_clause_bounded():
    agent = bare_agent()
    agent.reset("s", {})
    agent._parse_message_locally("s", "I'm looking for no leather boots from brands like Nike under $100", 1)
    state = agent._sessions["s"]
    assert state["category"] == "boots"
    assert state["store"] == "nike"
    assert state["price_max"] == 100
    assert "leather" in state["negated_terms"]
    assert "material" not in state["disclosed_slots"]
    assert state["disclosed_slots"]["brand"] == {"nike"}


def test_later_negation_removes_positive_slot_and_committed_text():
    agent = bare_agent()
    agent.vector_memory_config = type("Config", (), {"ewma_alpha": 0.30})()
    agent.reset("s", {}, user_id="u", sequence_index=0)
    agent._parse_message_locally("s", "I'm looking for leather boots", 1)
    agent._parse_message_locally("s", "No leather, please", 2)
    embedded = []
    agent.embed_dense_query = lambda text: embedded.append(text) or np.array([1.0, 0.0], np.float32)
    agent.end_session("s")
    trace = agent.get_memory_debug("s", consume=True)
    assert "leather" not in trace["preference_text"]
    assert all("leather" not in text for text in embedded)


def test_only_generated_questions_are_recorded():
    agent = bare_agent()
    agent.reset("s", {})
    state = agent._sessions["s"]
    assert agent._extract_asked_attributes("Which material do you prefer?", state) == {"material"}
    assert agent._extract_asked_attributes(
        "Which material do you prefer? What budget works for you?", state
    ) == {"material", "budget"}


def test_failed_respond_restores_session_state_but_retains_instrumentation():
    agent = bare_agent()
    agent.reset("s", {})
    before = deepcopy(agent._sessions["s"])

    def fail(*args, **kwargs):
        agent._sessions["s"]["history"].append({"role": "user", "content": "phantom"})
        raise RuntimeError("generation failed")

    agent._respond_custom = fail
    with pytest.raises(RuntimeError, match="generation failed"):
        agent.respond("s", "hello", 1, 5)
    assert agent._sessions["s"] == before
    assert agent.instrumentation["agent_errors"][-1]["type"] == "RuntimeError"
    assert agent.instrumentation["turns"][-1]["failed"] is True


def test_json_commit_persistence_failure_rolls_back_and_is_retryable(tmp_path, monkeypatch):
    store = JsonFileVectorMemoryStore(tmp_path / "memory.json")
    store.begin_session("u", "s", 0)
    original = store._persist
    monkeypatch.setattr(store, "_persist", lambda: (_ for _ in ()).throw(OSError("disk full")))
    with pytest.raises(OSError, match="disk full"):
        store.commit(user_id="u", session_id="s", sequence_index=0,
                     embedding_space_id="space", new_preferences=np.array([1, 0], np.float32))
    assert store.get_state("u") is None
    assert store.commits_for_user("u") == ()
    monkeypatch.setattr(store, "_persist", original)
    committed = store.commit(user_id="u", session_id="s", sequence_index=0,
                             embedding_space_id="space", new_preferences=np.array([1, 0], np.float32))
    assert committed.vector_changed


@pytest.mark.parametrize("operation", ["clear", "clear_user"])
def test_json_reset_persistence_failure_restores_memory(tmp_path, monkeypatch, operation):
    store = JsonFileVectorMemoryStore(tmp_path / "memory.json")
    store.commit(user_id="u", session_id="s", sequence_index=0,
                 embedding_space_id="space", new_preferences=np.array([1, 0], np.float32))
    monkeypatch.setattr(store, "_persist", lambda: (_ for _ in ()).throw(OSError("disk full")))
    with pytest.raises(OSError, match="disk full"):
        store.clear() if operation == "clear" else store.clear_user("u")
    assert store.get_state("u") is not None
    assert len(store.commits_for_user("u")) == 1


def test_cancel_session_and_ended_trace_retention_and_consumption():
    agent = bare_agent()
    agent.vector_memory_config = type("Config", (), {"ewma_alpha": 0.30})()
    agent.embed_dense_query = lambda text: np.array([1.0, 0.0], np.float32)
    agent.reset("discard", {}, user_id="discard-user", sequence_index=0)
    agent.discard_session("discard")
    agent.reset("retry", {}, user_id="discard-user", sequence_index=0)
    agent.discard_session("retry")
    for index in range(40):
        session = f"s-{index}"
        agent.reset(session, {})
        agent.end_session(session)
    assert len(agent._ended_lifecycle) == 32
    newest = "s-39"
    assert agent.get_memory_debug(newest, consume=True)["ended"] is True
    assert newest not in agent._ended_lifecycle


def test_safe_defaults_phrase_boundaries_and_catalog_families():
    assert inspect.signature(Agent).parameters["allow_catalog_embedding"].default is False
    assert contains_phrase("water resistant tan watch", "tan")
    assert not contains_phrase("Overwatch inspired water-resistant", "watch")
    assert not contains_phrase("Overwatch inspired water-resistant", "red")
    assert not contains_phrase("Overwatch inspired water-resistant", "tan")
    assert _catalog_family(["Clothing, Shoes & Jewelry", "Women", "Shoes", "Boots"]) == "shoes"
    assert _catalog_family(["Clothing, Shoes & Jewelry", "Women", "Watches"]) == "watches"
    assert _catalog_family(["Clothing, Shoes & Jewelry"]) == "other"


def test_catalogue_negative_masks_use_phrase_boundaries_and_close_is_idempotent(tmp_path):
    path = tmp_path / "catalog.jsonl"
    products = [
        {"parent_asin": "a", "title": "Overwatch inspired resistant gear", "categories": ["Clothing"]},
        {"parent_asin": "b", "title": "Red tan watch", "categories": ["Watches"]},
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in products), encoding="utf-8")
    catalogue = Catalogue(path)
    assert catalogue.eligibility({"negated_terms": {"watch", "red", "tan"}}).mask.tolist() == [True, False]
    catalogue.close()
    catalogue.close()


def test_dashboard_rejects_script_image_urls_and_catalog_reads_do_not_take_lifecycle_lock():
    assert _safe_image_url("javascript:alert(1)") is None
    assert _safe_image_url("https://example.test/image.jpg") == "https://example.test/image.jpg"

    app = BrowserApplication.__new__(BrowserApplication)
    app._lifecycle_lock = __import__("threading").RLock()
    app.catalog_rows = ({
        "asin": "a", "title": "Safe", "brand": "Brand", "price": 1.0,
        "avg_rating": 5.0, "rating_number": 1, "categories": ["Watches"],
        "_query_text": "safe brand watches", "_family": "watches",
    },)
    app.images = {"a": "javascript:alert(1)"}
    entered = Event()
    release = Event()

    def hold_lifecycle():
        with app._lifecycle_lock:
            entered.set()
            release.wait(2)

    thread = Thread(target=hold_lifecycle)
    thread.start()
    assert entered.wait(1)
    result = app.catalog_search(department="watches")
    release.set()
    thread.join(2)
    assert result["products"][0]["image_url"] is None
