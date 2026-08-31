from __future__ import annotations

from pathlib import Path

from system.shopping_agent.visualizer.server import BrowserApplication, buyer_mode_for_scenario


class FakeStore:
    def __init__(self):
        self.sequences = {}

    def next_sequence_index(self, user_id):
        return self.sequences.get(user_id, 0)


class FakeAgent:
    def __init__(self, store):
        self.store = store
        self.resets = []
        self.ends = []
        self.respond_options = []

    def reset(self, session_id, profile, *, user_id, sequence_index):
        self.resets.append((session_id, user_id, sequence_index))

    def respond(self, session_id, message, turn, top_k, *, buyer_mode, debug, emit_trace=None):
        self.respond_options.append({"debug": debug, "emit_trace": emit_trace})
        user_id = next(item[1] for item in self.resets if item[0] == session_id)
        target = "target-a" if user_id == "a" else "target-b"
        return {
            "message": "matches", "ask_attribute": "other",
            "recommendations": [{"parent_asin": target}],
            "debug": {"memory_trace": {"prior_ltm_exists": self.store.sequences.get(user_id, 0) > 0}},
        }

    def end_session(self, session_id):
        self.ends.append(session_id)
        user_id = next(item[1] for item in self.resets if item[0] == session_id)
        self.store.sequences[user_id] = self.store.sequences.get(user_id, 0) + 1

    def get_memory_debug(self, session_id, *, consume=False):
        return {"ltm_updated_after_turn": True, "preference_text": "color: blue"}

    def discard_session(self, session_id):
        pass

    def close(self):
        pass


class FailingAgent(FakeAgent):
    def respond(self, *args, **kwargs):
        raise RuntimeError("provider unavailable")


def _application():
    store = FakeStore()
    agent = FakeAgent(store)
    app = BrowserApplication.__new__(BrowserApplication)
    app.store = store
    app.agent = agent
    app.samples = {
        "a": {"sample_id": "a", "scenario_type": "buying", "ground_truth": {"parent_asin": "target-a"}, "user_profile": {}, "intent_card": {"hard_constraints": [], "soft_preferences": []}, "behavior": {}},
        "b": {"sample_id": "b", "scenario_type": "browsing", "ground_truth": {"parent_asin": "target-b"}, "user_profile": {}, "intent_card": {"hard_constraints": [], "soft_preferences": []}, "behavior": {}},
    }
    app.products = {
        "target-a": {"parent_asin": "target-a", "title": "A", "details": {}, "categories": []},
        "target-b": {"parent_asin": "target-b", "title": "B", "details": {}, "categories": []},
    }
    app.catalog_ids = set(app.products)
    app.images = {"target-a": "https://example.test/a.jpg"}
    app.active = {}
    app.finished_traces = {}
    return app, agent


def test_replacement_commits_once_and_sample_identity_advances_independently():
    app, agent = _application()
    first = app.start("a")
    app.start("b")
    assert agent.ends == [first.session_id]
    app.finish("a", reason="duplicate")
    assert agent.ends == [first.session_id]
    second_a = app.start("a")
    assert second_a.sequence_index == 1
    assert app.store.next_sequence_index("b") == 1  # b was replaced by the new a run


def test_target_hit_commits_and_returns_image_card_once():
    app, agent = _application()
    active = app.start("a")
    result = app.step("a", "hello")
    assert result["success"] is True
    assert result["recommendations"][0]["image_url"].endswith("a.jpg")
    assert result["debug"]["memory_trace"]["ltm_updated_after_session"] is True
    assert agent.respond_options == [{"debug": True, "emit_trace": True}]
    assert agent.ends == [active.session_id]
    app.finish("a", reason="finally")
    assert agent.ends == [active.session_id]


def test_scenario_mode_mapping_is_locked():
    assert buyer_mode_for_scenario("browsing").value == "browsing"
    for scenario in ["buying", "intent_override", "boundary"]:
        assert buyer_mode_for_scenario(scenario).value == "buying"


def test_catalog_search_covers_all_rows_watches_and_unknown_metadata():
    app, _ = _application()
    app.products = {
        f"watch-{index:02d}": {
            "parent_asin": f"watch-{index:02d}", "title": f"Chronograph Watch {index}",
            "store": "Time Co", "details": {}, "categories": ["Watches"],
            "price": None if index == 0 else 100 + index,
            "average_rating": None if index == 0 else 4.8,
            "rating_number": 1000 - index,
        }
        for index in range(30)
    }
    app.catalog_rows = app._build_catalog_rows()
    result = app.catalog_search(
        q="watch", department="watches", max_price=50, min_rating=4.9, page=1,
    )
    assert result["total"] == 1  # Unknown price/rating retains benefit of doubt.
    assert result["products"][0]["asin"] == "watch-00"

    unfiltered = app.catalog_search(department="watches", page=2)
    assert unfiltered["total"] == 30
    assert unfiltered["per_page"] == 24
    assert len(unfiltered["products"]) == 6


def test_catalog_empty_query_popularity_order_has_stable_asin_ties():
    app, _ = _application()
    app.products["another"] = {
        "parent_asin": "another", "title": "Another shirt", "details": {},
        "categories": ["Clothing"], "rating_number": 0,
    }
    app.catalog_rows = app._build_catalog_rows()
    result = app.catalog_search()
    assert [item["asin"] for item in result["products"]] == sorted(app.products)


def test_failed_browser_turn_does_not_advance_counter_and_can_be_discarded():
    app, _ = _application()
    failing = FailingAgent(app.store)
    app.agent = failing
    active = app.start("a")
    with __import__("pytest").raises(RuntimeError, match="provider unavailable"):
        app.step("a", "hello")
    assert active.turn == 0
    app.discard("a")
    assert "a" not in app.active


def test_manual_ui_rejects_error_payloads_instead_of_rendering_undefined_turns():
    html = (Path(__file__).parents[1] / "visualizer" / "conversation.html").read_text(
        encoding="utf-8"
    )
    assert "if (!res.ok)" in html
    assert "!Number.isInteger(data.turn)" in html
    assert "manualTurn = Math.max(0, manualTurn - 1)" in html
    assert "Request failed: ${e.message" in html


def test_conversation_ui_has_collapsible_demo_panels_and_narrow_defaults():
    html = (Path(__file__).parents[1] / "visualizer" / "conversation.html").read_text(
        encoding="utf-8"
    )
    assert 'id="toggle-sessions"' in html
    assert 'id="toggle-inspector"' in html
    assert 'id="focus-chat"' in html
    assert "function toggleFocusMode()" in html
    assert "window.matchMedia('(max-width: 1180px)')" in html
    assert "setPanelVisibility('sessions', !narrowLayout.matches)" in html
    assert "setPanelVisibility('inspector', !narrowLayout.matches)" in html
    assert "if (narrowLayout.matches) closePanels();" in html


def test_chat_sender_labels_do_not_expose_provider_or_model_name():
    html = (Path(__file__).parents[1] / "visualizer" / "conversation.html").read_text(
        encoding="utf-8"
    )
    assert "'Shopper (__MODEL_LABEL__)'" not in html
    assert "'Copilot (__MODEL_LABEL__)'" not in html
    assert "? 'Shopper'" in html
    assert "'Request Error' : 'Copilot'" in html


def test_demo_startup_logs_do_not_expose_provider_or_model_details():
    shopping_agent_dir = Path(__file__).parents[1]
    startup_source = "\n".join(
        (shopping_agent_dir / relative_path).read_text(encoding="utf-8")
        for relative_path in ("agent.py", "visualizer/server.py")
    )
    for provider_detail in (
        "[Hybrid Agent] Embedding backend:",
        "[Hybrid Agent] Model provider:",
        "Loading pre-computed embeddings:",
        "chat={selected.llm_client.model}",
        "cache={cache_path}",
    ):
        assert provider_detail not in startup_source
