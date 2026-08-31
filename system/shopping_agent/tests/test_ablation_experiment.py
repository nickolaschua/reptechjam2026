from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from system.shopping_agent import agent as agent_module
from system.shopping_agent.agent import Agent, ExperimentConfig
from system.shopping_agent.clarification import select_fixed_priority_attributes
from system.shopping_agent.model_client import ModelCall
from system.shopping_agent.run_ablation_experiments import CONDITIONS, paired_comparisons
from system.shopping_agent.tests.test_agent_vector_ranking import ranking_agent
from system.shopping_agent.visualizer.simulator import call_shopper_llm


class RecordingClient:
    provider = "test"
    model = "deterministic"

    def __init__(self) -> None:
        self.options = []

    def chat_result(self, messages, *, options=None, role="chat", **kwargs):
        self.options.append(dict(options or {}))
        return ModelCall("ok", self.model, 0.01, 1, role, self.provider)

    def chat(self, messages, **kwargs):
        return self.chat_result(messages, **kwargs).content

    def instrumentation(self):
        return []


def test_production_experiment_defaults_are_unchanged():
    config = ExperimentConfig()
    assert config.clarification_policy == "entropy"
    assert config.long_term_memory_read_enabled is True
    assert config.llm_temperature == 0.4
    assert config.llm_seed is None
    assert config.condition_name == "all_in"


def test_four_conditions_vary_only_the_two_registered_factors():
    assert list(CONDITIONS) == ["baseline", "entropy_only", "ltm_only", "all_in"]
    assert {
        name: (cfg.clarification_policy, cfg.long_term_memory_read_enabled)
        for name, cfg in CONDITIONS.items()
    } == {
        "baseline": ("fixed_priority", False),
        "entropy_only": ("entropy", False),
        "ltm_only": ("fixed_priority", True),
        "all_in": ("entropy", True),
    }
    assert {(cfg.llm_temperature, cfg.llm_seed) for cfg in CONDITIONS.values()} == {(0.0, 20260901)}


def test_fixed_priority_is_deterministic_and_intent_specific():
    remaining = {"material", "brand", "style", "use_case", "budget"}
    buying = select_fixed_priority_attributes(None, (), remaining, intent_mode="buying")
    browsing = select_fixed_priority_attributes(None, (), remaining, intent_mode="browsing")
    assert buying == ["material", "brand"]
    assert browsing == ["use_case", "style"]
    assert select_fixed_priority_attributes(None, (), remaining, intent_mode="buying") == buying


def test_shadow_memory_is_loaded_but_never_passed_to_score_catalog(monkeypatch):
    agent = ranking_agent()
    agent.experiment_config = ExperimentConfig("fixed_priority", False, 0.0, 7)
    agent.clarification_selector = select_fixed_priority_attributes
    agent._active_lifecycle["s"]["visible_state"] = SimpleNamespace(
        vector=np.asarray([0.0, 1.0], dtype=np.float32),
        update_count=3,
        embedding_space_id="space",
    )
    observed = {}
    original = agent_module.score_catalog

    def capture(catalog, current, memory, mode, config):
        observed["memory"] = memory
        return original(catalog, current, memory, mode, config)

    monkeypatch.setattr(agent_module, "score_catalog", capture)
    result = agent._respond_custom("s", "show options", 1, 2, emit_debug=False)
    trace = result["debug"]["memory_trace"]
    assert observed["memory"] is None
    assert trace["ltm_available"] is True
    assert trace["ltm_read_enabled"] is False
    assert trace["ltm_applied"] is False


def test_temperature_and_seed_reach_agent_and_shopper_calls():
    client = RecordingClient()
    agent = Agent.__new__(Agent)
    agent.llm_client = client
    agent.experiment_config = ExperimentConfig("entropy", True, 0.0, 20260901)
    agent.instrumentation = {"llm_calls": []}
    agent._sessions = {"s": {"debug_info": {}}}
    assert agent._call_llm("hello", session_id="s") == "ok"
    assert call_shopper_llm(
        "reply", "system", client=client, temperature=0.0, seed=20260901
    ) == "ok"
    assert [value["temperature"] for value in client.options] == [0.0, 0.0]
    assert [value["seed"] for value in client.options] == [20260901, 20260901]


def _row(user, condition, *, hit, rank):
    return {
        "condition": condition,
        "user_id": user,
        "sequence_index": 9,
        "target_asin": f"target-{user}",
        "session_role": "memory_probe",
        "hit": hit,
        "first_hit_turn": 1 if hit else None,
        "best_rank": rank,
        "reciprocal_rank": 0.0 if rank is None else 1.0 / rank,
        "target_full_rank": 10 if not hit else rank,
        "latency_seconds": 1.0,
        "model_call_count": 2,
    }


def test_paired_deltas_and_factorial_interaction_on_known_results():
    users = ["u1", "u2", "u3", "u4"]
    results = {
        "baseline": [_row(u, "baseline", hit=False, rank=None) for u in users],
        "entropy_only": [_row(u, "entropy_only", hit=True, rank=2) for u in users],
        "ltm_only": [_row(u, "ltm_only", hit=True, rank=1) for u in users],
        "all_in": [_row(u, "all_in", hit=True, rank=1) for u in users],
    }
    comparison = paired_comparisons(results)
    assert comparison["entropy_effect"]["hit_rate_at_10"]["delta"] == 1.0
    assert comparison["ltm_effect"]["mrr"]["delta"] == 1.0
    assert comparison["interaction"]["hit_rate_at_10"]["delta"] == -1.0
    assert comparison["independent_user_clusters"] == 4
