from __future__ import annotations

import sys
import unittest
import json
from copy import deepcopy
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np


SHOPPING_AGENT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SHOPPING_AGENT_DIR))

import agent as agent_module


class StateRoutingTests(unittest.TestCase):
    CANONICAL_KEYS = {
        "disclosed_slots",
        "accumulated_terms",
        "stashed_terms",
        "seen_asins",
        "history",
        "negated_terms",
        "asked_attributes",
        "category",
        "department",
        "price_max",
        "debug_info",
    }

    def setUp(self) -> None:
        patchers = [
            patch.object(agent_module.Agent, "_build_fts5_index"),
            patch.object(agent_module.Agent, "_build_category_index"),
            patch.object(agent_module.Agent, "_build_vector_index"),
            patch.object(agent_module.BaselineAgent, "_build_index"),
            patch.object(agent_module, "SentenceTransformer"),
        ]
        for patcher in patchers:
            patcher.start()
            self.addCleanup(patcher.stop)

        self.agent = agent_module.Agent("unused-catalog.jsonl")
        self.addCleanup(self.agent.connection.close)
        self.addCleanup(self.agent.baseline_agent.connection.close)
        self.routes: list[str] = []
        self.full_observations: list[dict] = []

        def fast_response(session_id: str, message: str, turn: int, top_k: int) -> dict:
            self.routes.append("fast")
            self.agent.baseline_agent._parse_message_locally(session_id, message)
            return {"message": "fast", "ask_attribute": "other", "recommendations": []}

        def full_response(session_id: str, message: str, turn: int, top_k: int) -> dict:
            self.routes.append("full")
            state = self.agent._sessions[session_id]
            self.full_observations.append(
                {
                    "state": state,
                    "state_id": id(state),
                    "category": state["category"],
                    "accumulated_terms": list(state["accumulated_terms"]),
                    "disclosed_slots": {
                        key: set(values) for key, values in state["disclosed_slots"].items()
                    },
                }
            )
            return {"message": "full", "ask_attribute": "other", "recommendations": []}

        self.agent.baseline_agent.respond = MagicMock(side_effect=fast_response)
        self.agent._respond_custom = MagicMock(side_effect=full_response)

    def assert_canonical_state(self, state: dict) -> None:
        self.assertEqual(set(state), self.CANONICAL_KEYS)
        self.assertIsInstance(state["disclosed_slots"], dict)
        self.assertTrue(all(isinstance(values, set) for values in state["disclosed_slots"].values()))
        self.assertIsInstance(state["accumulated_terms"], list)
        self.assertIsInstance(state["stashed_terms"], list)
        self.assertIsInstance(state["seen_asins"], set)
        self.assertIsInstance(state["history"], list)
        self.assertIsInstance(state["negated_terms"], set)
        self.assertIsInstance(state["asked_attributes"], set)
        self.assertIsInstance(state["category"], str)
        self.assertIsInstance(state["department"], str)
        self.assertIsInstance(state["price_max"], float)
        self.assertIsInstance(state["debug_info"], dict)

    def test_template_dialogue_stays_fast_and_persists_constraints(self) -> None:
        self.agent.reset("template", {})

        self.agent.respond("template", "I'm looking for boots.", 1, 10)
        self.agent.respond("template", "For that, what matters is: black; leather.", 2, 10)
        self.agent.respond(
            "template",
            "I don't have a preference for size; please use your judgment.",
            3,
            10,
        )

        state = self.agent._sessions["template"]
        self.assertEqual(self.routes, ["fast", "fast", "fast"])
        self.assertEqual(state["category"], "boots")
        self.assertEqual(state["disclosed_slots"]["color"], {"black"})
        self.assertEqual(state["disclosed_slots"]["material"], {"leather"})
        self.assertIn("size", state["asked_attributes"])

    def test_natural_dialogue_stays_full(self) -> None:
        self.agent.reset("natural", {})

        self.agent.respond("natural", "Could you help me find some boots?", 1, 10)
        self.agent.respond("natural", "Black would be ideal.", 2, 10)

        self.assertEqual(self.routes, ["full", "full"])

    def test_fast_then_full_observes_identical_shared_state(self) -> None:
        self.agent.reset("handoff", {})
        state = self.agent._sessions["handoff"]

        self.agent.respond(
            "handoff",
            "I'm looking for boots. A key requirement is: waterproof.",
            1,
            10,
        )
        expected_terms = list(state["accumulated_terms"])
        expected_slots = {key: set(values) for key, values in state["disclosed_slots"].items()}
        self.agent.respond("handoff", "Can you show me something less formal?", 2, 10)

        observation = self.full_observations[-1]
        self.assertEqual(self.routes, ["fast", "full"])
        self.assertIs(observation["state"], state)
        self.assertEqual(observation["state_id"], id(self.agent.baseline_agent._sessions["handoff"]))
        self.assertEqual(observation["category"], "boots")
        self.assertEqual(observation["accumulated_terms"], expected_terms)
        self.assertEqual(observation["disclosed_slots"], expected_slots)
        self.assertEqual(expected_slots["feature"], {"waterproof"})

    def test_sessions_are_isolated_including_nested_values(self) -> None:
        self.agent.reset("one", {})
        self.agent.reset("two", {})
        first = self.agent._sessions["one"]
        second = self.agent._sessions["two"]

        first["disclosed_slots"]["color"] = {"black"}
        first["accumulated_terms"].append("boots")
        first["seen_asins"].add("asin-1")
        first["history"].append({"role": "user", "content": "hello"})
        first["debug_info"]["route"] = "fast"

        self.assertIsNot(first, second)
        self.assertEqual(second["disclosed_slots"], {})
        self.assertEqual(second["accumulated_terms"], [])
        self.assertEqual(second["seen_asins"], set())
        self.assertEqual(second["history"], [])
        self.assertEqual(second["debug_info"], {})

    def test_matching_template_can_return_to_fast_after_full_turn(self) -> None:
        self.agent.reset("reroute", {})

        self.agent.respond("reroute", "I'm looking for boots.", 1, 10)
        self.agent.respond("reroute", "Could they be suitable for rain?", 2, 10)
        self.agent.respond("reroute", "For that, what matters is: leather.", 3, 10)

        self.assertEqual(self.routes, ["fast", "full", "fast"])
        self.assertEqual(self.agent._sessions["reroute"]["disclosed_slots"]["material"], {"leather"})

    def test_baseline_failure_falls_back_only_for_that_turn(self) -> None:
        self.agent.reset("failure", {})
        original_fast_response = self.agent.baseline_agent.respond.side_effect
        attempts = 0

        def fail_once(session_id: str, message: str, turn: int, top_k: int) -> dict:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise RuntimeError("synthetic baseline failure")
            return original_fast_response(session_id, message, turn, top_k)

        self.agent.baseline_agent.respond.side_effect = fail_once
        self.agent.respond("failure", "I'm looking for boots.", 1, 10)
        self.agent.respond("failure", "For that, what matters is: leather.", 2, 10)

        self.assertEqual(self.routes, ["full", "fast"])

    def test_partial_fast_mutation_is_rolled_back_before_full_fallback(self) -> None:
        self.agent.reset("rollback", {})
        state = self.agent._sessions["rollback"]
        state["category"] = "boots"
        state["disclosed_slots"]["color"] = {"black"}
        state["accumulated_terms"].extend(["boots", "black"])
        state["seen_asins"].add("asin-before")
        state["history"].append({"role": "user", "content": "earlier turn"})
        state["debug_info"]["nested"] = {"attempts": []}
        state_before_turn = deepcopy(state)
        received_states = []

        def mutate_then_fail(session_id: str, message: str, turn: int, top_k: int) -> dict:
            fast_state = self.agent.baseline_agent._sessions[session_id]
            fast_state["category"] = "corrupted"
            fast_state["disclosed_slots"]["color"].add("white")
            fast_state["disclosed_slots"]["feature"] = {"waterproof"}
            fast_state["accumulated_terms"].append("waterproof")
            fast_state["seen_asins"].add("asin-partial")
            fast_state["history"][0]["content"] = "mutated history"
            fast_state["debug_info"]["nested"]["attempts"].append("fast")
            raise RuntimeError("failure after partial mutation")

        def capture_full_state(session_id: str, message: str, turn: int, top_k: int) -> dict:
            received_states.append(deepcopy(self.agent._sessions[session_id]))
            return {"message": "full", "ask_attribute": "other", "recommendations": []}

        self.agent.baseline_agent.respond.side_effect = mutate_then_fail
        self.agent._respond_custom.side_effect = capture_full_state
        self.agent.respond("rollback", "I'm looking for waterproof boots.", 1, 10)

        self.assertEqual(received_states, [state_before_turn])

    def test_successful_fast_mutation_is_committed(self) -> None:
        self.agent.reset("commit", {})

        def mutate_and_succeed(session_id: str, message: str, turn: int, top_k: int) -> dict:
            state = self.agent.baseline_agent._sessions[session_id]
            state["category"] = "boots"
            state["disclosed_slots"]["feature"] = {"waterproof"}
            state["accumulated_terms"].extend(["boots", "waterproof"])
            state["seen_asins"].add("asin-committed")
            return {"message": "fast", "ask_attribute": "other", "recommendations": []}

        self.agent.baseline_agent.respond.side_effect = mutate_and_succeed
        self.agent.respond("commit", "I'm looking for waterproof boots.", 1, 10)

        state = self.agent._sessions["commit"]
        self.assertEqual(state["category"], "boots")
        self.assertEqual(state["disclosed_slots"]["feature"], {"waterproof"})
        self.assertEqual(state["accumulated_terms"], ["boots", "waterproof"])
        self.assertEqual(state["seen_asins"], {"asin-committed"})
        self.agent._respond_custom.assert_not_called()

    def test_rollback_preserves_shared_mapping_ownership(self) -> None:
        self.agent.reset("ownership", {})

        def mutate_then_fail(session_id: str, message: str, turn: int, top_k: int) -> dict:
            self.agent.baseline_agent._sessions[session_id]["category"] = "corrupted"
            raise RuntimeError("synthetic baseline failure")

        self.agent.baseline_agent.respond.side_effect = mutate_then_fail
        self.agent._respond_custom.return_value = {
            "message": "full",
            "ask_attribute": "other",
            "recommendations": [],
        }
        self.agent.respond("ownership", "I'm looking for boots.", 1, 10)

        self.assertIs(self.agent._sessions, self.agent.baseline_agent._sessions)
        self.assertIs(
            self.agent._sessions["ownership"],
            self.agent.baseline_agent._sessions["ownership"],
        )
        self.assertEqual(self.agent._sessions["ownership"]["category"], "clothing")

    def test_shared_mapping_and_both_reset_paths_keep_full_schema(self) -> None:
        self.assertIs(self.agent._sessions, self.agent.baseline_agent._sessions)

        self.agent.reset("agent-reset", {})
        agent_state = self.agent._sessions["agent-reset"]
        self.assertIs(agent_state, self.agent.baseline_agent._sessions["agent-reset"])
        self.assert_canonical_state(agent_state)

        self.agent.baseline_agent.reset("baseline-reset", {})
        baseline_state = self.agent._sessions["baseline-reset"]
        self.assertIs(baseline_state, self.agent.baseline_agent._sessions["baseline-reset"])
        self.assert_canonical_state(baseline_state)

    def test_dense_query_uses_only_active_semantic_state(self) -> None:
        state = agent_module.Agent._new_session_state()
        state.update({"category": "boots", "department": "shoes"})
        state["disclosed_slots"] = {"color": {"white"}}
        state["history"] = [{"role": "user", "content": "black boots"}]
        state["stashed_terms"] = ["black"]
        first = agent_module._state_to_retrieval_query(state)
        state["history"] = [{"role": "user", "content": "entirely different raw history"}]
        second = agent_module._state_to_retrieval_query(state)
        self.assertEqual(first, "boots shoes white")
        self.assertEqual(second, first)
        self.assertNotIn("black", first)

    def test_dense_query_multiple_values_are_deterministic(self) -> None:
        state = agent_module.Agent._new_session_state()
        state.update({"category": "Boots", "department": "Shoes"})
        state["disclosed_slots"] = {
            "Style": {"Zulu", "alpha"},
            "color": {"white", "Black"},
            "waterproof": {"yes"},
            "unused": {"no"},
        }
        self.assertEqual(
            agent_module._state_to_retrieval_query(state),
            "Boots Shoes Black white alpha Zulu waterproof",
        )

    @staticmethod
    def _ranking_metadata(searchable_bag: str, title: str) -> dict:
        return {
            "searchable_bag": searchable_bag,
            "brand": "",
            "rating_number": 1,
            "title": title,
        }

    def test_custom_ranking_does_not_boost_stashed_terms(self) -> None:
        self.agent.reset("custom-rank", {})
        state = self.agent._sessions["custom-rank"]
        state.update({
            "category": "",
            "department": "",
            "accumulated_terms": ["boots"],
            "stashed_terms": ["black"],
        })
        ids = ["plain"] + ["stashed"] + [f"filler-{i}" for i in range(8)]
        self.agent.catalog_ids = ids
        self.agent.catalog_ids_arr = np.array(ids)
        self.agent.catalog_prices = np.ones(len(ids))
        self.agent.catalog_categories_set = [set() for _ in ids]
        self.agent.catalog_metadata = {
            pid: self._ranking_metadata("boots black" if pid == "stashed" else "boots", pid)
            for pid in ids
        }
        self.agent.connection = MagicMock()
        self.agent.connection.execute.return_value.fetchall.return_value = [(pid,) for pid in ids]
        self.agent._update_state_via_llm = MagicMock()
        self.agent._call_llm = MagicMock(return_value="Here are matches.")
        result = agent_module.Agent._respond_custom(self.agent, "custom-rank", "show me options", 1, 1)
        self.assertEqual(result["recommendations"][0]["parent_asin"], "plain")

    def test_baseline_ranking_does_not_boost_stashed_terms(self) -> None:
        baseline = self.agent.baseline_agent
        baseline.reset("baseline-rank", {})
        state = baseline._sessions["baseline-rank"]
        state.update({"category": "", "department": "", "stashed_terms": ["black"]})
        baseline.catalog_ids = ["plain", "stashed"]
        baseline.catalog_metadata = {
            "plain": self._ranking_metadata("", "plain"),
            "stashed": self._ranking_metadata("black", "stashed"),
        }
        baseline.connection = MagicMock()
        baseline.connection.execute.return_value.fetchall.return_value = [("plain",), ("stashed",)]
        result = agent_module.BaselineAgent.respond(baseline, "baseline-rank", "unrouted text", 1, 1)
        self.assertEqual(result["recommendations"][0]["parent_asin"], "plain")

    def test_llm_disclosed_slots_replace_stale_mapping_and_terms(self) -> None:
        self.agent.reset("llm-replace", {})
        state = self.agent._sessions["llm-replace"]
        state["category"] = "boots"
        state["disclosed_slots"] = {"color": {"black"}, "material": {"leather"}}
        state["accumulated_terms"] = ["boots", "black", "leather"]
        response = {
            "category": "boots",
            "department": "shoes",
            "price_max": 9999.0,
            "disclosed_slots": {"color": "white"},
            "negated_terms": [],
            "asked_attributes": [],
        }
        self.agent._call_llm = MagicMock(return_value=json.dumps(response))
        self.agent._update_state_via_llm("llm-replace", "make them white")
        self.assertEqual(state["disclosed_slots"], {"color": {"white"}})
        self.assertIn("white", state["accumulated_terms"])
        self.assertNotIn("black", state["accumulated_terms"])
        self.assertNotIn("leather", state["accumulated_terms"])

    def test_lowercase_looking_for_matches_capitalized_category(self) -> None:
        baseline = self.agent.baseline_agent
        baseline.reset("lower", {})
        baseline.reset("upper", {})
        baseline._parse_message_locally("lower", "i'm looking for women's shoes")
        baseline._parse_message_locally("upper", "I'm looking for women's shoes")
        self.assertEqual(baseline._sessions["lower"]["category"], "women's shoes")
        self.assertEqual(baseline._sessions["lower"]["category"], baseline._sessions["upper"]["category"])

    def test_other_template_capitalization_variants_use_existing_branches(self) -> None:
        baseline = self.agent.baseline_agent
        baseline.reset("variants", {})
        baseline._sessions["variants"]["disclosed_slots"]["color"] = {"black"}
        baseline._parse_message_locally("variants", "i DON'T HAVE A PREFERENCE FOR color; please use your judgment.")
        self.assertNotIn("color", baseline._sessions["variants"]["disclosed_slots"])
        baseline._parse_message_locally("variants", "what i need is: WHITE.")
        self.assertEqual(baseline._sessions["variants"]["disclosed_slots"]["color"], {"WHITE"})
        baseline._parse_message_locally("variants", "a KEY REQUIREMENT IS: leather.")
        self.assertEqual(baseline._sessions["variants"]["disclosed_slots"]["material"], {"leather"})
        baseline._parse_message_locally("variants", "WHAT MATTERS IS: blue; cotton.")
        self.assertIn("blue", baseline._sessions["variants"]["disclosed_slots"]["color"])
        self.assertIn("cotton", baseline._sessions["variants"]["disclosed_slots"]["material"])


if __name__ == "__main__":
    unittest.main()
