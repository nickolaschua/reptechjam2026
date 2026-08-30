from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SHOPPING_AGENT_DIR = Path(__file__).resolve().parents[1]
for path in (PROJECT_ROOT, SHOPPING_AGENT_DIR):
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)

from memory_adapter import FastMemoryQLMPAdapter
from memory_store import InMemoryUserMemoryStore, MemoryStoreSnapshot
import run_longitudinal_eval as longitudinal
import run_b0_validation as b0_validation
from longitudinal_eval.directives import (
    build_directive_system_prompt,
    established_facts_before,
    semantic_disclosure_validation,
)
from longitudinal_eval.microbenchmark import load_cases, validate_cases
from longitudinal_eval.validate_fixture import load_jsonl, validate_fixture
from nickolas.memory.qlmp import MemoryItem, MemoryPolarity, MemorySource


DATA_DIR = PROJECT_ROOT / "techjam-conversational-search" / "data"
FIXTURE_DIR = SHOPPING_AGENT_DIR / "longitudinal_eval"


def item(identifier: str, text: str = "feature: breathable") -> MemoryItem:
    return MemoryItem(
        id=identifier,
        text=text,
        embedding=np.asarray([1.0, 0.0], dtype=np.float64),
        source=MemorySource.USER,
        polarity=MemoryPolarity.POSITIVE,
        scope="clothing",
    )


class DirectiveSpyAgent:
    def __init__(self, target: str = "target") -> None:
        self.target = target
        self.reset_calls = []
        self.respond_calls = []
        self.ended = False
        self.embedding_space_id = "test-space"

    def reset(self, session_id, profile, *, user_id=None, sequence_index=None):
        self.reset_calls.append((session_id, profile, user_id, sequence_index))

    def get_visible_memories(self, session_id):
        return ()

    def respond(self, session_id, user_message, turn, top_k):
        self.respond_calls.append((session_id, user_message, turn, top_k))
        return {
            "message": "found",
            "ask_attribute": "material",
            "recommendations": [{"parent_asin": self.target}],
        }

    def get_memory_debug(self, session_id):
        return {
            "final_fast_memory": {
                "category": "shirts",
                "disclosed_slots": {"material": ["breathable cotton"]},
                "negated_terms": [],
            },
            "historical_memory_applied": False,
        }

    def end_session(self, session_id, outcome=None, purchased_product=None, evidence=None):
        self.ended = True
        return (item("committed", "material: breathable cotton"),)


class ReplayAgent:
    def __init__(self, store: InMemoryUserMemoryStore) -> None:
        self.memory_store = store
        self.embedding_space_id = "test-space"
        self.active = {}

    def reset(self, session_id, profile, *, user_id=None, sequence_index=None):
        self.memory_store.validate_new_session(user_id, session_id, sequence_index)
        self.active[session_id] = (user_id, sequence_index)

    def get_visible_memories(self, session_id):
        user, index = self.active[session_id]
        return self.memory_store.get_memories(user, before_sequence_index=index)

    def respond(self, session_id, user_message, turn, top_k):
        return {"message": "keep looking", "ask_attribute": "other", "recommendations": []}

    def get_memory_debug(self, session_id):
        return {"final_fast_memory": {}, "historical_memory_applied": False}

    def end_session(self, session_id, outcome=None, purchased_product=None, evidence=None):
        user, index = self.active[session_id]
        self.memory_store.add_memories(
            user_id=user,
            session_id=session_id,
            sequence_index=index,
            embedding_space_id="test-space",
            memories=[],
        )
        return ()


class FixedTranscriptParityAgent:
    def __init__(self, store: InMemoryUserMemoryStore) -> None:
        self.memory_store = store
        self.embedding_space_id = "test-space"
        self.active = {}
        self.respond_calls = []
        self.messages = {}

    def reset(self, session_id, profile, *, user_id=None, sequence_index=None):
        self.memory_store.validate_new_session(user_id, session_id, sequence_index)
        self.active[session_id] = (user_id, sequence_index)
        self.messages[session_id] = []

    def get_visible_memories(self, session_id):
        user, index = self.active[session_id]
        return self.memory_store.get_memories(user, before_sequence_index=index)

    def respond(self, session_id, user_message, turn, top_k):
        self.respond_calls.append((session_id, user_message, turn, top_k))
        self.messages[session_id].append(user_message)
        return {
            "message": "stable prose",
            "ask_attribute": "material",
            "recommendations": [
                {"parent_asin": "b"},
                {"parent_asin": "a"},
            ],
        }

    def get_memory_debug(self, session_id):
        return {
            "final_fast_memory": {"messages": list(self.messages[session_id])},
            "historical_memory_applied": False,
        }

    def end_session(self, session_id, outcome=None, purchased_product=None, evidence=None):
        user, index = self.active[session_id]
        memory = item(f"{session_id}-memory", f"feature: session {index}")
        self.memory_store.add_memories(
            user_id=user,
            session_id=session_id,
            sequence_index=index,
            embedding_space_id="test-space",
            memories=[memory],
        )
        return (memory,)


class StochasticParityAgent(FixedTranscriptParityAgent):
    def __init__(self, store, generated_order):
        super().__init__(store)
        self.generated_order = generated_order

    def _call_llm(
        self, prompt, system_prompt="", session_id=None, response_json=False
    ):
        return self.generated_order

    def respond(self, session_id, user_message, turn, top_k):
        self.respond_calls.append((session_id, user_message, turn, top_k))
        self.messages[session_id].append(user_message)
        order = self._call_llm(
            f"state:{user_message}",
            "fixed-system",
            session_id=session_id,
            response_json=True,
        ).split(",")
        return {
            "message": "stable prose",
            "ask_attribute": "material",
            "recommendations": [{"parent_asin": value} for value in order],
        }


class Phase6BenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.fixture = json.loads((FIXTURE_DIR / "users_40.json").read_text(encoding="utf-8"))
        cls.small_fixture = json.loads((FIXTURE_DIR / "fixture_small.json").read_text(encoding="utf-8"))
        cls.samples = {value["sample_id"]: value for value in load_jsonl(DATA_DIR / "public_set.jsonl")}
        cls.products = {value["parent_asin"]: value for value in load_jsonl(DATA_DIR / "catalog.jsonl")}

    def test_01_research_fixture_shape_profiles_and_ids(self) -> None:
        users = longitudinal.ordered_fixture_users(self.fixture)
        self.assertEqual(len(users), 4)
        runtime_ids = []
        for user in users:
            self.assertEqual(len(user["sessions"]), 10)
            self.assertEqual([value["sequence_index"] for value in user["sessions"]], list(range(10)))
            profile = user["constant_profile"]
            self.assertTrue(all(profile is user["constant_profile"] for _ in user["sessions"]))
            runtime_ids.extend(f"{user['user_id']}_s{value['sequence_index']}" for value in user["sessions"])
        self.assertEqual(len(runtime_ids), len(set(runtime_ids)))

    def test_02_full_validator_resolves_sources_targets_and_private_profiles(self) -> None:
        result = validate_fixture(self.fixture, self.samples, self.products)
        self.assertTrue(result["valid"], result["errors"])
        self.assertEqual((result["user_count"], result["session_count"]), (4, 40))
        for user in self.fixture["users"]:
            profile_text = json.dumps(user["constant_profile"]).casefold()
            for preference in user["shopper_private_persona"]["latent_preferences"]:
                for term in preference["leakage_terms"]:
                    self.assertNotIn(term.casefold(), profile_text)

    def test_03_every_session_is_ordered_real_and_source_target_agrees(self) -> None:
        for user in self.fixture["users"]:
            for session in user["sessions"]:
                source = self.samples[session["source_sample_id"]]
                self.assertEqual(source["scenario_type"], "buying")
                self.assertEqual(source["ground_truth"]["parent_asin"], session["target_asin"])
                self.assertIn(session["target_asin"], self.products)
                for index in session["relevant_prior_sequence_indices"]:
                    self.assertLess(index, session["sequence_index"])

    def test_04_probes_and_u3_history_sizes(self) -> None:
        plan = longitudinal.probe_replay_plan(self.fixture)
        self.assertEqual(len(plan), 11)
        by_user = {}
        for value in plan:
            by_user.setdefault(value["user_id"], []).append(value["history_size"])
        self.assertEqual(by_user["u3_distractor"], [0, 1, 3, 5, 9])
        for user in self.fixture["users"]:
            self.assertIn("probe", user["sessions"][9]["session_role"])
        self.assertTrue(self.fixture["users"][1]["sessions"][9]["longitudinal_directive"]["current_override"])

    def test_05_future_preferences_are_not_injected_early(self) -> None:
        user = self.fixture["users"][0]
        self.assertEqual(established_facts_before(user["sessions"], 1), ())
        before_p2 = " ".join(established_facts_before(user["sessions"], 3)).casefold()
        self.assertIn("breathable", before_p2)
        self.assertNotIn("neutral", before_p2)
        self.assertNotIn("120", before_p2)
        prompt = build_directive_system_prompt("base", {}, ["prefers breathable"], is_probe=True)
        self.assertIn("Do not volunteer any established historical fact", prompt)

    def test_06_directives_are_private_and_disclosure_chain_logs(self) -> None:
        directive = {
            "disclose": ["Across purchases, I generally prefer breathable materials."],
            "reinforce": [], "session_only": [], "current_override": [],
        }
        profile = {"summary": "broad", "preference_tags": []}
        fixture = {"users": [{
            "user_id": "u", "constant_profile": profile,
            "sessions": [{
                "sequence_index": 0, "source_sample_id": "s", "target_asin": "target",
                "session_role": "establish", "longitudinal_directive": directive,
            }],
        }]}
        samples = {"s": {"sample_id": "s", "scenario_type": "buying", "ground_truth": {"parent_asin": "target"}, "user_profile": {}}}
        products = {"target": {"parent_asin": "target", "title": "Plain Shirt"}}
        prompts = []
        fake = DirectiveSpyAgent()

        def shopper(prompt, system, model):
            prompts.append((prompt, system))
            return "i'm looking for a shirt and i generally prefer breathable cotton"

        result = longitudinal.run_longitudinal_evaluation(
            fake, fixture, samples, {"target"}, {"target": ["shirts"]}, products,
            shopper_call=shopper,
            system_prompt_builder=lambda sample, product, category: "base shopper prompt",
            hidden_field_builder=lambda sample, products: ({"hard_constraints": []}, {}),
        )
        self.assertEqual(fake.reset_calls, [("u_s0", profile, "u", 0)])
        self.assertEqual(len(fake.respond_calls[0]), 4)
        self.assertIsNot(fake.respond_calls[0][1], directive)
        self.assertIn("PRIVATE LONGITUDINAL EVALUATOR CONTROL", prompts[0][1])
        chain = result["sessions"][0]["semantic_disclosure_validation"][0]
        self.assertTrue(chain["shopper_expressed"])
        self.assertTrue(chain["fast_memory_captured"])
        self.assertTrue(chain["memory_committed"])
        self.assertFalse(result["sessions"][0]["target_leakage"]["leaked"])
        self.assertFalse(result["sessions"][0]["historical_memory_applied"])

    def test_07_session_only_is_not_filtered_and_annotations_do_not_commit(self) -> None:
        state = {
            "category": "rain jackets",
            "price_max": 9999.0,
            "disclosed_slots": {"feature": {"waterproof hooded"}},
            "negated_terms": set(),
        }
        drafts = FastMemoryQLMPAdapter.extract_drafts(
            state, user_id="u", session_id="u_s0", sequence_index=0
        )
        rendered = " ".join(value.text for value in drafts)
        self.assertIn("waterproof hooded", rendered)
        self.assertNotIn("session_only", rendered)
        self.assertNotIn("target_attribute_audit", rendered)

    def test_08_snapshot_round_trip_preserves_zero_commits_vectors_and_space(self) -> None:
        store = InMemoryUserMemoryStore()
        store.add_memories(
            user_id="u", session_id="u_s0", sequence_index=0,
            embedding_space_id="test-space", memories=[item("m0")],
        )
        store.add_memories(
            user_id="u", session_id="u_s1", sequence_index=1,
            embedding_space_id="test-space", memories=[],
        )
        snapshot = store.export_snapshot()
        payload = snapshot.to_payload(include_embeddings=True)
        restored_snapshot = MemoryStoreSnapshot.from_payload(payload)
        restored = InMemoryUserMemoryStore()
        restored.import_snapshot(restored_snapshot, expected_embedding_space_id="test-space")
        self.assertEqual([value.session_id for value in restored.commits_for_user("u")], ["u_s0", "u_s1"])
        self.assertEqual(restored.commits_for_user("u")[1].memory_count, 0)
        np.testing.assert_array_equal(restored.get_memories("u")[0].embedding, [1.0, 0.0])

    def test_09_counterfactual_replay_starts_clean_and_reuses_exact_probe(self) -> None:
        history = InMemoryUserMemoryStore()
        for user in self.fixture["users"]:
            for index in range(9):
                history.add_memories(
                    user_id=user["user_id"], session_id=f"{user['user_id']}_s{index}",
                    sequence_index=index, embedding_space_id="test-space", memories=[],
                )
        starting_counts = []

        def factory(store):
            starting_counts.append(len(store.export_snapshot().commits))
            return ReplayAgent(store)

        result = longitudinal.run_counterfactual_probe_replays(
            factory, history.export_snapshot(), self.fixture, self.samples,
            set(self.products),
            {asin: product.get("categories", []) for asin, product in self.products.items()},
            self.products,
            shopper_call=lambda prompt, system, model: "i'm looking for something suitable",
            system_prompt_builder=lambda sample, product, category: "base",
            hidden_field_builder=lambda sample, products: ({"hard_constraints": []}, {}),
            max_turns=1,
        )
        self.assertEqual(starting_counts.count(0), 4)
        self.assertIn(9, starting_counts)
        self.assertEqual(len(result["sessions"]), 11)
        for value in result["comparisons"]["per_user"].values():
            self.assertTrue(value["same_probe_config"])
            self.assertIsNone(value["memory_lift"])
        self.assertEqual(result["comparisons"]["status"], "shadow_mode_not_interpreted")

    def test_10_small_fixture_and_microbenchmark_remain_valid(self) -> None:
        small = validate_fixture(
            self.small_fixture, self.samples, self.products, require_research_shape=False
        )
        self.assertTrue(small["valid"], small["errors"])
        micro = validate_cases(load_cases())
        self.assertTrue(micro["valid"], micro["errors"])
        self.assertEqual(micro["case_count"], 4)

    def test_11_fixed_transcript_parity_is_ordered_clean_and_shadow_visible(self) -> None:
        fixture = {
            "users": [{
                "user_id": "u",
                "constant_profile": {},
                "sessions": [
                    {"sequence_index": 0, "source_sample_id": "s0", "target_asin": "a"},
                    {"sequence_index": 1, "source_sample_id": "s1", "target_asin": "a"},
                ],
            }]
        }
        captured = [
            {"session_id": "u_s0", "turns": [{"shopper": "first exact message"}]},
            {"session_id": "u_s1", "turns": [{"shopper": "second exact message"}]},
        ]
        agents = []

        def factory(store):
            agent = FixedTranscriptParityAgent(store)
            agents.append(agent)
            return agent

        result = longitudinal.run_strict_shadow_no_history_parity(
            factory, fixture, captured, {"a", "b"}
        )
        self.assertEqual(result["total_paired_turns"], 2)
        self.assertEqual(result["identical_recommendation_turns"], 2)
        self.assertEqual(result["differing_recommendation_turns"], 0)
        self.assertTrue(result["all_shopper_inputs_identical"])
        self.assertEqual(result["no_history_sessions_with_prior_memory"], [])
        second = next(
            value for value in result["session_checks"] if value["session_id"] == "u_s1"
        )
        self.assertEqual(second["shadow_prior_memory_count"], 1)
        self.assertEqual(second["no_history_prior_memory_count"], 0)
        self.assertEqual(
            [call[1] for agent in agents for call in agent.respond_calls],
            [
                "first exact message",
                "second exact message",
                "first exact message",
                "second exact message",
            ],
        )
        self.assertTrue(all(len(call) == 4 for agent in agents for call in agent.respond_calls))

    def test_12_parity_comparison_detects_recommendation_order_not_just_sets(self) -> None:
        def condition(name, recommendations):
            return {
                "condition": name,
                "sessions": [{
                    "session_id": "u_s0",
                    "shopper_inputs_sha256": "same",
                    "shopper_inputs": ["same message"],
                    "prior_visible_memory_items": [],
                    "historical_memory_applied": False,
                    "turns": [{
                        "turn": 1,
                        "shopper_input_sha256": "same-turn",
                        "recommendations": recommendations,
                        "target_rank": 1,
                        "ask_attribute": "other",
                        "fast_memory": {},
                        "route": "fast",
                        "agent_message": "same",
                    }],
                }],
            }

        result = longitudinal.compare_fixed_transcript_conditions(
            condition("SHADOW_HISTORY", ["a", "b"]),
            condition("NO_HISTORY", ["b", "a"]),
        )
        self.assertEqual(result["differing_recommendation_turns"], 1)
        self.assertEqual(result["recommendation_order_parity_rate"], 0.0)

    def test_13_b0_output_guard_keeps_frozen_and_phase4_artifacts_read_only(self) -> None:
        frozen_hashes = b0_validation.file_hashes(b0_validation.FROZEN_M0_DIR)
        with self.assertRaises(ValueError):
            b0_validation.validate_output_directory(
                b0_validation.FROZEN_M0_DIR / "phase61"
            )
        with self.assertRaises(ValueError):
            b0_validation.validate_output_directory(
                SHOPPING_AGENT_DIR / "baseline_results" / "new_baseline"
            )
        with tempfile.TemporaryDirectory() as directory:
            allowed = b0_validation.validate_output_directory(Path(directory) / "b0")
            self.assertEqual(allowed, (Path(directory) / "b0").resolve())
        self.assertEqual(
            frozen_hashes,
            b0_validation.file_hashes(b0_validation.FROZEN_M0_DIR),
        )

    def test_14_strict_parity_replays_identical_agent_llm_calls(self) -> None:
        fixture = {
            "users": [{
                "user_id": "u",
                "constant_profile": {},
                "sessions": [{
                    "sequence_index": 0,
                    "source_sample_id": "s0",
                    "target_asin": "a",
                }],
            }]
        }
        captured = [{
            "session_id": "u_s0",
            "turns": [{"shopper": "identical shopper input"}],
        }]
        generated = iter(("a,b", "b,a"))

        def factory(store):
            return StochasticParityAgent(store, next(generated))

        result = longitudinal.run_strict_shadow_no_history_parity(
            factory, fixture, captured, {"a", "b"}
        )
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["identical_recommendation_turns"], 1)
        self.assertTrue(result["agent_llm_call_control"]["enabled"])
        self.assertEqual(result["agent_llm_call_control"]["recorded_call_count"], 1)
        self.assertEqual(result["agent_llm_call_control"]["replayed_call_count"], 1)
        self.assertEqual(result["agent_llm_call_control"]["prompt_mismatch_count"], 0)


if __name__ == "__main__":
    unittest.main()
