from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys
import unittest
from unittest.mock import MagicMock, patch

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SHOPPING_AGENT_DIR = Path(__file__).resolve().parents[1]
for path in (PROJECT_ROOT, SHOPPING_AGENT_DIR):
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)

import agent as agent_module
from embedding_backends import OPENAI_EMBEDDING_SPACE_ID
from memory_adapter import FastMemoryQLMPAdapter, qlmp_items, qlmp_projection_memory
from memory_store import InMemoryUserMemoryStore
import run_longitudinal_eval as longitudinal
from nickolas.memory.qlmp import (
    MemoryItem,
    MemoryPolarity,
    MemorySource,
    build_cosine_memory_baseline,
    build_naive_memory_baseline,
    project_memory_residual,
)


def unit_vector(dimension: int = 3072, axis: int = 0) -> np.ndarray:
    value = np.zeros(dimension, dtype=np.float32)
    value[axis] = 1.0
    return value


def memory_item(identifier: str, polarity: MemoryPolarity = MemoryPolarity.POSITIVE) -> MemoryItem:
    return MemoryItem(
        id=identifier,
        text=f"feature: {identifier}",
        embedding=unit_vector(),
        source=MemorySource.USER,
        polarity=polarity,
        scope="boots",
    )


class FakeRunnerAgent:
    def __init__(self, target: str) -> None:
        self.target = target
        self.resets: list[tuple[str, dict, str, int]] = []
        self.ends: list[str] = []

    def reset(self, session_id, profile, *, user_id=None, sequence_index=None):
        self.resets.append((session_id, profile, user_id, sequence_index))

    def respond(self, session_id, user_message, turn, top_k):
        return {
            "message": "found it",
            "ask_attribute": "other",
            "recommendations": [{"parent_asin": self.target}],
        }

    def end_session(self, session_id, outcome=None, purchased_product=None, evidence=None):
        self.ends.append(session_id)


class Phase5LongitudinalMemoryTests(unittest.TestCase):
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
        ]
        for patcher in patchers:
            patcher.start()
            self.addCleanup(patcher.stop)
        self.agent = agent_module.Agent("unused-catalog.jsonl")
        self.addCleanup(self.agent.connection.close)
        self.addCleanup(self.agent.baseline_agent.connection.close)
        self.embedding_inputs: list[str] = []

        def embed(text: str) -> np.ndarray:
            self.embedding_inputs.append(text)
            return unit_vector()

        self.agent.embedding_backend.embed_query = MagicMock(side_effect=embed)

    def final_state(self, session_id: str) -> dict:
        state = self.agent._sessions[session_id]
        state["category"] = "running shoes"
        state["department"] = "shoes"
        state["disclosed_slots"] = {
            "color": {"white"},
            "material": {"mesh"},
            "feature": {"waterproof"},
        }
        state["negated_terms"] = {"black"}
        return state

    @staticmethod
    def runner_inputs(session_count: int = 1):
        profile = {
            "purchase_frequency": "regular",
            "average_prior_rating": None,
            "rating_style": "balanced",
            "preference_tags": [],
            "summary": "constant",
        }
        sessions = [
            {"sequence_index": index, "source_sample_id": f"sample_{index}"}
            for index in range(session_count)
        ]
        fixture = {"users": [{"user_id": "u1", "constant_profile": profile, "sessions": sessions}]}
        samples = {
            f"sample_{index}": {
                "sample_id": f"sample_{index}",
                "scenario_type": "buying",
                "ground_truth": {"parent_asin": "target"},
                "user_profile": {"summary": "source profile must be replaced"},
            }
            for index in range(session_count)
        }
        products = {"target": {"parent_asin": "target", "title": "Target"}}
        categories = {"target": ["clothing"]}
        return profile, fixture, samples, products, categories

    def test_01_backward_compatible_reset_is_anonymous(self) -> None:
        self.agent.reset("anonymous", {"summary": "ignored"})
        self.assertEqual(set(self.agent._sessions["anonymous"]), self.CANONICAL_KEYS)
        debug = self.agent.get_memory_debug("anonymous")
        self.assertIsNone(debug["user_id"])
        self.assertEqual(debug["visible_prior_memory_count"], 0)
        self.assertFalse(debug["historical_memory_applied"])
        self.assertIsNone(self.agent.end_session("anonymous", outcome={"hit": True}))
        self.agent.embedding_backend.embed_query.assert_not_called()

    def test_02_longitudinal_metadata_stays_outside_fast_memory(self) -> None:
        self.agent.reset("s3", {}, user_id="u1", sequence_index=3)
        debug = self.agent.get_memory_debug("s3")
        self.assertEqual((debug["user_id"], debug["sequence_index"]), ("u1", 3))
        state = self.agent._sessions["s3"]
        self.assertEqual(set(state), self.CANONICAL_KEYS)
        query = agent_module._state_to_retrieval_query(state)
        self.assertNotIn("u1", query)
        self.assertNotIn("s3", query)

    def test_03_final_fast_memory_extracts_only_atomic_active_facts(self) -> None:
        state = agent_module.Agent._new_session_state()
        state.update(
            {
                "category": "Running Shoes",
                "disclosed_slots": {
                    "color": {"White"},
                    "material": {"Mesh"},
                    "feature": {"Waterproof"},
                },
                "negated_terms": {"Black"},
                "price_max": 150.0,
                "stashed_terms": ["leather"],
                "history": [{"content": "secret transcript"}],
                "seen_asins": {"target-asin"},
                "debug_info": {"rank": 1},
                "accumulated_terms": ["duplicate", "waterproof"],
            }
        )
        drafts = FastMemoryQLMPAdapter.extract_drafts(
            state,
            user_id="u1",
            session_id="s0",
            sequence_index=0,
        )
        self.assertEqual(
            [draft.text for draft in drafts],
            [
                "category: running shoes",
                "budget: at most 150",
                "color: white",
                "feature: waterproof",
                "material: mesh",
                "avoid: black",
            ],
        )
        rendered = " ".join(draft.text for draft in drafts)
        for excluded in ("leather", "secret transcript", "target-asin", "duplicate", "rank"):
            self.assertNotIn(excluded, rendered)

    def test_04_negative_memory_keeps_negative_polarity(self) -> None:
        self.agent.reset("s0", {}, user_id="u1", sequence_index=0)
        self.final_state("s0")
        items = self.agent.end_session("s0")
        negative = next(item for item in items if item.text == "avoid: black")
        self.assertIs(negative.polarity, MemoryPolarity.NEGATIVE)

    def test_05_memory_uses_exact_m0_embedding_space(self) -> None:
        self.agent.reset("s0", {}, user_id="u1", sequence_index=0)
        self.final_state("s0")
        self.agent.end_session("s0")
        records = self.agent.memory_store.snapshot("u1")
        self.assertTrue(records)
        self.assertEqual(
            {record.embedding_space_id for record in records},
            {OPENAI_EMBEDDING_SPACE_ID},
        )
        self.assertTrue(all(np.linalg.norm(record.item.embedding) == 1.0 for record in records))

    def test_06_end_session_commits_and_next_session_sees_prior_items(self) -> None:
        self.agent.reset("s0", {}, user_id="u1", sequence_index=0)
        self.final_state("s0")
        created = self.agent.end_session("s0")
        self.agent.reset("s1", {}, user_id="u1", sequence_index=1)
        visible = self.agent.get_visible_memories("s1")
        self.assertEqual([item.id for item in visible], [item.id for item in created])
        self.assertEqual(self.agent.get_memory_debug("s1")["visible_prior_memory_count"], 5)

    def test_07_no_future_session_leakage(self) -> None:
        store = InMemoryUserMemoryStore()
        store.add_memories(
            user_id="u1",
            session_id="s0",
            sequence_index=0,
            embedding_space_id=OPENAI_EMBEDDING_SPACE_ID,
            memories=[memory_item("past")],
        )
        store.add_memories(
            user_id="u1",
            session_id="s2",
            sequence_index=2,
            embedding_space_id=OPENAI_EMBEDDING_SPACE_ID,
            memories=[memory_item("future")],
        )
        self.assertEqual(
            [item.id for item in store.get_memories("u1", before_sequence_index=2)],
            ["past"],
        )

    def test_08_cross_user_isolation(self) -> None:
        store = InMemoryUserMemoryStore()
        store.add_memories(
            user_id="u1",
            session_id="u1_s0",
            sequence_index=0,
            embedding_space_id=OPENAI_EMBEDDING_SPACE_ID,
            memories=[memory_item("u1-only")],
        )
        self.assertEqual(store.get_memories("u2", before_sequence_index=1), ())

    def test_09_longitudinal_runner_scores_before_end_session(self) -> None:
        _profile, fixture, samples, products, categories = self.runner_inputs()
        fake = FakeRunnerAgent("target")
        events: list[str] = []

        def hook(event, session_id, scored):
            events.append(event)
            if event == "scored":
                self.assertTrue(scored["hit"])
                self.assertEqual(fake.ends, [])

        longitudinal.run_longitudinal_evaluation(
            fake,
            fixture,
            samples,
            {"target"},
            categories,
            products,
            shopper_call=lambda prompt, system, model: "I'm looking for clothing.",
            system_prompt_builder=lambda sample, product, category: "persona",
            hidden_field_builder=lambda sample, products: ({"hard_constraints": []}, {}),
            event_hook=hook,
        )
        self.assertEqual(events, ["scored", "ended"])
        self.assertEqual(fake.ends, ["u1_s0"])

    def test_10_outcome_purchase_and_target_never_become_memory_facts(self) -> None:
        self.agent.reset("s0", {}, user_id="u1", sequence_index=0)
        self.final_state("s0")
        self.agent.end_session(
            "s0",
            outcome={"hit": True, "best_rank": 1, "reciprocal_rank": 1.0},
            purchased_product="TARGET-ASIN-SECRET",
            evidence={"target_asin": "TARGET-ASIN-SECRET"},
        )
        text = " ".join(record.item.text for record in self.agent.memory_store.snapshot("u1"))
        embedded = " ".join(self.embedding_inputs)
        for forbidden in ("target-asin-secret", "best_rank", "reciprocal_rank"):
            self.assertNotIn(forbidden, text.casefold())
            self.assertNotIn(forbidden, embedded.casefold())
        self.assertNotIn("hit:", text.casefold())
        self.assertNotIn("hit:", embedded.casefold())
        self.agent.reset("s1", {}, user_id="u1", sequence_index=1)
        self.assertEqual(self.agent._sessions["s1"], agent_module.Agent._new_session_state())

    def test_11_shadow_history_is_never_consulted_by_ranking(self) -> None:
        self.agent.reset("s0", {}, user_id="u1", sequence_index=0)
        self.final_state("s0")
        self.agent.end_session("s0")
        self.agent.reset("with-history", {}, user_id="u1", sequence_index=1)
        self.agent.get_visible_memories = MagicMock(side_effect=AssertionError("ranking read memory"))
        expected = {
            "message": "same",
            "ask_attribute": "other",
            "recommendations": [{"parent_asin": "a"}, {"parent_asin": "b"}],
        }
        self.agent._respond_custom = MagicMock(return_value=deepcopy(expected))
        with_history = self.agent.respond("with-history", "natural request", 1, 10)

        fresh = agent_module.Agent("unused-catalog.jsonl")
        self.addCleanup(fresh.connection.close)
        self.addCleanup(fresh.baseline_agent.connection.close)
        fresh.reset("without-history", {})
        fresh._respond_custom = MagicMock(return_value=deepcopy(expected))
        without_history = fresh.respond("without-history", "natural request", 1, 10)
        self.assertEqual(with_history, without_history)
        self.agent.get_visible_memories.assert_not_called()

    def test_12_stored_items_feed_existing_qlmp_helpers_directly(self) -> None:
        store = InMemoryUserMemoryStore()
        positive = memory_item("positive")
        negative = memory_item("negative", MemoryPolarity.NEGATIVE)
        records = store.add_memories(
            user_id="u1",
            session_id="s0",
            sequence_index=0,
            embedding_space_id=OPENAI_EMBEDDING_SPACE_ID,
            memories=[positive, negative],
        )
        items = qlmp_items(records, expected_embedding_space_id=OPENAI_EMBEDDING_SPACE_ID)
        q = unit_vector().astype(np.float64)
        naive = build_naive_memory_baseline(q, items, query_scope="boots")
        cosine = build_cosine_memory_baseline(q, items, query_scope="boots")
        projection = project_memory_residual(
            q,
            qlmp_projection_memory(items[0]),
            np.empty((q.size, 0)),
        )
        self.assertEqual(naive.selected_memory_ids, ("positive",))
        self.assertEqual(cosine.selected_memory_ids, ("positive",))
        self.assertEqual(projection.projected_residual.shape, q.shape)

    def test_13_fixture_sessions_are_sorted_chronologically(self) -> None:
        fixture = {
            "users": [{
                "user_id": "u1",
                "constant_profile": {},
                "sessions": [
                    {"sequence_index": 2, "source_sample_id": "c"},
                    {"sequence_index": 0, "source_sample_id": "a"},
                    {"sequence_index": 1, "source_sample_id": "b"},
                ],
            }]
        }
        ordered = longitudinal.ordered_fixture_users(fixture)
        self.assertEqual(
            [session["sequence_index"] for session in ordered[0]["sessions"]],
            [0, 1, 2],
        )

    def test_14_one_constant_profile_object_is_reused_without_shopper_history(self) -> None:
        profile, fixture, samples, products, categories = self.runner_inputs(2)
        fake = FakeRunnerAgent("target")
        observed_profiles: list[dict] = []
        observed_prompts: list[str] = []

        def prompt_builder(sample, product, category):
            observed_profiles.append(sample["user_profile"])
            return f"profile={sample['user_profile']['summary']}"

        def shopper(prompt, system, model):
            observed_prompts.append(prompt)
            return "I'm looking for clothing."

        longitudinal.run_longitudinal_evaluation(
            fake,
            fixture,
            samples,
            {"target"},
            categories,
            products,
            shopper_call=shopper,
            system_prompt_builder=prompt_builder,
            hidden_field_builder=lambda sample, products: ({"hard_constraints": []}, {}),
        )
        self.assertTrue(all(value is profile for value in observed_profiles))
        self.assertTrue(all(reset[1] is profile for reset in fake.resets))
        self.assertEqual(observed_prompts, [
            "Start the conversation by telling the assistant what you are looking for.",
            "Start the conversation by telling the assistant what you are looking for.",
        ])

    def test_15_evaluator_annotations_never_reach_agent_or_shopper_sample(self) -> None:
        _profile, fixture, samples, products, categories = self.runner_inputs()
        fixture["users"][0]["sessions"][0].update(
            {
                "memory_relevance": "helpful",
                "expected_conflict": True,
                "target_attribute_audit": {"secret": "annotation"},
            }
        )
        samples["sample_0"]["memory_relevance"] = "stale"
        fake = FakeRunnerAgent("target")

        def prompt_builder(sample, product, category):
            self.assertTrue(longitudinal.EVALUATOR_ONLY_FIELDS.isdisjoint(sample))
            return "persona"

        result = longitudinal.run_longitudinal_evaluation(
            fake,
            fixture,
            samples,
            {"target"},
            categories,
            products,
            shopper_call=lambda prompt, system, model: "I'm looking for clothing.",
            system_prompt_builder=prompt_builder,
            hidden_field_builder=lambda sample, products: ({"hard_constraints": []}, {}),
        )
        annotations = result["sessions"][0]["evaluation_annotations"]
        self.assertEqual(annotations["memory_relevance"], "helpful")
        self.assertTrue(annotations["expected_conflict"])

    def test_16_store_instances_are_fresh_and_non_global(self) -> None:
        first = InMemoryUserMemoryStore()
        second = InMemoryUserMemoryStore()
        first.add_memories(
            user_id="u1",
            session_id="s0",
            sequence_index=0,
            embedding_space_id=OPENAI_EMBEDDING_SPACE_ID,
            memories=[memory_item("only-first")],
        )
        self.assertEqual(second.snapshot(), ())
        first.clear()
        self.assertEqual(first.snapshot(), ())


if __name__ == "__main__":
    unittest.main()
