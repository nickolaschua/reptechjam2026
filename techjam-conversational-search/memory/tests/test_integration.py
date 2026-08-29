from __future__ import annotations

import unittest

import numpy as np

from memory import MemoryConfig, MemorySystem, SlowMemoryEpisode
from memory.tests.helpers import (
    KeywordEmbeddingProvider,
    PRODUCTS,
    TemporaryCatalog,
    catalog_embeddings,
)
from starter.agent import Agent


class CountingProvider(KeywordEmbeddingProvider):
    def __init__(self) -> None:
        self.calls: list[tuple[str, ...]] = []

    def embed(self, texts):
        self.calls.append(tuple(str(value) for value in texts))
        return super().embed(texts)


class IntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.catalog = TemporaryCatalog()
        self.ids = [str(product["parent_asin"]) for product in PRODUCTS]

    def tearDown(self) -> None:
        self.catalog.close()

    def system(
        self,
        *,
        enabled: bool = True,
        weight: float = 0.02,
        provider: KeywordEmbeddingProvider | None = None,
        vectors: np.ndarray | None = None,
    ) -> MemorySystem:
        encoder = provider or KeywordEmbeddingProvider()
        return MemorySystem(
            self.catalog.path,
            self.ids,
            PRODUCTS,
            config=MemoryConfig(memory_enabled=enabled, lambda_memory=weight),
            embedding_provider=encoder,
            catalog_embeddings=catalog_embeddings(encoder) if vectors is None else vectors,
        )

    @staticmethod
    def finish(system: MemorySystem, user: str, session: str, sequence: int, message: str):
        system.begin_session(user, session, sequence_index=sequence)
        system.update_session(session, message, 1)
        return system.end_session(session, {"status": "ignored"}, "ignored-product", ["ignored"])

    def test_m0_and_m1_commit_identical_episodes_but_only_m1_reranks(self) -> None:
        m0 = self.system(enabled=False, weight=1.0)
        m1 = self.system(enabled=True, weight=1.0)
        for system in (m0, m1):
            self.finish(system, "user", "history", 0, "I'm looking for shoes. red hiking")
            system.begin_session("user", "current", sequence_index=1)
            system.update_session("current", "I'm looking for shoes, but I'm still exploring.", 1)

        baseline = ["a-blue-shoe", "b-black-dress", "z-red-shoe", "c-green-shoe"]
        self.assertEqual(m0.rerank_candidates("current", baseline), baseline)
        self.assertNotEqual(m1.rerank_candidates("current", baseline), baseline)
        self.assertEqual(
            m0.store.episodes_for_user("user")[0].summary_text,
            m1.store.episodes_for_user("user")[0].summary_text,
        )

    def test_user_isolation_begin_snapshot_and_active_session_exclusion(self) -> None:
        system = self.system(weight=1.0)
        self.finish(system, "left", "left-0", 0, "I'm looking for shoes. red")
        system.begin_session("left", "left-1", sequence_index=1)
        system.update_session("left-1", "I'm looking for shoes. blue", 1)
        system.begin_session("right", "right-0", sequence_index=0)
        system.update_session("right-0", "I'm looking for shoes, but I'm still exploring.", 1)

        self.assertEqual(
            [item.session_id for item in system.store.visible_episodes("left-1")],
            ["left-0"],
        )
        self.assertEqual(system.store.visible_episodes("right-0"), ())
        self.assertNotIn("left-1", [
            item.session_id for item in system.store.visible_episodes("left-1")
        ])

    def test_sequence_monotonicity_duplicate_commit_and_cleanup(self) -> None:
        system = self.system()
        episode = self.finish(system, "user", "s0", 0, "I'm looking for shoes. red")
        self.assertEqual(episode.sequence_index, 0)
        self.assertNotIn("s0", system._states)
        with self.assertRaises(RuntimeError):
            system.end_session("s0")
        with self.assertRaises(ValueError):
            system.begin_session("user", "old", sequence_index=0)
        self.assertEqual(
            system.begin_session("user", "gap", sequence_index=3).sequence_index,
            3,
        )

    def test_summary_is_embedded_exactly_once_and_outcomes_never_enrich_it(self) -> None:
        provider = CountingProvider()
        matrix = catalog_embeddings(provider)
        provider.calls.clear()
        system = self.system(provider=provider, vectors=matrix)
        system.begin_session("user", "s0", sequence_index=0)
        system.update_session("s0", "I'm looking for shoes. red", 1)
        episode = system.end_session(
            "s0",
            {"status": "purchased", "target": "secret"},
            purchased_product="z-red-shoe",
            evidence={"value": "secret-evidence"},
        )
        self.assertEqual(len(provider.calls), 1)
        self.assertEqual(provider.calls[0], (episode.summary_text,))
        self.assertNotIn("secret", episode.summary_text)
        self.assertNotIn("purchased", episode.summary_text)

    def test_missing_or_incompatible_product_vectors_preserve_exact_order(self) -> None:
        provider = KeywordEmbeddingProvider()
        missing = MemorySystem(
            self.catalog.path,
            self.ids,
            config=MemoryConfig(),
            embedding_provider=provider,
        )
        self.finish(missing, "user", "s0", 0, "I'm looking for shoes. red")
        missing.begin_session("user", "s1", sequence_index=1)
        missing.update_session("s1", "I'm looking for shoes, but I'm still exploring.", 1)
        baseline = ["z-red-shoe", "a-blue-shoe"]
        self.assertEqual(missing.rerank_candidates("s1", baseline), baseline)

        incompatible = self.system(weight=1.0)
        incompatible.store._episodes["user"] = (SlowMemoryEpisode(
            "user", "old", 0, "old", (1.0, 0.0, 0.0, 0.0), "another-space"
        ),)
        incompatible.begin_session("user", "new", sequence_index=1)
        incompatible.update_session("new", "I'm looking for shoes, but I'm still exploring.", 1)
        self.assertEqual(incompatible.rerank_candidates("new", baseline), baseline)

    def test_anonymous_agent_bypasses_memory_and_m0_matches_legacy(self) -> None:
        legacy = Agent(self.catalog.path)
        m0 = Agent(self.catalog.path, memory_mode="M0")
        anonymous_session = "anonymous"
        explicit_session = "explicit"
        legacy.reset(anonymous_session, {})
        m0.reset(explicit_session, {}, user_id="user", sequence_index=0)
        message = "I'm looking for shoes. A key requirement is: red."
        legacy_result = legacy.respond(anonymous_session, message, 1, 3)
        explicit_result = m0.respond(explicit_session, message, 1, 3)
        self.assertIsNone(legacy._memory_system)
        self.assertEqual(
            legacy_result["recommendations"],
            explicit_result["recommendations"],
        )

    def test_agent_modes_share_baseline_candidates_and_m1_only_changes_rerank(self) -> None:
        m0 = Agent(self.catalog.path, memory_config=MemoryConfig(
            memory_enabled=False, lambda_memory=1.0
        ))
        m1 = Agent(self.catalog.path, memory_config=MemoryConfig(
            memory_enabled=True, lambda_memory=1.0
        ))
        provider0 = KeywordEmbeddingProvider()
        provider1 = KeywordEmbeddingProvider()
        m0._memory_system = MemorySystem(
            self.catalog.path, self.ids, PRODUCTS, config=m0._memory_config,
            embedding_provider=provider0, catalog_embeddings=catalog_embeddings(provider0),
        )
        m1._memory_system = MemorySystem(
            self.catalog.path, self.ids, PRODUCTS, config=m1._memory_config,
            embedding_provider=provider1, catalog_embeddings=catalog_embeddings(provider1),
        )
        for agent in (m0, m1):
            agent.reset("history", {}, user_id="user", sequence_index=0)
            agent.respond("history", "I'm looking for shoes. red hiking", 1, 4)
            agent.end_session("history", "ignored")
            agent.reset("current", {}, user_id="user", sequence_index=1)
            agent.respond("current", "I'm looking for shoes, but I'm still exploring.", 1, 4)

        trace0 = m0.get_debug_trace("current", 1)
        trace1 = m1.get_debug_trace("current", 1)
        self.assertEqual(trace0["baseline_ranking"], trace1["baseline_ranking"])
        self.assertEqual(trace0["final_ranking"], trace0["baseline_ranking"])
        self.assertNotEqual(trace1["final_ranking"], trace1["baseline_ranking"])

    def test_fresh_systems_start_empty_and_repeated_runs_are_deterministic(self) -> None:
        def run() -> tuple[str, ...]:
            system = self.system(weight=1.0)
            self.assertEqual(system.store.episodes_for_user("user"), ())
            self.finish(system, "user", "s0", 0, "I'm looking for shoes. red hiking")
            system.begin_session("user", "s1", sequence_index=1)
            system.update_session("s1", "I'm looking for items, but I'm still exploring.", 1)
            return tuple(system.rerank_candidates("s1", self.ids))

        self.assertEqual(run(), run())

    def test_agent_rejects_ambiguous_or_legacy_configuration(self) -> None:
        with self.assertRaises(ValueError):
            Agent(self.catalog.path, memory_mode="M1", memory_config=MemoryConfig())
        with self.assertRaises(ValueError):
            Agent(self.catalog.path, memory_mode="A7")


if __name__ == "__main__":
    unittest.main()
