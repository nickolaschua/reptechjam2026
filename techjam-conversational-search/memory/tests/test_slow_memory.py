from __future__ import annotations

import unittest

import numpy as np

from memory import (
    ConstraintKind,
    FastMemoryState,
    SlowMemoryEpisode,
    TypedConstraint,
    aggregate_slow_vector,
    distill_summary,
    rerank_with_slow_memory,
)


def episode(
    session_id: str,
    sequence_index: int,
    vector: tuple[float, ...],
    *,
    user_id: str = "user",
    space: str = "space",
) -> SlowMemoryEpisode:
    return SlowMemoryEpisode(
        user_id=user_id,
        session_id=session_id,
        sequence_index=sequence_index,
        summary_text=session_id,
        embedding=vector,
        embedding_space_id=space,
    )


class SlowMemoryTests(unittest.TestCase):
    def test_summary_has_fixed_group_order_and_typed_facts(self) -> None:
        state = FastMemoryState("s", "u", 0, category="shoes", intent="buying")
        state.hard_constraints.append(TypedConstraint(
            "under $60", kind=ConstraintKind.BUDGET, hard=True
        ))
        state.soft_preferences.append(TypedConstraint("red", kind=ConstraintKind.COLOR))
        state.negatives.append(TypedConstraint(
            "leather", kind=ConstraintKind.MATERIAL, hard=True, negated=True
        ))
        self.assertEqual(
            distill_summary(state),
            "category=shoes; intent=buying; hard facts: budget=under $60; "
            "soft facts: color=red; negatives: material=leather",
        )

    def test_no_history_returns_none(self) -> None:
        self.assertIsNone(aggregate_slow_vector(
            (), user_id="user", current_sequence_index=1,
            embedding_space_id="space", tau=6.0,
        ))

    def test_one_episode_is_normalized(self) -> None:
        vector = aggregate_slow_vector(
            (episode("s0", 0, (3.0, 4.0)),),
            user_id="user",
            current_sequence_index=1,
            embedding_space_id="space",
            tau=6.0,
        )
        np.testing.assert_allclose(vector, np.asarray([0.6, 0.8]), atol=1e-6)

    def test_multiple_episodes_use_exponential_age_and_newer_has_more_influence(self) -> None:
        vector = aggregate_slow_vector(
            (episode("old", 0, (1.0, 0.0)), episode("new", 1, (0.0, 1.0))),
            user_id="user",
            current_sequence_index=2,
            embedding_space_id="space",
            tau=1.0,
        )
        expected = np.asarray([np.exp(-2.0), np.exp(-1.0)])
        expected /= np.linalg.norm(expected)
        np.testing.assert_allclose(vector, expected, atol=1e-6)
        self.assertGreater(float(vector[1]), float(vector[0]))

    def test_user_sequence_and_space_filters(self) -> None:
        episodes = (
            episode("same", 0, (1.0, 0.0)),
            episode("other-user", 0, (0.0, 1.0), user_id="other"),
            episode("future", 2, (0.0, 1.0)),
            episode("other-space", 1, (0.0, 1.0), space="different"),
        )
        vector = aggregate_slow_vector(
            episodes,
            user_id="user",
            current_sequence_index=2,
            embedding_space_id="space",
            tau=6.0,
        )
        np.testing.assert_array_equal(vector, np.asarray([1.0, 0.0], dtype=np.float32))

    def test_rerank_uses_one_based_baseline_rank_and_deterministic_ties(self) -> None:
        ranked = rerank_with_slow_memory(
            ["b", "a", "c"],
            np.asarray([[0.0, 1.0], [0.0, 1.0], [1.0, 0.0]], dtype=np.float32),
            np.asarray([1.0, 0.0], dtype=np.float32),
            lambda_memory=1.0,
        )
        self.assertEqual(ranked, ["c", "b", "a"])


if __name__ == "__main__":
    unittest.main()
