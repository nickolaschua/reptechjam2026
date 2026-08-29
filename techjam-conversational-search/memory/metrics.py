"""Small rank-comparison helpers for a separate longitudinal evaluator."""

from __future__ import annotations

from typing import Iterable, Sequence


def reciprocal_rank(rank: int | None) -> float:
    return 0.0 if rank is None or rank <= 0 else 1.0 / rank


def reciprocal_rank_uplift(baseline_rank: int | None, memory_rank: int | None) -> float:
    return reciprocal_rank(memory_rank) - reciprocal_rank(baseline_rank)


def rank_uplift(baseline_rank: int | None, memory_rank: int | None, *, missing_rank: int = 11) -> int:
    """Positive means memory improved rank; ``missing_rank`` sets the cutoff+1."""

    baseline = baseline_rank if baseline_rank is not None else missing_rank
    memory = memory_rank if memory_rank is not None else missing_rank
    return int(baseline - memory)


def memory_harm_rate(
    baseline_ranks: Sequence[int | None] | Iterable[int | None],
    memory_ranks: Sequence[int | None] | Iterable[int | None],
    *,
    missing_rank: int = 11,
) -> float:
    baseline = list(baseline_ranks)
    memory = list(memory_ranks)
    if len(baseline) != len(memory):
        raise ValueError("rank sequences must have equal length")
    if not baseline:
        return 0.0
    harmed = sum(
        1 for left, right in zip(baseline, memory)
        if (right if right is not None else missing_rank) > (left if left is not None else missing_rank)
    )
    return harmed / len(baseline)


__all__ = ["memory_harm_rate", "rank_uplift", "reciprocal_rank", "reciprocal_rank_uplift"]
