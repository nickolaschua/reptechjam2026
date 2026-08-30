"""Lifecycle integration for the Fast/Slow memory baseline."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .config import MemoryConfig
from .embeddings import EmbeddingProvider, EmbeddingService
from .fast_memory import SemanticParser, override_intent as override_fast_intent
from .fast_memory import update_state
from .slow_memory import aggregate_slow_vector, distill_summary, rerank_with_slow_memory
from .store import InMemoryMemoryStore
from .types import FastMemoryState, MemoryDebugTrace, SlowMemoryEpisode


class MemorySystem:
    """Own active Fast Memory and completed per-user Slow Memory episodes."""

    def __init__(
        self,
        catalog_path: str | Path,
        ids: Sequence[str],
        products: Sequence[Mapping[str, Any]] | None = None,
        *,
        config: MemoryConfig | None = None,
        embedding_provider: EmbeddingProvider | None = None,
        catalog_embeddings: np.ndarray | None = None,
        store: InMemoryMemoryStore | None = None,
        semantic_parser: SemanticParser | None = None,
    ) -> None:
        # ``products`` remains accepted for constructor compatibility but is
        # deliberately unused: product metadata creates no second route.
        del products
        self.config = config or MemoryConfig()
        self.store = store or InMemoryMemoryStore()
        self.embeddings = EmbeddingService(
            self.config,
            catalog_path,
            ids,
            provider=embedding_provider,
            catalog_embeddings=catalog_embeddings,
        )
        self._semantic_parser = semantic_parser
        self._states: dict[str, FastMemoryState] = {}
        self._traces: dict[str, dict[int, MemoryDebugTrace]] = {}

    def begin_session(
        self,
        user_id: str,
        session_id: str,
        user_profile: Mapping[str, Any] | None = None,
        sequence_index: int | None = None,
    ) -> FastMemoryState:
        """Start Fast Memory and freeze Slow Memory visibility.

        ``user_profile`` is accepted at the orchestration boundary but is not
        read by this baseline.
        """

        del user_profile
        record = self.store.begin_session(user_id, session_id, sequence_index)
        state = FastMemoryState(
            session_id=record.session_id,
            user_id=record.user_id,
            sequence_index=record.sequence_index,
        )
        self._states[record.session_id] = state
        self._traces[record.session_id] = {}
        return state

    def update_session(
        self,
        session_id: str,
        user_message: str,
        turn: int,
        semantic_parser: SemanticParser | None = None,
    ) -> FastMemoryState:
        self.store.active_session(session_id)
        try:
            state = self._states[str(session_id)]
        except KeyError as exc:
            raise RuntimeError(f"unknown active session {session_id!r}") from exc
        parser = semantic_parser if semantic_parser is not None else self._semantic_parser
        return update_state(state, user_message, turn, parser)

    def override_intent(
        self,
        session_id: str,
        category: str,
        turn: int,
        *,
        intent: str | None = None,
    ) -> FastMemoryState:
        self.store.active_session(session_id)
        try:
            state = self._states[str(session_id)]
        except KeyError as exc:
            raise RuntimeError(f"unknown active session {session_id!r}") from exc
        return override_fast_intent(state, category, turn, intent=intent)

    def rerank_candidates(
        self,
        session_id: str,
        candidate_ids: Sequence[str],
    ) -> list[str]:
        """Optionally apply the one and only Slow Memory rerank."""

        record = self.store.active_session(session_id)
        try:
            state = self._states[str(session_id)]
        except KeyError as exc:
            raise RuntimeError(f"unknown active session {session_id!r}") from exc
        baseline = [str(value) for value in candidate_ids]
        final = list(baseline)
        visible_ids = tuple(episode.session_id for episode in record.visible_episodes)
        catalog = self.embeddings.catalog
        applied = False

        if not baseline:
            reason = "empty_candidates"
        elif not self.config.memory_enabled:
            reason = "memory_disabled"
        elif catalog is None:
            reason = "product_embeddings_unavailable"
        else:
            slow_vector = aggregate_slow_vector(
                record.visible_episodes,
                user_id=record.user_id,
                current_sequence_index=record.sequence_index,
                embedding_space_id=catalog.space_id,
                tau=self.config.tau,
            )
            if slow_vector is None:
                reason = "no_compatible_history"
            elif slow_vector.shape[0] != catalog.dimension:
                reason = "incompatible_product_embeddings"
            else:
                candidate_vectors = catalog.vectors_for(baseline)
                if candidate_vectors is None:
                    reason = "product_embeddings_unavailable"
                else:
                    final = rerank_with_slow_memory(
                        baseline,
                        candidate_vectors,
                        slow_vector,
                        lambda_memory=self.config.lambda_memory,
                    )
                    reason = "memory_applied"
                    applied = True

        self._traces[str(session_id)][state.source_turn] = MemoryDebugTrace(
            session_id=str(session_id),
            turn=state.source_turn,
            memory_applied=applied,
            reason=reason,
            visible_episode_ids=visible_ids,
            baseline_ranking=tuple(baseline),
            final_ranking=tuple(final),
            embedding_space_id="" if catalog is None else catalog.space_id,
        )
        return final

    def end_session(
        self,
        session_id: str,
        outcome: Any = None,
        purchased_product: Any = None,
        evidence: Any = None,
    ) -> SlowMemoryEpisode:
        """Distill, embed once, and commit; lifecycle outcomes are ignored."""

        del outcome, purchased_product, evidence
        record = self.store.active_session(session_id)
        try:
            state = self._states[str(session_id)]
        except KeyError as exc:
            raise RuntimeError(f"unknown active session {session_id!r}") from exc
        summary = distill_summary(state)
        embedding = self.embeddings.embed_once(summary)
        episode = SlowMemoryEpisode(
            user_id=record.user_id,
            session_id=record.session_id,
            sequence_index=record.sequence_index,
            summary_text=summary,
            embedding=tuple(float(value) for value in embedding),
            embedding_space_id=self.embeddings.space_id,
        )
        # Cleanup happens only after the store has accepted the episode.
        self.store.commit(session_id, episode)
        del self._states[str(session_id)]
        return episode

    def get_debug_trace(
        self,
        session_id: str,
        turn: int | None = None,
    ) -> dict[str, object] | dict[str, dict[str, object]] | None:
        traces = self._traces.get(str(session_id))
        if traces is None:
            return None
        if turn is not None:
            trace = traces.get(int(turn))
            return trace.to_dict() if trace is not None else None
        return {
            str(number): trace.to_dict()
            for number, trace in sorted(traces.items())
        }


__all__ = ["MemorySystem"]
