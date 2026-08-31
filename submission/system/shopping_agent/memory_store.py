"""Versioned user-isolated storage for one normalized vector per user."""
from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
from enum import Enum
import json
import math
import os
from pathlib import Path
from threading import RLock
from typing import Any, Iterable, Mapping
import numpy as np

SNAPSHOT_VERSION = 2
_MIXTURE_EPSILON = 1e-12

def _nonempty(value: object, name: str) -> str:
    text = str(value)
    if not text.strip(): raise ValueError(f"{name} must be non-empty")
    return text

def _sequence(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("sequence_index must be a non-negative integer")
    return value

def _unit_vector(value: object, name: str = "vector") -> np.ndarray:
    vector = np.array(value, dtype=np.float32, copy=True)
    if vector.ndim != 1 or vector.size == 0 or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be a finite, non-empty one-dimensional vector")
    norm = float(np.linalg.norm(vector))
    if not np.isclose(norm, 1.0, rtol=1e-5, atol=1e-6):
        raise ValueError(f"{name} must be L2-normalized; got norm {norm:.8g}")
    vector /= norm
    vector.setflags(write=False)
    return vector


class MemoryUpdateMode(str, Enum):
    """Supported one-vector long-term-memory update strategies."""

    ADAPTIVE = "adaptive"
    FIXED = "fixed"


@dataclass(frozen=True)
class MemoryUpdatePolicy:
    """Configuration boundary for the current centroid update implementation."""

    mode: MemoryUpdateMode = MemoryUpdateMode.ADAPTIVE
    alpha_min: float = 0.0
    alpha_max: float = 0.30
    fixed_alpha: float = 0.30

    def __post_init__(self) -> None:
        try:
            mode = MemoryUpdateMode(self.mode)
        except ValueError as exc:
            raise ValueError("mode must be 'adaptive' or 'fixed'") from exc
        values = (float(self.alpha_min), float(self.alpha_max), float(self.fixed_alpha))
        if not all(math.isfinite(value) for value in values):
            raise ValueError("memory update alpha values must be finite")
        alpha_min, alpha_max, fixed_alpha = values
        if not 0.0 <= alpha_min <= alpha_max <= 1.0:
            raise ValueError("adaptive alpha bounds must satisfy 0 <= alpha_min <= alpha_max <= 1")
        if not 0.0 <= fixed_alpha <= 1.0:
            raise ValueError("fixed_alpha must be between zero and one")
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "alpha_min", alpha_min)
        object.__setattr__(self, "alpha_max", alpha_max)
        object.__setattr__(self, "fixed_alpha", fixed_alpha)

    @classmethod
    def adaptive(cls, *, alpha_min: float = 0.0, alpha_max: float = 0.30) -> "MemoryUpdatePolicy":
        return cls(mode=MemoryUpdateMode.ADAPTIVE, alpha_min=alpha_min, alpha_max=alpha_max)

    @classmethod
    def fixed(cls, alpha: float = 0.30) -> "MemoryUpdatePolicy":
        return cls(mode=MemoryUpdateMode.FIXED, fixed_alpha=alpha)


DEFAULT_MEMORY_UPDATE_POLICY = MemoryUpdatePolicy()


@dataclass(frozen=True)
class MemoryUpdateResult:
    """Vector result and vector-free diagnostics for one evidence observation."""

    vector: np.ndarray
    raw_similarity: float
    bounded_similarity: float
    effective_alpha: float
    fallback_reason: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "vector", _unit_vector(self.vector))


def update_memory_vector(
    old_vector: np.ndarray,
    new_vector: np.ndarray,
    policy: MemoryUpdatePolicy = DEFAULT_MEMORY_UPDATE_POLICY,
) -> MemoryUpdateResult:
    """Pure normalized centroid update with bounded novelty adaptation."""

    if not isinstance(policy, MemoryUpdatePolicy):
        raise ValueError("update_policy must be a MemoryUpdatePolicy")
    old = _unit_vector(old_vector, "old_vector")
    new = _unit_vector(new_vector, "new_vector")
    if old.shape != new.shape:
        raise ValueError("old_vector and new_vector dimensions must match")
    raw_similarity = float(old @ new)
    bounded_similarity = float(np.clip(raw_similarity, 0.0, 1.0))
    if policy.mode is MemoryUpdateMode.ADAPTIVE:
        effective_alpha = policy.alpha_min + (
            policy.alpha_max - policy.alpha_min
        ) * (1.0 - bounded_similarity)
    else:
        effective_alpha = policy.fixed_alpha
    mixture = (1.0 - effective_alpha) * old + effective_alpha * new
    norm = float(np.linalg.norm(mixture))
    fallback_reason = None
    if not math.isfinite(norm) or norm <= _MIXTURE_EPSILON:
        combined = old
        fallback_reason = "near_zero_mixture"
    else:
        combined = mixture / norm
    return MemoryUpdateResult(
        vector=combined,
        raw_similarity=raw_similarity,
        bounded_similarity=bounded_similarity,
        effective_alpha=float(effective_alpha),
        fallback_reason=fallback_reason,
    )

@dataclass(frozen=True)
class LongTermMemoryState:
    user_id: str
    vector: np.ndarray
    embedding_space_id: str
    last_committed_sequence: int
    update_count: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "user_id", _nonempty(self.user_id, "user_id"))
        object.__setattr__(self, "embedding_space_id", _nonempty(self.embedding_space_id, "embedding_space_id"))
        object.__setattr__(self, "last_committed_sequence", _sequence(self.last_committed_sequence))
        if isinstance(self.update_count, bool) or not isinstance(self.update_count, int) or self.update_count < 1:
            raise ValueError("update_count must be a positive integer")
        object.__setattr__(self, "vector", _unit_vector(self.vector))

@dataclass(frozen=True)
class SessionCommit:
    user_id: str
    session_id: str
    sequence_index: int
    embedding_space_id: str
    vector_changed: bool

@dataclass(frozen=True)
class LongTermMemoryCommit:
    vector_changed: bool
    state: LongTermMemoryState | None
    update_mode: MemoryUpdateMode
    raw_update_similarity: float | None
    bounded_update_similarity: float | None
    effective_alpha: float | None
    update_fallback_reason: str | None

@dataclass(frozen=True)
class MemoryStoreSnapshot:
    states: tuple[LongTermMemoryState, ...]
    commits: tuple[SessionCommit, ...]

    def filtered(self, *, user_id: str | None = None, sequence_indices: Iterable[int] | None = None,
                 before_sequence_index: int | None = None) -> "MemoryStoreSnapshot":
        user = None if user_id is None else _nonempty(user_id, "user_id")
        allowed = None if sequence_indices is None else frozenset(_sequence(v) for v in sequence_indices)
        before = None if before_sequence_index is None else _sequence(before_sequence_index)
        commits = tuple(c for c in self.commits if (user is None or c.user_id == user)
                        and (allowed is None or c.sequence_index in allowed)
                        and (before is None or c.sequence_index < before))
        allowed_commits = {(c.user_id, c.sequence_index) for c in commits}
        return MemoryStoreSnapshot(
            tuple(s for s in self.states if (s.user_id, s.last_committed_sequence) in allowed_commits),
            commits,
        )

    def to_payload(self, *, include_embeddings: bool = True) -> dict[str, Any]:
        states = []
        for state in self.states:
            item: dict[str, Any] = {"user_id": state.user_id, "embedding_space_id": state.embedding_space_id,
                                    "last_committed_sequence": state.last_committed_sequence,
                                    "update_count": state.update_count}
            if include_embeddings: item["vector"] = state.vector.tolist()
            states.append(item)
        return {"version": SNAPSHOT_VERSION, "kind": "gated-vector-memory", "states": states,
                "commits": [c.__dict__ for c in self.commits]}

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "MemoryStoreSnapshot":
        version = payload.get("version")
        if version != SNAPSHOT_VERSION or payload.get("kind") != "gated-vector-memory":
            if version == 1:
                raise ValueError("QLMP snapshot version 1 is not supported; migration is intentionally unavailable")
            raise ValueError("unsupported gated-vector-memory snapshot version")
        states = []
        for raw in payload.get("states", ()):
            value = dict(raw)
            if "vector" not in value: raise ValueError("replay snapshots must include vectors")
            states.append(LongTermMemoryState(**value))
        commits = tuple(SessionCommit(**dict(raw)) for raw in payload.get("commits", ()))
        return cls(tuple(states), commits)

class InMemoryVectorMemoryStore:
    """User-isolated vector states with strict monotonic session chronology."""
    def __init__(self) -> None:
        self._states: dict[str, LongTermMemoryState] = {}
        self._state_history: dict[str, tuple[LongTermMemoryState, ...]] = {}
        self._commits: dict[str, tuple[SessionCommit, ...]] = {}
        self._session_ids: set[str] = set()
        self._active_users: dict[str, str] = {}
        self._active_sessions: dict[str, str] = {}
        self._lock = RLock()

    def validate_new_session(self, user_id: str, session_id: str, sequence_index: int) -> None:
        user, session, index = _nonempty(user_id, "user_id"), _nonempty(session_id, "session_id"), _sequence(sequence_index)
        with self._lock:
            if session in self._session_ids: raise ValueError(f"duplicate committed session_id {session!r}")
            prior = self._commits.get(user, ())
            if prior and index <= prior[-1].sequence_index:
                raise ValueError(f"out-of-order sequence_index {index} for {user!r}; expected greater than {prior[-1].sequence_index}")

    def begin_session(self, user_id: str, session_id: str, sequence_index: int) -> None:
        user, session = _nonempty(user_id, "user_id"), _nonempty(session_id, "session_id")
        with self._lock:
            self.validate_new_session(user, session, sequence_index)
            if user in self._active_users:
                raise ValueError(f"user {user!r} already has an active session")
            if session in self._active_sessions:
                raise ValueError(f"session {session!r} is already active")
            self._active_users[user] = session
            self._active_sessions[session] = user

    def get_state(self, user_id: str, *, before_sequence_index: int | None = None) -> LongTermMemoryState | None:
        user = _nonempty(user_id, "user_id")
        before = None if before_sequence_index is None else _sequence(before_sequence_index)
        with self._lock:
            state = self._states.get(user)
            if state is not None and before is not None and state.last_committed_sequence >= before: return None
            return state

    def commit(self, *, user_id: str, session_id: str, sequence_index: int, embedding_space_id: str,
               new_preferences: np.ndarray | None,
               update_policy: MemoryUpdatePolicy = DEFAULT_MEMORY_UPDATE_POLICY) -> LongTermMemoryCommit:
        user, session = _nonempty(user_id, "user_id"), _nonempty(session_id, "session_id")
        space, index = _nonempty(embedding_space_id, "embedding_space_id"), _sequence(sequence_index)
        if not isinstance(update_policy, MemoryUpdatePolicy):
            raise ValueError("update_policy must be a MemoryUpdatePolicy")
        new_vector = None if new_preferences is None else _unit_vector(new_preferences, "new_preferences")
        with self._lock:
            self.validate_new_session(user, session, index)
            active_session = self._active_users.get(user)
            if active_session is not None and active_session != session:
                raise ValueError(f"user {user!r} already has an active session")
            active_user = self._active_sessions.get(session)
            if active_user is not None and active_user != user:
                raise ValueError(f"session {session!r} belongs to a different active user")
            prior = self._states.get(user)
            if prior is not None and prior.embedding_space_id != space:
                raise ValueError("stored memory belongs to a different embedding space")
            state, changed = prior, False
            update_result = None
            if new_vector is not None:
                if prior is None:
                    combined, count, changed = new_vector, 1, True
                else:
                    update_result = update_memory_vector(prior.vector, new_vector, update_policy)
                    combined, count = update_result.vector, prior.update_count+1
                    changed = not np.allclose(combined, prior.vector, rtol=1e-6, atol=1e-7)
                state = LongTermMemoryState(user, combined, space, index, count)
                self._states[user] = state
            elif prior is not None:
                state = LongTermMemoryState(user, prior.vector, space, index, prior.update_count)
                self._states[user] = state
            if state is not None:
                self._state_history[user] = (*self._state_history.get(user, ()), state)
            commit = SessionCommit(user, session, index, space, changed)
            self._commits[user] = (*self._commits.get(user, ()), commit)
            self._session_ids.add(session)
            self._active_users.pop(user, None)
            self._active_sessions.pop(session, None)
            return LongTermMemoryCommit(
                vector_changed=changed,
                state=state,
                update_mode=update_policy.mode,
                raw_update_similarity=None if update_result is None else update_result.raw_similarity,
                bounded_update_similarity=None if update_result is None else update_result.bounded_similarity,
                effective_alpha=None if update_result is None else update_result.effective_alpha,
                update_fallback_reason=None if update_result is None else update_result.fallback_reason,
            )

    def cancel_session(self, session_id: str) -> None:
        """Release an active session without recording a longitudinal commit."""

        session = _nonempty(session_id, "session_id")
        with self._lock:
            user = self._active_sessions.pop(session, None)
            if user is not None and self._active_users.get(user) == session:
                self._active_users.pop(user, None)

    def commits_for_user(self, user_id: str) -> tuple[SessionCommit, ...]:
        with self._lock: return tuple(self._commits.get(_nonempty(user_id, "user_id"), ()))

    def next_sequence_index(self, user_id: str) -> int:
        """Return the next valid longitudinal sequence for ``user_id``."""

        commits = self.commits_for_user(user_id)
        return 0 if not commits else commits[-1].sequence_index + 1

    def describe_user(self, user_id: str) -> dict[str, Any]:
        """Return vector-free state suitable for a demo/debug display."""

        user = _nonempty(user_id, "user_id")
        with self._lock:
            state = self._states.get(user)
            commits = self._commits.get(user, ())
            return {
                "user_id": user,
                "exists": state is not None,
                "version": SNAPSHOT_VERSION,
                "kind": "gated-vector-memory",
                "embedding_space_id": None if state is None else state.embedding_space_id,
                "last_committed_sequence": None if state is None else state.last_committed_sequence,
                "update_count": 0 if state is None else state.update_count,
                "session_commit_count": len(commits),
            }

    def export_snapshot(self, user_id: str | None = None) -> MemoryStoreSnapshot:
        with self._lock:
            users = sorted(self._commits) if user_id is None else [_nonempty(user_id, "user_id")]
            return MemoryStoreSnapshot(tuple(s for u in users for s in self._state_history.get(u, ())),
                                       tuple(c for u in users for c in self._commits.get(u, ())))

    def import_snapshot(self, snapshot: MemoryStoreSnapshot, *, expected_embedding_space_id: str | None = None) -> None:
        if not isinstance(snapshot, MemoryStoreSnapshot): raise ValueError("snapshot must be a MemoryStoreSnapshot")
        with self._lock:
            if self._states or self._state_history or self._commits or self._session_ids or self._active_users: raise ValueError("memory snapshots may only be imported into an empty store")
            state_history: dict[str, list[LongTermMemoryState]] = {}
            for state in snapshot.states:
                state_history.setdefault(state.user_id, []).append(state)
                if expected_embedding_space_id is not None and state.embedding_space_id != expected_embedding_space_id:
                    raise ValueError("snapshot belongs to a different embedding space")
            new_commits: dict[str, tuple[SessionCommit, ...]] = {}
            new_session_ids: set[str] = set()
            for commit in sorted(snapshot.commits, key=lambda c: (c.user_id, c.sequence_index, c.session_id)):
                if commit.session_id in new_session_ids: raise ValueError(f"duplicate committed session_id {commit.session_id!r}")
                prior = new_commits.get(commit.user_id, ())
                if prior and commit.sequence_index <= prior[-1].sequence_index: raise ValueError("snapshot contains non-monotonic chronology")
                new_commits[commit.user_id] = (*prior, commit)
                new_session_ids.add(commit.session_id)
            new_states: dict[str, LongTermMemoryState] = {}
            new_state_history: dict[str, tuple[LongTermMemoryState, ...]] = {}
            for user, values in state_history.items():
                values.sort(key=lambda value: value.last_committed_sequence)
                if any(left.last_committed_sequence >= right.last_committed_sequence for left, right in zip(values, values[1:])):
                    raise ValueError("snapshot contains non-monotonic state chronology")
                state = values[-1]
                commits = new_commits.get(user, ())
                if not commits or commits[-1].sequence_index != state.last_committed_sequence:
                    raise ValueError("snapshot state chronology does not match its commits")
                new_state_history[user] = tuple(values)
                new_states[user] = state
            self._commits = new_commits
            self._session_ids = new_session_ids
            self._state_history = new_state_history
            self._states = new_states

    def clear(self) -> None:
        with self._lock: self._states.clear(); self._state_history.clear(); self._commits.clear(); self._session_ids.clear(); self._active_users.clear(); self._active_sessions.clear()

    def clear_user(self, user_id: str) -> None:
        """Reset exactly one user without touching any other user's memory."""

        user = _nonempty(user_id, "user_id")
        with self._lock:
            if user in self._active_users:
                raise ValueError(f"cannot reset active user {user!r}")
            removed = self._commits.pop(user, ())
            self._states.pop(user, None)
            self._state_history.pop(user, None)
            self._session_ids.difference_update(commit.session_id for commit in removed)


class JsonFileVectorMemoryStore(InMemoryVectorMemoryStore):
    """The existing store semantics with an atomic JSON snapshot on commit."""

    def __init__(self, path: str | Path) -> None:
        super().__init__()
        self.path = Path(path)
        if self.path.exists():
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            super().import_snapshot(MemoryStoreSnapshot.from_payload(payload))

    def _persist(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f".{self.path.name}.tmp")
        payload = self.export_snapshot().to_payload(include_embeddings=True)
        temporary.write_text(
            json.dumps(payload, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, self.path)

    def _transaction(self, mutation: Any) -> Any:
        with self._lock:
            before = (
                deepcopy(self._states), deepcopy(self._state_history),
                deepcopy(self._commits), set(self._session_ids),
                dict(self._active_users), dict(self._active_sessions),
            )
            try:
                result = mutation()
                self._persist()
                return result
            except Exception:
                (
                    self._states, self._state_history, self._commits,
                    self._session_ids, self._active_users, self._active_sessions,
                ) = before
                raise

    def commit(self, **kwargs: Any) -> LongTermMemoryCommit:
        return self._transaction(lambda: super(JsonFileVectorMemoryStore, self).commit(**kwargs))

    def clear(self) -> None:
        self._transaction(lambda: super(JsonFileVectorMemoryStore, self).clear())

    def clear_user(self, user_id: str) -> None:
        self._transaction(lambda: super(JsonFileVectorMemoryStore, self).clear_user(user_id))

InMemoryUserMemoryStore = InMemoryVectorMemoryStore
__all__ = ["InMemoryUserMemoryStore", "InMemoryVectorMemoryStore", "JsonFileVectorMemoryStore",
           "DEFAULT_MEMORY_UPDATE_POLICY", "LongTermMemoryCommit", "LongTermMemoryState",
           "MemoryStoreSnapshot", "MemoryUpdateMode", "MemoryUpdatePolicy", "MemoryUpdateResult",
           "SessionCommit", "update_memory_vector"]
