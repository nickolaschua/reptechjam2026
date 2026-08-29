"""Process-lifetime Slow Memory storage with chronology and leakage guards."""

from __future__ import annotations

from dataclasses import dataclass
from threading import RLock

from .types import SlowMemoryEpisode


@dataclass(frozen=True)
class ActiveSession:
    user_id: str
    session_id: str
    sequence_index: int
    visible_episodes: tuple[SlowMemoryEpisode, ...]


class InMemoryMemoryStore:
    """A deterministic store partitioned solely by stable ``user_id``."""

    def __init__(self) -> None:
        self._episodes: dict[str, tuple[SlowMemoryEpisode, ...]] = {}
        self._active: dict[str, ActiveSession] = {}
        self._active_user: dict[str, str] = {}
        self._session_owner: dict[str, str] = {}
        self._committed_sessions: set[str] = set()
        self._lock = RLock()

    def begin_session(
        self,
        user_id: str,
        session_id: str,
        sequence_index: int | None = None,
    ) -> ActiveSession:
        if not str(user_id).strip() or not str(session_id).strip():
            raise ValueError("user_id and session_id must be non-empty")
        user_id, session_id = str(user_id), str(session_id)
        with self._lock:
            owner = self._session_owner.get(session_id)
            if owner is not None:
                if owner != user_id:
                    raise ValueError(f"session {session_id!r} already belongs to another user")
                raise ValueError(f"duplicate session_id {session_id!r}")
            if user_id in self._active_user:
                raise ValueError(f"user {user_id!r} already has an uncommitted session")
            prior = self._episodes.get(user_id, ())
            minimum = prior[-1].sequence_index + 1 if prior else 0
            index = minimum if sequence_index is None else int(sequence_index)
            if index < minimum:
                raise ValueError(
                    f"out-of-order sequence_index {index} for {user_id!r}; "
                    f"expected at least {minimum}"
                )
            record = ActiveSession(
                user_id=user_id,
                session_id=session_id,
                sequence_index=index,
                visible_episodes=tuple(
                    episode for episode in prior if episode.sequence_index < index
                ),
            )
            self._active[session_id] = record
            self._active_user[user_id] = session_id
            self._session_owner[session_id] = user_id
            return record

    def active_session(self, session_id: str) -> ActiveSession:
        with self._lock:
            try:
                return self._active[str(session_id)]
            except KeyError as exc:
                if str(session_id) in self._committed_sessions:
                    raise RuntimeError(f"session {session_id!r} has already ended") from exc
                raise RuntimeError(f"unknown active session {session_id!r}") from exc

    def visible_episodes(self, session_id: str) -> tuple[SlowMemoryEpisode, ...]:
        return self.active_session(session_id).visible_episodes

    def commit(self, session_id: str, episode: SlowMemoryEpisode) -> None:
        with self._lock:
            record = self.active_session(session_id)
            if episode.session_id != record.session_id or episode.user_id != record.user_id:
                raise ValueError("episode identity does not match active session")
            if episode.sequence_index != record.sequence_index:
                raise ValueError("episode sequence does not match active session")
            prior = self._episodes.get(record.user_id, ())
            minimum = prior[-1].sequence_index + 1 if prior else 0
            if episode.sequence_index < minimum:
                raise ValueError("episode commit is duplicate or out of order")
            self._episodes[record.user_id] = (*prior, episode)
            del self._active[record.session_id]
            del self._active_user[record.user_id]
            self._committed_sessions.add(record.session_id)

    def episodes_for_user(self, user_id: str) -> tuple[SlowMemoryEpisode, ...]:
        with self._lock:
            return self._episodes.get(str(user_id), ())

    def session_user(self, session_id: str) -> str | None:
        with self._lock:
            return self._session_owner.get(str(session_id))


__all__ = ["ActiveSession", "InMemoryMemoryStore"]
