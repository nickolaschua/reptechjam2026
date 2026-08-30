"""Deterministic process-lifetime storage for Phase-5 shadow memories."""

from __future__ import annotations

from dataclasses import dataclass
from threading import RLock
from datetime import datetime
from typing import Any, Iterable, Mapping

import numpy as np

from nickolas.memory.qlmp import MemoryItem, MemoryPolarity, MemorySource


def _nonempty(value: object, name: str) -> str:
    text = str(value)
    if not text.strip():
        raise ValueError(f"{name} must be non-empty")
    return text


def _sequence(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("sequence_index must be a non-negative integer")
    return value


@dataclass(frozen=True)
class StoredMemory:
    """A QLMP item plus the identity and chronology QLMP does not model."""

    user_id: str
    session_id: str
    sequence_index: int
    embedding_space_id: str
    item: MemoryItem


@dataclass(frozen=True)
class SessionCommit:
    """A committed session, retained even when it produced no memory facts."""

    user_id: str
    session_id: str
    sequence_index: int
    embedding_space_id: str
    memory_count: int


@dataclass(frozen=True)
class MemoryStoreSnapshot:
    """Portable, immutable store state used by longitudinal probe replay.

    Normal evaluator logs use vector-free descriptions.  This object retains
    the real embeddings so a replay does not call an embedding provider or
    silently change embedding spaces.
    """

    records: tuple[StoredMemory, ...]
    commits: tuple[SessionCommit, ...]

    def filtered(
        self,
        *,
        user_id: str | None = None,
        sequence_indices: Iterable[int] | None = None,
        before_sequence_index: int | None = None,
    ) -> "MemoryStoreSnapshot":
        user = None if user_id is None else _nonempty(user_id, "user_id")
        allowed = (
            None
            if sequence_indices is None
            else frozenset(_sequence(value) for value in sequence_indices)
        )
        before = (
            None
            if before_sequence_index is None
            else _sequence(before_sequence_index)
        )

        def keep(value: StoredMemory | SessionCommit) -> bool:
            return (
                (user is None or value.user_id == user)
                and (allowed is None or value.sequence_index in allowed)
                and (before is None or value.sequence_index < before)
            )

        return MemoryStoreSnapshot(
            records=tuple(record for record in self.records if keep(record)),
            commits=tuple(commit for commit in self.commits if keep(commit)),
        )

    def to_payload(self, *, include_embeddings: bool = True) -> dict[str, Any]:
        """Return a JSON-compatible snapshot; vectors are explicit opt-in data."""

        commits = [
            {
                "user_id": commit.user_id,
                "session_id": commit.session_id,
                "sequence_index": commit.sequence_index,
                "embedding_space_id": commit.embedding_space_id,
                "memory_count": commit.memory_count,
            }
            for commit in self.commits
        ]
        records: list[dict[str, Any]] = []
        for record in self.records:
            item = record.item
            rendered: dict[str, Any] = {
                "user_id": record.user_id,
                "session_id": record.session_id,
                "sequence_index": record.sequence_index,
                "embedding_space_id": record.embedding_space_id,
                "item": {
                    "id": item.id,
                    "text": item.text,
                    "source": item.source.value,
                    "polarity": item.polarity.value,
                    "scope": item.scope,
                    "timestamp": (
                        None if item.timestamp is None else item.timestamp.isoformat()
                    ),
                    "confidence": item.confidence,
                },
            }
            if include_embeddings:
                rendered["item"]["embedding"] = item.embedding.tolist()
            records.append(rendered)
        return {"version": 1, "records": records, "commits": commits}

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "MemoryStoreSnapshot":
        if payload.get("version") != 1:
            raise ValueError("unsupported memory snapshot version")
        commits = tuple(SessionCommit(**dict(value)) for value in payload.get("commits", ()))
        records: list[StoredMemory] = []
        for raw in payload.get("records", ()):
            value = dict(raw)
            item_value = dict(value.pop("item"))
            if "embedding" not in item_value:
                raise ValueError("replay snapshots must include embeddings")
            timestamp = item_value.get("timestamp")
            item = MemoryItem(
                id=item_value["id"],
                text=item_value["text"],
                embedding=np.asarray(item_value["embedding"], dtype=np.float64),
                source=MemorySource(item_value["source"]),
                polarity=MemoryPolarity(item_value["polarity"]),
                scope=item_value.get("scope"),
                timestamp=None if timestamp is None else datetime.fromisoformat(timestamp),
                confidence=item_value.get("confidence", 1.0),
            )
            records.append(StoredMemory(item=item, **value))
        return cls(records=tuple(records), commits=commits)


class InMemoryUserMemoryStore:
    """Small per-instance store with strict identity and chronology guards."""

    def __init__(self) -> None:
        self._records: dict[str, tuple[StoredMemory, ...]] = {}
        self._commits: dict[str, tuple[SessionCommit, ...]] = {}
        self._session_ids: dict[str, tuple[str, int]] = {}
        self._memory_ids: set[str] = set()
        self._lock = RLock()

    def validate_new_session(
        self,
        user_id: str,
        session_id: str,
        sequence_index: int,
    ) -> None:
        user = _nonempty(user_id, "user_id")
        session = _nonempty(session_id, "session_id")
        index = _sequence(sequence_index)
        with self._lock:
            if session in self._session_ids:
                raise ValueError(f"duplicate committed session_id {session!r}")
            prior = self._commits.get(user, ())
            if prior and index <= prior[-1].sequence_index:
                raise ValueError(
                    f"out-of-order sequence_index {index} for {user!r}; "
                    f"expected greater than {prior[-1].sequence_index}"
                )

    def add_memories(
        self,
        *,
        user_id: str,
        session_id: str,
        sequence_index: int,
        embedding_space_id: str,
        memories: Iterable[MemoryItem],
    ) -> tuple[StoredMemory, ...]:
        user = _nonempty(user_id, "user_id")
        session = _nonempty(session_id, "session_id")
        space = _nonempty(embedding_space_id, "embedding_space_id")
        index = _sequence(sequence_index)
        items = tuple(memories)
        if any(not isinstance(item, MemoryItem) for item in items):
            raise ValueError("memories must contain only QLMP MemoryItem values")
        ids = [item.id for item in items]
        if len(ids) != len(set(ids)):
            raise ValueError("memory IDs must be unique within a commit")

        with self._lock:
            self.validate_new_session(user, session, index)
            duplicates = self._memory_ids.intersection(ids)
            if duplicates:
                raise ValueError(f"duplicate memory ID {sorted(duplicates)[0]!r}")
            records = tuple(
                StoredMemory(user, session, index, space, item) for item in items
            )
            self._records[user] = (*self._records.get(user, ()), *records)
            commit = SessionCommit(user, session, index, space, len(records))
            self._commits[user] = (*self._commits.get(user, ()), commit)
            self._session_ids[session] = (user, index)
            self._memory_ids.update(ids)
            return records

    def get_records(
        self,
        user_id: str,
        before_sequence_index: int | None = None,
    ) -> tuple[StoredMemory, ...]:
        user = _nonempty(user_id, "user_id")
        before = None if before_sequence_index is None else _sequence(before_sequence_index)
        with self._lock:
            records = self._records.get(user, ())
            if before is not None:
                records = tuple(
                    record for record in records if record.sequence_index < before
                )
            return tuple(records)

    def get_memories(
        self,
        user_id: str,
        before_sequence_index: int | None = None,
    ) -> tuple[MemoryItem, ...]:
        return tuple(
            record.item
            for record in self.get_records(user_id, before_sequence_index)
        )

    def commits_for_user(self, user_id: str) -> tuple[SessionCommit, ...]:
        user = _nonempty(user_id, "user_id")
        with self._lock:
            return tuple(self._commits.get(user, ()))

    def snapshot(self, user_id: str | None = None) -> tuple[StoredMemory, ...]:
        with self._lock:
            if user_id is not None:
                return self.get_records(user_id)
            records = [record for values in self._records.values() for record in values]
            return tuple(
                sorted(
                    records,
                    key=lambda record: (
                        record.user_id,
                        record.sequence_index,
                        record.session_id,
                        record.item.id,
                    ),
                )
            )

    def export_snapshot(self, user_id: str | None = None) -> MemoryStoreSnapshot:
        """Capture records and zero-memory commits for safe counterfactual replay."""

        with self._lock:
            if user_id is None:
                users = sorted(self._commits)
            else:
                users = [_nonempty(user_id, "user_id")]
            commits = tuple(
                commit
                for user in users
                for commit in self._commits.get(user, ())
            )
            records = tuple(
                record
                for user in users
                for record in self._records.get(user, ())
            )
            return MemoryStoreSnapshot(records=records, commits=commits)

    def import_snapshot(
        self,
        snapshot: MemoryStoreSnapshot,
        *,
        expected_embedding_space_id: str | None = None,
    ) -> None:
        """Replay a snapshot into an empty store using normal chronology guards."""

        if not isinstance(snapshot, MemoryStoreSnapshot):
            raise ValueError("snapshot must be a MemoryStoreSnapshot")
        with self._lock:
            if self._records or self._commits or self._session_ids or self._memory_ids:
                raise ValueError("memory snapshots may only be imported into an empty store")
            records_by_session: dict[str, list[StoredMemory]] = {}
            for record in snapshot.records:
                records_by_session.setdefault(record.session_id, []).append(record)
            ordered_commits = sorted(
                snapshot.commits,
                key=lambda value: (value.user_id, value.sequence_index, value.session_id),
            )
            known_sessions = {commit.session_id for commit in ordered_commits}
            unknown = set(records_by_session).difference(known_sessions)
            if unknown:
                raise ValueError(f"snapshot records lack a commit for {sorted(unknown)[0]!r}")
            for commit in ordered_commits:
                records = records_by_session.get(commit.session_id, [])
                if len(records) != commit.memory_count:
                    raise ValueError(
                        f"snapshot memory count mismatch for {commit.session_id!r}"
                    )
                if expected_embedding_space_id is not None and (
                    commit.embedding_space_id != expected_embedding_space_id
                ):
                    raise ValueError("snapshot belongs to a different embedding space")
                if any(
                    record.user_id != commit.user_id
                    or record.sequence_index != commit.sequence_index
                    or record.embedding_space_id != commit.embedding_space_id
                    for record in records
                ):
                    raise ValueError("snapshot record envelope does not match its commit")
                self.add_memories(
                    user_id=commit.user_id,
                    session_id=commit.session_id,
                    sequence_index=commit.sequence_index,
                    embedding_space_id=commit.embedding_space_id,
                    memories=[record.item for record in records],
                )

    def clear(self) -> None:
        with self._lock:
            self._records.clear()
            self._commits.clear()
            self._session_ids.clear()
            self._memory_ids.clear()


__all__ = [
    "InMemoryUserMemoryStore",
    "MemoryStoreSnapshot",
    "SessionCommit",
    "StoredMemory",
]
