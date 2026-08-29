"""Small typed data model shared by Fast and Slow Memory."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping


class ConstraintKind(str, Enum):
    CATEGORY = "category"
    BUDGET = "budget"
    MATERIAL = "material"
    COLOR = "color"
    SIZE = "size"
    STYLE = "style"
    BRAND = "brand"
    USE_CASE = "use_case"
    FEATURE = "feature"


@dataclass(frozen=True)
class TypedConstraint:
    value: str
    kind: ConstraintKind = ConstraintKind.FEATURE
    hard: bool = False
    negated: bool = False
    explicit: bool = True
    strength: float = 1.0
    confidence: float = 1.0
    source_turn: int = 0
    source: str = "user"
    intent_epoch: int = 0


@dataclass
class FastMemoryState:
    """Observable, structured state for one active conversation."""

    session_id: str
    user_id: str
    sequence_index: int
    category: str = ""
    category_source_turn: int = 0
    intent: str = "buying"
    intent_source_turn: int = 0
    hard_constraints: list[TypedConstraint] = field(default_factory=list)
    soft_preferences: list[TypedConstraint] = field(default_factory=list)
    negatives: list[TypedConstraint] = field(default_factory=list)
    history: list[str] = field(default_factory=list)
    source_turn: int = 0
    intent_epoch: int = 0
    topic_override: bool = False
    intent_locked: bool = False
    confidence: float = 0.0
    override_seed: str | None = None

    @property
    def active_constraints(self) -> tuple[TypedConstraint, ...]:
        return tuple(sorted(
            (*self.hard_constraints, *self.soft_preferences),
            key=lambda item: item.source_turn,
        ))

    @property
    def constraint_values(self) -> tuple[str, ...]:
        return tuple(item.value for item in self.active_constraints)

    @property
    def phrases(self) -> tuple[str, ...]:
        return (self.category or "clothing item", *self.constraint_values)

    @property
    def query_text(self) -> str:
        return " ".join(self.phrases).strip()

    @property
    def constraints_by_kind(self) -> Mapping[ConstraintKind, tuple[TypedConstraint, ...]]:
        return {
            kind: tuple(
                item
                for item in (*self.active_constraints, *self.negatives)
                if item.kind == kind
            )
            for kind in ConstraintKind
        }

    def facts(self, kind: ConstraintKind) -> tuple[TypedConstraint, ...]:
        """Return positive and negative facts in a typed slot."""

        return self.constraints_by_kind[kind]


@dataclass(frozen=True)
class SlowMemoryEpisode:
    user_id: str
    session_id: str
    sequence_index: int
    summary_text: str
    embedding: tuple[float, ...]
    embedding_space_id: str


@dataclass(frozen=True)
class MemoryDebugTrace:
    session_id: str
    turn: int
    memory_applied: bool
    reason: str
    visible_episode_ids: tuple[str, ...]
    baseline_ranking: tuple[str, ...]
    final_ranking: tuple[str, ...]
    embedding_space_id: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "session_id": self.session_id,
            "turn": self.turn,
            "memory_applied": self.memory_applied,
            "reason": self.reason,
            "visible_episode_ids": self.visible_episode_ids,
            "baseline_ranking": self.baseline_ranking,
            "final_ranking": self.final_ranking,
            "embedding_space_id": self.embedding_space_id,
        }


__all__ = [
    "ConstraintKind",
    "FastMemoryState",
    "MemoryDebugTrace",
    "SlowMemoryEpisode",
    "TypedConstraint",
]
