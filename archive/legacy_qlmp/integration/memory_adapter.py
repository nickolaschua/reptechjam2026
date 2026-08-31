"""Thin Nickolas Fast Memory -> existing QLMP model adapter."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import re
from typing import Iterable, Mapping, Sequence

import numpy as np

from nickolas.memory.qlmp import MemoryItem, MemoryPolarity, MemorySource


_SPACE_RE = re.compile(r"dimensions=(\d+)")
_POSITIVE_MARKERS = {"true", "yes", "affirmative", "required", "included"}
_NEGATIVE_MARKERS = {"false", "no", "none", "n/a", "null", "other", ""}
_NON_FACT_SLOT_NAMES = {
    "accumulated_terms",
    "asked_attributes",
    "debug_info",
    "history",
    "seen_asins",
    "stashed_terms",
}


def _canonical(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip()).casefold()


def _iter_values(value: object) -> list[object]:
    if isinstance(value, (set, frozenset, list, tuple)):
        return sorted(value, key=lambda item: _canonical(item))
    return [value]


def _memory_id(
    *,
    user_id: str,
    session_id: str,
    sequence_index: int,
    fact_type: str,
    value: str,
    polarity: MemoryPolarity,
) -> str:
    payload = json.dumps(
        {
            "fact_type": fact_type,
            "polarity": polarity.value,
            "sequence_index": sequence_index,
            "session_id": session_id,
            "user_id": user_id,
            "value": value,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return "memory_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class MemoryDraft:
    """Deterministic semantic fields before the explicit embedding step."""

    id: str
    text: str
    source: MemorySource
    polarity: MemoryPolarity
    scope: str | None
    confidence: float = 1.0


class FastMemoryQLMPAdapter:
    """Create atomic QLMP records without implementing any QLMP mathematics."""

    def __init__(self, embedding_backend: object, embedding_space_id: str) -> None:
        self.embedding_backend = embedding_backend
        self.embedding_space_id = str(embedding_space_id)
        backend_space = str(getattr(embedding_backend, "embedding_space_id", ""))
        if not self.embedding_space_id or backend_space != self.embedding_space_id:
            raise ValueError("memory and agent embedding spaces must match exactly")

    @staticmethod
    def extract_drafts(
        final_fast_memory: Mapping[str, object],
        *,
        user_id: str,
        session_id: str,
        sequence_index: int,
    ) -> tuple[MemoryDraft, ...]:
        """Extract only active category/slot/negative facts in a fixed order."""

        user = str(user_id)
        session = str(session_id)
        if not user.strip() or not session.strip():
            raise ValueError("user_id and session_id must be non-empty")
        if isinstance(sequence_index, bool) or not isinstance(sequence_index, int):
            raise ValueError("sequence_index must be an integer")

        category = _canonical(final_fast_memory.get("category"))
        # "clothing" is the Nickolas reset default and has no provenance. Omit
        # it unless it appears as an explicit disclosed slot.
        slots = final_fast_memory.get("disclosed_slots", {})
        if not isinstance(slots, Mapping):
            slots = {}
        disclosed_category = "category" in {_canonical(key) for key in slots}
        meaningful_category = bool(category and (category != "clothing" or disclosed_category))
        scope = category if meaningful_category else None

        negatives = {
            value
            for raw in _iter_values(final_fast_memory.get("negated_terms", ()))
            if (value := _canonical(raw))
        }
        facts: list[tuple[str, str, MemoryPolarity, str]] = []
        if meaningful_category:
            facts.append(("category", category, MemoryPolarity.POSITIVE, f"category: {category}"))

        raw_price_max = final_fast_memory.get("price_max", 9999.0)
        if not isinstance(raw_price_max, bool):
            try:
                price_max = float(raw_price_max)
            except (TypeError, ValueError):
                price_max = 9999.0
            if math.isfinite(price_max) and 0.0 <= price_max < 9999.0:
                budget = format(price_max, "g")
                facts.append(
                    (
                        "budget",
                        budget,
                        MemoryPolarity.POSITIVE,
                        f"budget: at most {budget}",
                    )
                )

        for raw_attribute in sorted(slots, key=lambda value: _canonical(value)):
            attribute = _canonical(raw_attribute)
            if not attribute or attribute in _NON_FACT_SLOT_NAMES:
                continue
            for raw_value in _iter_values(slots[raw_attribute]):
                value = _canonical(raw_value)
                if value in _POSITIVE_MARKERS:
                    value = attribute
                elif value in _NEGATIVE_MARKERS:
                    continue
                if not value or value in negatives:
                    continue
                if attribute == "category" and value == category and meaningful_category:
                    continue
                facts.append(
                    (attribute, value, MemoryPolarity.POSITIVE, f"{attribute}: {value}")
                )

        for value in sorted(negatives):
            facts.append(("negative", value, MemoryPolarity.NEGATIVE, f"avoid: {value}"))

        drafts: list[MemoryDraft] = []
        seen: set[tuple[str, str, MemoryPolarity]] = set()
        for fact_type, value, polarity, text in facts:
            key = (fact_type, value, polarity)
            if key in seen:
                continue
            seen.add(key)
            drafts.append(
                MemoryDraft(
                    id=_memory_id(
                        user_id=user,
                        session_id=session,
                        sequence_index=sequence_index,
                        fact_type=fact_type,
                        value=value,
                        polarity=polarity,
                    ),
                    text=text,
                    source=MemorySource.USER,
                    polarity=polarity,
                    scope=scope,
                )
            )
        return tuple(drafts)

    def embed_drafts(self, drafts: Sequence[MemoryDraft]) -> tuple[MemoryItem, ...]:
        """Explicitly embed drafts using the agent's existing query backend."""

        items: list[MemoryItem] = []
        expected_match = _SPACE_RE.search(self.embedding_space_id)
        expected_dimension = int(expected_match.group(1)) if expected_match else None
        for draft in drafts:
            vector = np.array(
                self.embedding_backend.embed_query(draft.text),
                dtype=np.float64,
                copy=True,
            )
            if vector.ndim != 1 or vector.size == 0 or not np.all(np.isfinite(vector)):
                raise ValueError("memory embedding must be a finite one-dimensional vector")
            if expected_dimension is not None and vector.size != expected_dimension:
                raise ValueError(
                    f"memory embedding dimension {vector.size} does not match "
                    f"embedding space dimension {expected_dimension}"
                )
            norm = float(np.linalg.norm(vector))
            if not math.isclose(norm, 1.0, rel_tol=1e-5, abs_tol=1e-6):
                raise ValueError("memory embedding backend must return L2-normalized vectors")
            # M0's float32 unit tolerance is deliberately wider than QLMP's
            # float64 geometry tolerance.  Re-normalize only this owned local
            # copy before MemoryItem takes ownership; never alter the backend
            # vector or relax QLMP's epsilon.
            vector /= norm
            items.append(
                MemoryItem(
                    id=draft.id,
                    text=draft.text,
                    embedding=vector,
                    source=draft.source,
                    polarity=draft.polarity,
                    scope=draft.scope,
                    timestamp=None,
                    confidence=draft.confidence,
                )
            )
        return tuple(items)

    def extract_and_embed(
        self,
        final_fast_memory: Mapping[str, object],
        *,
        user_id: str,
        session_id: str,
        sequence_index: int,
    ) -> tuple[MemoryItem, ...]:
        drafts = self.extract_drafts(
            final_fast_memory,
            user_id=user_id,
            session_id=session_id,
            sequence_index=sequence_index,
        )
        return self.embed_drafts(drafts)


def qlmp_items(
    values: Iterable[object],
    *,
    expected_embedding_space_id: str | None = None,
) -> tuple[MemoryItem, ...]:
    """Return stored items in the exact existing QLMP helper input schema."""

    items: list[MemoryItem] = []
    for value in values:
        item = value if isinstance(value, MemoryItem) else getattr(value, "item", None)
        if not isinstance(item, MemoryItem):
            raise ValueError("values must be MemoryItem or stored-memory records")
        record_space = getattr(value, "embedding_space_id", None)
        if (
            expected_embedding_space_id is not None
            and record_space is not None
            and record_space != expected_embedding_space_id
        ):
            raise ValueError("stored memory belongs to a different embedding space")
        items.append(item)
    return tuple(items)


def qlmp_projection_memory(item: MemoryItem) -> np.ndarray:
    """Expose the vector accepted by existing project_memory_residual()."""

    if not isinstance(item, MemoryItem):
        raise ValueError("item must be a QLMP MemoryItem")
    return item.embedding


__all__ = [
    "FastMemoryQLMPAdapter",
    "MemoryDraft",
    "qlmp_items",
    "qlmp_projection_memory",
]
