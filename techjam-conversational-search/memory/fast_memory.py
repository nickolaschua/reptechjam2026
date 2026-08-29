"""Fast Memory state and deterministic observable-message parsing.

The organizer templates intentionally preserve the starter agent's parsing at
the category/positive-constraint boundary. Free-form handling only enriches
the typed state used for the final Slow Memory summary.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Iterable, Protocol, runtime_checkable

from .types import ConstraintKind, FastMemoryState, TypedConstraint


WS_RE = re.compile(r"\s+")
INITIAL_PREFIX = "i'm looking for "
BUYING_PREFIX = "a key requirement is:"
DISCLOSURE_PREFIX = "for that, what matters is:"
OVERRIDE_PREFIX = "actually, ignore my earlier preference. what i need is:"
EXPLORATORY_CUES = ("still exploring", "just browsing", "not sure yet", "open to ideas")
TOPIC_SHIFT_PATTERNS = (
    re.compile(
        r"actually[,]?\s+(?:forget|ignore)\b.+?\b(?:i\s+)?need\s+(?:an?\s+|the\s+)?(.+?)(?:\s+for\b|[.!?]|$)",
        re.I,
    ),
    re.compile(r"(?:switch(?:ing)?|change|move) (?:the )?(?:category|topic) to\s+(.+)", re.I),
    re.compile(r"(?:now|instead)[, ]+(?:i'm|i am) looking for\s+(.+)", re.I),
    re.compile(r"(?:let's|lets) (?:look for|find)\s+(.+?) instead", re.I),
    re.compile(r"actually[,]? i (?:want|need)\s+(.+?) instead", re.I),
)
MATERIALS = ("cotton", "polyester", "nylon", "leather", "wool", "spandex", "silk", "rayon", "fabric")
COLORS = ("black", "white", "blue", "red", "pink", "green", "brown", "gray", "grey", "purple", "yellow", "orange")
USE_CASES = ("hiking", "running", "gym", "winter", "outdoor", "work", "walking", "sports", "casual", "travel")


def normalize(value: object) -> str:
    return WS_RE.sub(" ", str(value or "").lower()).strip()


def clean_constraint(value: str) -> str:
    return WS_RE.sub(" ", value).strip().rstrip(".").strip()


def classify_constraint(value: str) -> ConstraintKind:
    lowered = normalize(value)
    if "budget" in lowered or "price" in lowered or re.search(
        r"(?:\$|<=|under|below|less than|ceiling|up to)\s*\$?\d", lowered
    ):
        return ConstraintKind.BUDGET
    if any(re.search(rf"\b{re.escape(word)}\b", lowered) for word in MATERIALS):
        return ConstraintKind.MATERIAL
    if "color" in lowered or any(re.search(rf"\b{word}\b", lowered) for word in COLORS):
        return ConstraintKind.COLOR
    if any(word in lowered for word in ("size", "sizing", "width", "wide", "narrow")):
        return ConstraintKind.SIZE
    if any(word in lowered for word in ("department", "style", "fit", "sleeve", "neck", "formal")):
        return ConstraintKind.STYLE
    if any(word in lowered for word in ("brand", "store", "maker", "made by")):
        return ConstraintKind.BRAND
    if any(re.search(rf"\b{re.escape(word)}\b", lowered) for word in USE_CASES):
        return ConstraintKind.USE_CASE
    return ConstraintKind.FEATURE


@dataclass(frozen=True)
class FastMemoryUpdate:
    """Authoritative typed result supplied by an optional semantic parser.

    Omitted scalar fields retain their current value. Constraint collections
    are applied in their supplied hard/soft/negative groups. Set
    ``replace_constraints`` for a full authoritative replacement.
    """

    category: str | None = None
    intent: str | None = None
    hard_constraints: tuple[TypedConstraint, ...] = ()
    soft_preferences: tuple[TypedConstraint, ...] = ()
    negatives: tuple[TypedConstraint, ...] = ()
    topic_override: bool = False
    replace_constraints: bool = False
    confidence: float | None = None


@runtime_checkable
class SemanticParser(Protocol):
    def parse(self, message: str, turn: int) -> FastMemoryUpdate | None:
        """Return an authoritative update, or ``None`` for fallback parsing."""


def _constraint(
    value: str,
    *,
    turn: int,
    epoch: int,
    hard: bool,
    negated: bool = False,
    explicit: bool = True,
    strength: float | None = None,
    confidence: float | None = None,
    source: str,
) -> TypedConstraint:
    cleaned = clean_constraint(value)
    return TypedConstraint(
        value=cleaned,
        kind=classify_constraint(cleaned),
        hard=hard,
        negated=negated,
        explicit=explicit,
        strength=(1.0 if hard or negated else 0.75) if strength is None else strength,
        confidence=(1.0 if hard or negated else 0.85) if confidence is None else confidence,
        source_turn=turn,
        source=source,
        intent_epoch=epoch,
    )


class _StateUpdater:
    def __init__(self, state: FastMemoryState) -> None:
        self.state = state

    @staticmethod
    def _contains(items: Iterable[TypedConstraint], value: str) -> bool:
        target = normalize(value)
        return any(normalize(item.value) == target for item in items)

    def _append(self, item: TypedConstraint) -> None:
        if not item.value:
            return
        state = self.state
        if item.negated:
            item = replace(item, hard=True, negated=True)
            if not self._contains(state.negatives, item.value):
                state.negatives.append(item)
            return
        target = normalize(item.value)
        if self._contains(state.hard_constraints, target):
            return
        matching_soft = next(
            (value for value in state.soft_preferences if normalize(value.value) == target),
            None,
        )
        if matching_soft is not None:
            if item.hard:
                state.soft_preferences.remove(matching_soft)
                state.hard_constraints.append(replace(item, source_turn=matching_soft.source_turn))
            return
        (state.hard_constraints if item.hard else state.soft_preferences).append(item)

    def _remove_seed(self) -> None:
        seed = self.state.override_seed
        if not seed:
            return
        target = normalize(seed)
        removed: TypedConstraint | None = None
        kept_hard: list[TypedConstraint] = []
        kept_soft: list[TypedConstraint] = []
        for item in self.state.hard_constraints:
            if normalize(item.value) == target and removed is None:
                removed = item
            else:
                kept_hard.append(item)
        for item in self.state.soft_preferences:
            if normalize(item.value) == target and removed is None:
                removed = item
            else:
                kept_soft.append(item)
        self.state.hard_constraints = kept_hard
        self.state.soft_preferences = kept_soft
        if removed is not None:
            self._append(replace(
                removed,
                negated=True,
                hard=True,
                explicit=True,
                strength=1.0,
                confidence=1.0,
                source_turn=self.state.source_turn,
                source="replacement_rejection",
            ))
        self.state.override_seed = None

    def topic_shift(self, category: str, turn: int) -> None:
        state = self.state
        category = clean_constraint(category.split(".", 1)[0])
        category = re.sub(r"\s+instead$", "", category, flags=re.I).strip()
        if not category:
            return
        state.intent_epoch += 1
        state.topic_override = True
        state.category = category
        state.category_source_turn = turn
        portable_budget: list[TypedConstraint] = []
        for item in state.active_constraints:
            if item.kind != ConstraintKind.BUDGET:
                continue
            match = re.search(
                r"(?:budget(?:\s+is|\s+around)?|under|below|less than|ceiling(?:\s+of)?|up to|<=)\s*\$?\s*\d+(?:\.\d+)?",
                item.value,
                re.I,
            )
            if match:
                portable_budget.append(replace(
                    item,
                    value=clean_constraint(match.group(0)),
                    intent_epoch=state.intent_epoch,
                ))
        state.hard_constraints = [item for item in portable_budget if item.hard]
        state.soft_preferences = [item for item in portable_budget if not item.hard]
        state.negatives = []
        state.override_seed = None
        state.confidence = 1.0

    def apply_semantic(self, update: FastMemoryUpdate, turn: int) -> None:
        state = self.state
        if update.topic_override and update.category is not None:
            self.topic_shift(update.category, turn)
        elif update.category is not None:
            state.category = clean_constraint(update.category)
            state.category_source_turn = turn
        if update.intent is not None:
            state.intent = str(update.intent)
            state.intent_source_turn = turn
            state.intent_locked = True
        if update.replace_constraints:
            state.hard_constraints = []
            state.soft_preferences = []
            state.negatives = []
        for collection, hard, negated in (
            (update.hard_constraints, True, False),
            (update.soft_preferences, False, False),
            (update.negatives, True, True),
        ):
            for item in collection:
                if not isinstance(item, TypedConstraint):
                    raise TypeError("semantic constraints must be TypedConstraint values")
                source_turn = item.source_turn if item.source_turn > 0 else turn
                self._append(replace(
                    item,
                    hard=hard,
                    negated=negated,
                    source_turn=source_turn,
                    intent_epoch=state.intent_epoch,
                ))
        if update.confidence is not None:
            state.confidence = float(update.confidence)

    def _parse_initial(self, message: str, turn: int, *, later: bool = False) -> bool:
        normalized = normalize(message)
        if not normalized.startswith(INITIAL_PREFIX):
            return False
        body = message[len("I'm looking for "):]
        if ", but I'm still exploring" in body:
            category = body.split(", but I'm still exploring", 1)[0]
            remainder = ""
            intent = "browsing"
        else:
            category, separator, remainder = body.partition(".")
            if not separator:
                remainder = ""
            intent = "browsing" if any(cue in normalized for cue in EXPLORATORY_CUES) else "buying"
        if later:
            self.topic_shift(category, turn)
        else:
            self.state.category = clean_constraint(category)
            self.state.category_source_turn = turn
        self.state.intent = intent
        self.state.intent_source_turn = turn
        self.state.intent_locked = True
        remainder = remainder.strip()
        if normalize(remainder).startswith(BUYING_PREFIX):
            self._append(_constraint(
                remainder.split(":", 1)[1], turn=turn, epoch=self.state.intent_epoch,
                hard=True, source="explicit_requirement",
            ))
        elif remainder:
            value = clean_constraint(remainder)
            self._append(_constraint(
                value, turn=turn, epoch=self.state.intent_epoch,
                hard=False, source="initial_preference",
            ))
            self.state.override_seed = value
        self.state.confidence = 1.0 if self.state.category else 0.5
        return True

    def parse_deterministic(self, message: str, turn: int) -> None:
        state = self.state
        normalized = normalize(message)
        if turn == 1 and self._parse_initial(message, turn):
            return
        if normalized.startswith(DISCLOSURE_PREFIX):
            payload = message.split(":", 1)[1].strip().rstrip(".")
            for value in payload.split("; "):
                self._append(_constraint(
                    value, turn=turn, epoch=state.intent_epoch, hard=False,
                    explicit=True, strength=0.85, confidence=0.9,
                    source="explicit_disclosure",
                ))
            state.confidence = max(state.confidence, 0.9)
            return
        if normalized.startswith(OVERRIDE_PREFIX):
            self._remove_seed()
            self._append(_constraint(
                message.split(":", 1)[1], turn=turn, epoch=state.intent_epoch,
                hard=True, source="explicit_replacement",
            ))
            state.confidence = 1.0
            return
        if turn > 1 and normalized.startswith(INITIAL_PREFIX):
            self._parse_initial(message, turn, later=True)
            return
        if turn == 1 and not state.intent_locked:
            state.intent = "browsing" if any(cue in normalized for cue in EXPLORATORY_CUES) else "buying"
            state.intent_source_turn = turn
            state.intent_locked = True
        self._parse_free_form(message, turn)

    def _parse_free_form(self, message: str, turn: int) -> None:
        state = self.state
        lowered = normalize(message)
        for pattern in TOPIC_SHIFT_PATTERNS:
            match = pattern.search(message)
            if match:
                self.topic_shift(match.group(1), turn)
                return

        negative = re.search(
            r"(?:don't want|dont want|do not want|avoid|not interested in|no|\bnot)\s+([^,.;!?]+)",
            message,
            re.I,
        )
        if negative:
            value = clean_constraint(negative.group(1))
            self._append(_constraint(
                value, turn=turn, epoch=state.intent_epoch, hard=True,
                negated=True, source="explicit_rejection",
            ))
            target = normalize(value)
            state.hard_constraints = [item for item in state.hard_constraints if normalize(item.value) != target]
            state.soft_preferences = [item for item in state.soft_preferences if normalize(item.value) != target]
            state.confidence = max(state.confidence, 1.0)
            return

        hard_match = re.search(
            r"(?:i (?:need|must have|require)|must be|requirement(?: is)?|under|below)\s*:?\s*(.+)",
            message,
            re.I,
        )
        if hard_match:
            self._append(_constraint(
                hard_match.group(1), turn=turn, epoch=state.intent_epoch,
                hard=True, source="explicit_requirement",
            ))
            state.confidence = max(state.confidence, 1.0)
            return
        preference = re.search(
            r"(?:i (?:prefer|like|would like)|preference(?: is)?|nice to have)\s*:?\s*(.+)",
            message,
            re.I,
        )
        if preference:
            self._append(_constraint(
                preference.group(1), turn=turn, epoch=state.intent_epoch,
                hard=False, source="explicit_preference",
            ))
            state.confidence = max(state.confidence, 0.85)
            return

        if any(cue in lowered for cue in ("ignore ", "forget ", "doesn't matter", "doesnt matter")):
            for existing in list(state.active_constraints):
                if normalize(existing.value) in lowered:
                    state.hard_constraints = [item for item in state.hard_constraints if item != existing]
                    state.soft_preferences = [item for item in state.soft_preferences if item != existing]
                    self._append(replace(
                        existing, negated=True, hard=True,
                        source_turn=turn, source="explicit_rejection",
                    ))
            return

        ignored = ("don't have", "dont have", "not quite right", "ask me about")
        if clean_constraint(message) and not any(cue in lowered for cue in ignored):
            self._append(_constraint(
                message, turn=turn, epoch=state.intent_epoch, hard=False,
                explicit=False, strength=0.60, confidence=0.60,
                source="inferred_free_form",
            ))
            state.confidence = max(state.confidence, 0.60)


def update_state(
    state: FastMemoryState,
    message: str,
    turn: int,
    semantic_parser: SemanticParser | None = None,
) -> FastMemoryState:
    """Update Fast Memory once, using semantic output or deterministic parsing."""

    if turn < 1:
        raise ValueError("turn must be positive")
    if turn <= state.source_turn:
        raise ValueError(f"turn {turn} is not after prior turn {state.source_turn}")

    update = semantic_parser.parse(message, turn) if semantic_parser is not None else None
    if update is not None and not isinstance(update, FastMemoryUpdate):
        raise TypeError("semantic parser must return FastMemoryUpdate or None")

    state.history.append(message)
    state.source_turn = turn
    updater = _StateUpdater(state)
    if update is None:
        updater.parse_deterministic(message, turn)
    else:
        updater.apply_semantic(update, turn)
    return state


def override_intent(
    state: FastMemoryState,
    category: str,
    turn: int,
    *,
    intent: str | None = None,
) -> FastMemoryState:
    """Apply an explicit upstream topic/intent override."""

    if turn < 1 or turn < state.source_turn:
        raise ValueError("override turn cannot precede the current session turn")
    if turn > state.source_turn:
        state.source_turn = turn
    updater = _StateUpdater(state)
    updater.topic_shift(category, turn)
    if intent is not None:
        state.intent = str(intent)
        state.intent_source_turn = turn
        state.intent_locked = True
    return state


__all__ = [
    "FastMemoryUpdate",
    "SemanticParser",
    "classify_constraint",
    "clean_constraint",
    "normalize",
    "override_intent",
    "update_state",
]
