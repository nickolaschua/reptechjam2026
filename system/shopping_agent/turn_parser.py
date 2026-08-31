"""Constrained free-text turn parsing for the active shopping agent.

This is the production form of Winston's bolt-on parser: Llama 3.1 is constrained
by an Ollama JSON schema, then deterministic validation decides which extracted
facts are safe to act on.  Category resolution uses only the completed lexical
soft-slot arm and is advisory.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import re
import time
from typing import Any, Literal, Mapping, Protocol

try:
    from .category_resolver import CategoryResolver
    from .model_client import ModelClient, ModelError
    from .runtime import get_runtime_providers
except ImportError:  # pragma: no cover - direct script compatibility
    from category_resolver import CategoryResolver
    from model_client import ModelClient, ModelError
    from runtime import get_runtime_providers


ALLOWED_ATTRIBUTES = (
    "category",
    "material",
    "color",
    "size",
    "style",
    "brand",
    "budget",
    "feature",
    "use_case",
    "other",
)
SLOT_ATTRIBUTES = tuple(value for value in ALLOWED_ATTRIBUTES if value != "category")
DEPARTMENTS = (
    "womens",
    "mens",
    "girls",
    "boys",
    "baby-girls",
    "baby-boys",
    "unisex-adult",
    "unisex-child",
    None,
)
QUALITY_PRIORS = ("none", "well_rated", "reputable_brand")
SPECIFICITIES = (
    "scenario_only",
    "type_with_wishes",
    "type_with_requirements",
)
MESSAGE_TYPES = (
    "exact",
    "compatibility",
    "symptom",
    "product_type",
    "use_case",
    "feature",
)
HARD_ATTRIBUTES = frozenset({"brand", "material", "size", "budget"})

SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "category_phrase": {
            "type": "string",
            "maxLength": 40,
            "description": "The product type in the user's words, 1-4 words. Never a brand.",
        },
        "department": {
            "type": ["string", "null"],
            "enum": list(DEPARTMENTS),
            "description": "The stated wearer demographic, otherwise null. Never infer it.",
        },
        "slots": {
            "type": "array",
            "maxItems": 8,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "attribute": {"type": "string", "enum": list(SLOT_ATTRIBUTES)},
                    "value": {"type": "string", "maxLength": 40},
                    "declined": {"type": "boolean"},
                    "negated": {"type": "boolean"},
                },
                "required": ["attribute", "value", "declined", "negated"],
            },
        },
        "price_max": {"type": ["number", "null"]},
        "price_min": {"type": ["number", "null"]},
        "quality_prior": {"type": "string", "enum": list(QUALITY_PRIORS)},
        "exploring": {"type": "boolean"},
        "specificity": {"type": "string", "enum": list(SPECIFICITIES)},
    },
    "required": [
        "category_phrase",
        "department",
        "slots",
        "price_max",
        "price_min",
        "quality_prior",
        "exploring",
        "specificity",
    ],
}

PROMPT = """Extract shopping constraints from the message. Output only what the user said.

Rules:
- department: only if stated. A dress does not imply womens.
- Do not create a slot for a self-correction, aside, or question the user answered.
- quality_prior is none unless ratings, reviews, popularity, or brand reputation is explicit.
- slots may be empty. Never use placeholders or shopping-stance words as slot values.
- declined is true only when the user says the attribute does not matter.
- negated is true only when the user explicitly rejects that value; still record the value.
- specificity is scenario_only without a product type, type_with_wishes for soft wishes,
  and type_with_requirements only for firm size, brand, material, or price requirements.

Message: {utterance}"""

_MATERIALS = frozenset({
    "cotton", "polyester", "nylon", "leather", "wool", "spandex", "silk", "rayon",
    "denim", "suede", "canvas", "mesh", "fleece", "cashmere", "linen", "velvet",
    "satin", "acrylic", "viscose", "elastane", "rubber", "metal", "plastic", "gold",
    "silver", "alloy", "stainless", "sterling", "crystal", "gemstone", "wood", "bamboo",
    "microfiber", "chiffon", "lace", "jersey", "corduroy", "tweed", "synthetic",
    "fabric", "pearl", "diamond", "brass", "copper", "titanium", "nickel", "zinc",
    "resin", "felt", "flannel", "sequin",
})
_MODIFIERS = frozenset({
    "real", "genuine", "faux", "natural", "pure", "solid", "blend", "blended", "soft", "fine",
})
_NEGATIONS = frozenset({"not", "without", "no", "non"})
_SIZE_WORDS = frozenset({
    "plus", "petite", "tall", "big", "wide", "narrow", "regular", "slim", "toddler",
    "kids", "junior", "infant", "youth", "adult", "short", "long", "xs", "s", "m", "l",
    "xl", "xxl", "xxxl", "small", "medium", "large", "size", "width",
})
_SIZE_NUMERIC = re.compile(r"^\d{1,2}(?:\.\d)?[a-z]{0,3}$|^\d{2}x\d{2}$")
_JUNK_VALUES = frozenset({
    "", "not specified", "unspecified", "none", "n/a", "na", "any", "not mentioned",
    "does not matter", "doesn't matter", "no preference", "not applicable", "null", "unknown",
    "not stated", "scenario_only", "type_with_wishes", "type_with_requirements", "well_rated",
    "reputable_brand", "reputable/trusted", "reputable", "trusted", "reputable or trusted",
    "exploring", "still exploring", "browsing", "just browsing", "open to options", "not sure",
})
_DECLINE_CUES = (
    "don't care", "dont care", "doesn't matter", "doesnt matter", "does not matter",
    "do not care", "not important", "no preference",
)
_WEARER_RE = re.compile(
    r"\b(?:wom[ae]n|female|lad(?:y|ies)|girls?|wife|mum|mom|mother|sister|daughter|niece|"
    r"girlfriend|m[ae]n|male|guys?|husband|dad|father|brother|son|nephew|boyfriend|"
    r"boys?|kids?|child(?:ren)?|toddler|bab(?:y|ies)|infant|unisex|anyone|everyone|"
    r"his|hers?|him|she|he)\b",
    re.IGNORECASE,
)
_NEG_CUES = r"(?:not|no|non|don'?t|dont|never|without|avoid|hates?|nothing|isn'?t|aren'?t|can'?t|won'?t)"
_NEG_WINDOW = 6
_MODEL_CODE_RE = re.compile(r"\b([A-Za-z]{1,4}-?\d{3,}[A-Za-z0-9-]*)\b")
_GRADES = frozenset({"316L", "925", "S925", "14K", "18K", "10K", "585", "750", "24K", "9K", "UV400"})
_EXPLORING_RE = re.compile(
    r"\b(?:just browsing|still exploring|not sure (?:yet|what)|open to (?:ideas|suggestions|anything)|"
    r"any (?:ideas|suggestions|recommendations)|no idea|looking around|what do you (?:have|recommend))\b",
    re.IGNORECASE,
)
_COMPAT_RE = re.compile(
    r"\b(?:i (?:already )?own|to (?:go|fit|match) with my|(?:fits?|for) my "
    r"(?:watch|bracelet|sneakers|boots|shoes|band))\b",
    re.IGNORECASE,
)


class TurnParserError(RuntimeError):
    """Base error for a failed authoritative free-text parse."""

    def __init__(
        self,
        message: str,
        *,
        model: str,
        latency_seconds: float,
        attempts: int,
        cause_type: str | None = None,
    ) -> None:
        super().__init__(message)
        self.model = model
        self.latency_seconds = float(latency_seconds)
        self.attempts = int(attempts)
        self.retry_count = max(0, self.attempts - 1)
        self.cause_type = cause_type


class ParserRequestError(TurnParserError):
    """Raised after the constrained Ollama request exhausts its retry."""


class CategoryResolutionError(TurnParserError):
    """Raised when the completed resolver cannot produce telemetry."""


@dataclass(frozen=True)
class ParsedSlot:
    attribute: str
    value: str
    tier: Literal["hard", "soft"]


@dataclass(frozen=True)
class ParsedTurn:
    category: str | None
    positive_slots: tuple[ParsedSlot, ...]
    negatives: tuple[ParsedSlot, ...]
    declined_attributes: tuple[str, ...]
    price_min: float | None
    price_max: float | None
    department: str | None
    specificity: str
    intent: Literal["buying", "browsing"]
    message_type: str
    model_code: str | None
    resolver_candidates: tuple[str, ...]
    resolver_confidence: float
    raw_parse: Mapping[str, Any]

    @property
    def soft_slot_values(self) -> tuple[str, ...]:
        return tuple(slot.value for slot in self.positive_slots if slot.tier == "soft")

    @property
    def has_trusted_hard_constraint(self) -> bool:
        return bool(
            self.department
            or self.negatives
            or any(slot.tier == "hard" for slot in self.positive_slots)
        )


class TurnParser(Protocol):
    model: str

    def parse(self, message: str, turn: int) -> ParsedTurn: ...


def validate_raw_parse(value: object) -> dict[str, Any]:
    """Validate the response even though Ollama constrained its decoding."""

    if not isinstance(value, dict):
        raise ValueError("parser response must be an object")
    required = set(SCHEMA["required"])
    missing = sorted(required - set(value))
    if missing:
        raise ValueError(f"parser response is missing required keys: {', '.join(missing)}")
    if set(value) - set(SCHEMA["properties"]):
        raise ValueError("parser response contains unknown keys")
    category = value["category_phrase"]
    if not isinstance(category, str) or len(category) > 40:
        raise ValueError("category_phrase must be a string of at most 40 characters")
    if value["department"] not in DEPARTMENTS:
        raise ValueError("department is outside the constrained enum")
    slots = value["slots"]
    if not isinstance(slots, list) or len(slots) > 8:
        raise ValueError("slots must be an array with at most eight entries")
    for slot in slots:
        if not isinstance(slot, dict) or set(slot) != {"attribute", "value", "declined", "negated"}:
            raise ValueError("each slot must contain exactly attribute, value, declined, and negated")
        if slot["attribute"] not in SLOT_ATTRIBUTES:
            raise ValueError("slot attribute is outside the constrained enum")
        if not isinstance(slot["value"], str) or len(slot["value"]) > 40:
            raise ValueError("slot value must be a string of at most 40 characters")
        if type(slot["declined"]) is not bool or type(slot["negated"]) is not bool:
            raise ValueError("slot declined and negated values must be booleans")
    for key in ("price_min", "price_max"):
        price = value[key]
        if price is not None and (isinstance(price, bool) or not isinstance(price, (int, float))):
            raise ValueError(f"{key} must be a number or null")
        if price is not None and not math.isfinite(float(price)):
            raise ValueError(f"{key} must be finite")
    if value["quality_prior"] not in QUALITY_PRIORS:
        raise ValueError("quality_prior is outside the constrained enum")
    if type(value["exploring"]) is not bool:
        raise ValueError("exploring must be a boolean")
    if value["specificity"] not in SPECIFICITIES:
        raise ValueError("specificity is outside the constrained enum")
    return dict(value)


def negation_supported(utterance: str, value: str) -> bool:
    words = [word for word in re.split(r"[^a-z0-9']+", value.lower()) if len(word) > 2]
    if not words:
        return False
    head = re.escape(words[0])
    gap = r"(?:\s+[a-z0-9'&-]+){0," + str(_NEG_WINDOW) + r"}"
    return re.search(
        rf"\b{_NEG_CUES}{gap}\s+(?:[a-z0-9'&-]+\s+)?{head}",
        utterance.lower(),
    ) is not None


def hard_claim_holds(attribute: str, value: str, stores: frozenset[str]) -> bool:
    tokens = [token for token in re.split(r"[^a-z0-9]+", value.lower()) if token]
    if not tokens or _NEGATIONS & set(tokens):
        return False
    core = [token for token in tokens if token not in _MODIFIERS]
    if not core:
        return False
    if attribute == "brand":
        return value.strip().lower() in stores or " ".join(core) in stores
    if attribute == "budget":
        return any(character.isdigit() for character in value)
    if attribute == "size":
        return len(core) <= 2 and all(token in _SIZE_WORDS or _SIZE_NUMERIC.match(token) for token in core)
    if attribute == "material":
        return len(core) <= 2 and all(token in _MATERIALS for token in core)
    return True


def tier_of(slot: Mapping[str, Any], stores: frozenset[str]) -> Literal["hard", "soft", "decline"]:
    if slot.get("declined"):
        return "decline"
    attribute = str(slot["attribute"])
    if attribute not in HARD_ATTRIBUTES:
        return "soft"
    return "hard" if hard_claim_holds(attribute, str(slot["value"]), stores) else "soft"


def resolver_soft_slot_values(
    parse: Mapping[str, Any], stores: frozenset[str]
) -> tuple[str, ...]:
    """Reproduce the measured resolver arm before state-side negation repair."""

    values: list[str] = []
    for slot in parse.get("slots", ()):
        value = str(slot.get("value") or "").strip()
        if value.lower().rstrip(".") in _JUNK_VALUES:
            continue
        if slot.get("declined") or slot.get("negated"):
            continue
        if tier_of(slot, stores) == "soft":
            values.append(value)
    return tuple(values)


def clean_parse(raw: Mapping[str, Any], utterance: str) -> dict[str, Any]:
    slots: list[dict[str, Any]] = []
    seen: set[tuple[str, str, bool]] = set()
    for candidate in raw.get("slots", []):
        value = str(candidate.get("value") or "").strip()
        lowered = value.lower().rstrip(".")
        declined = bool(candidate.get("declined"))
        negated = bool(candidate.get("negated"))
        if negated:
            core = re.sub(r"^(?:not|no|non|without|never)\s+", "", value, flags=re.IGNORECASE)
            negated = negation_supported(utterance, core)
        if declined or any(cue in lowered for cue in _DECLINE_CUES):
            declined, value, lowered = True, "", ""
        elif lowered in _JUNK_VALUES:
            continue
        key = (str(candidate.get("attribute")), lowered, declined)
        if key in seen:
            continue
        seen.add(key)
        slots.append({
            "attribute": str(candidate["attribute"]),
            "value": value,
            "declined": declined,
            "negated": negated,
        })
    result = dict(raw)
    result["slots"] = slots
    for key in ("price_max", "price_min"):
        price = result.get(key)
        result[key] = None if price is None or float(price) <= 0 else float(price)
    if result.get("department") and not _WEARER_RE.search(utterance):
        result["department"] = None
    return result


def model_code(message: str) -> str | None:
    for match in _MODEL_CODE_RE.finditer(message):
        code = match.group(1)
        if code.upper() in _GRADES or code.isdigit() or not re.search(r"[A-Za-z]", code):
            continue
        return code.upper()
    return None


def _intent_of(parse: Mapping[str, Any], message: str, stores: frozenset[str]) -> Literal["buying", "browsing"]:
    if _EXPLORING_RE.search(message):
        return "browsing"
    if _COMPAT_RE.search(message) or model_code(message):
        return "buying"
    hard = sum(1 for slot in parse.get("slots", ()) if tier_of(slot, stores) == "hard")
    concrete_color = any(
        slot["attribute"] == "color" and not slot.get("declined") and str(slot.get("value") or "").strip()
        for slot in parse.get("slots", ())
    )
    priced = parse.get("price_max") is not None or parse.get("price_min") is not None
    return "buying" if hard or concrete_color or priced else "browsing"


def _message_type_of(parse: Mapping[str, Any], message: str) -> str:
    if model_code(message):
        return "exact"
    if _COMPAT_RE.search(message):
        return "compatibility"
    if not str(parse.get("category_phrase") or "").strip():
        return "symptom"
    slots = [slot for slot in parse.get("slots", ()) if not slot.get("declined")]
    if not slots and parse.get("price_max") is None and parse.get("price_min") is None:
        return "product_type"
    if {slot["attribute"] for slot in slots} <= {"use_case"}:
        return "use_case"
    return "feature"


def parsed_turn_from_raw(
    raw: Mapping[str, Any],
    message: str,
    *,
    stores: frozenset[str],
    resolver_candidates: tuple[str, ...] = (),
    resolver_confidence: float = 0.0,
) -> ParsedTurn:
    cleaned = clean_parse(validate_raw_parse(dict(raw)), message)
    positives: list[ParsedSlot] = []
    negatives: list[ParsedSlot] = []
    declined: list[str] = []
    for slot in cleaned["slots"]:
        tier = tier_of(slot, stores)
        attribute = slot["attribute"]
        if tier == "decline":
            if attribute not in declined:
                declined.append(attribute)
            continue
        value = str(slot["value"])
        if slot.get("negated"):
            value = re.sub(r"^(?:not|no|non|without|never)\s+", "", value, flags=re.IGNORECASE)
            negatives.append(ParsedSlot(attribute, value, "hard" if tier == "hard" else "soft"))
        else:
            positives.append(ParsedSlot(attribute, value, tier))
    return ParsedTurn(
        category=str(cleaned.get("category_phrase") or "").strip() or None,
        positive_slots=tuple(positives),
        negatives=tuple(negatives),
        declined_attributes=tuple(declined),
        price_min=cleaned.get("price_min"),
        price_max=cleaned.get("price_max"),
        department=cleaned.get("department"),
        specificity=str(cleaned["specificity"]),
        intent=_intent_of(cleaned, message, stores),
        message_type=_message_type_of(cleaned, message),
        model_code=model_code(message),
        resolver_candidates=tuple(resolver_candidates),
        resolver_confidence=float(resolver_confidence),
        raw_parse=cleaned,
    )


class WinstonTurnParser:
    """Authoritative constrained parser with one retry and typed failures."""

    def __init__(
        self,
        resolver: CategoryResolver,
        *,
        client: ModelClient | None = None,
        catalog_stores: frozenset[str] | None = None,
    ) -> None:
        self.resolver = resolver
        self.client = client or get_runtime_providers().llm_client
        self.model = self.client.model
        self.last_call: dict[str, Any] | None = None
        if catalog_stores is None:
            catalog_stores = resolver.catalog_stores
        self.catalog_stores = catalog_stores

    @staticmethod
    def _decode_content(content: str) -> dict[str, Any]:
        return validate_raw_parse(json.loads(content))

    def _request_parse(self, message: str) -> dict[str, Any]:
        call = self.client.chat_result(
            [{"role": "user", "content": PROMPT.format(utterance=message)}],
            format=SCHEMA,
            options={"temperature": 0, "num_predict": 512},
            role="parser",
            validator=self._decode_content,
        )
        self.last_call = call.instrumentation()
        return self._decode_content(call.content)

    def parse(self, message: str, turn: int) -> ParsedTurn:
        del turn
        started = time.perf_counter()
        try:
            raw = self._request_parse(message)
        except ModelError as exc:
            self.last_call = exc.instrumentation()
            raise ParserRequestError(
                f"constrained parser failed: {exc}",
                model=exc.model,
                latency_seconds=exc.latency_seconds,
                attempts=exc.attempts,
                cause_type=type(exc).__name__,
            ) from exc

        cleaned = clean_parse(raw, message)
        try:
            soft_values = resolver_soft_slot_values(raw, self.catalog_stores)
            candidates, confidence = self.resolver.resolve(
                [cleaned.get("category_phrase") or "", *soft_values],
                top_n=3,
            )
        except Exception as exc:
            latency = time.perf_counter() - started
            raise CategoryResolutionError(
                f"category resolution failed: {exc}",
                model=self.model,
                latency_seconds=latency,
                attempts=1 if self.last_call is None else int(self.last_call["attempts"]),
                cause_type=type(exc).__name__,
            ) from exc
        return parsed_turn_from_raw(
            cleaned,
            message,
            stores=self.catalog_stores,
            resolver_candidates=candidates,
            resolver_confidence=confidence,
        )
__all__ = [
    "ALLOWED_ATTRIBUTES",
    "CategoryResolutionError",
    "HARD_ATTRIBUTES",
    "MESSAGE_TYPES",
    "PROMPT",
    "ParsedSlot",
    "ParsedTurn",
    "ParserRequestError",
    "SCHEMA",
    "TurnParser",
    "TurnParserError",
    "WinstonTurnParser",
    "clean_parse",
    "hard_claim_holds",
    "model_code",
    "negation_supported",
    "parsed_turn_from_raw",
    "resolver_soft_slot_values",
    "tier_of",
    "validate_raw_parse",
]
