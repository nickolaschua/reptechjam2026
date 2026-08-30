"""Bolt-on parse for free-text turns. Sits on the No-Exact-Match edge.

The evaluator can only say seven things (local_evaluator.py:159-185). Those
return None here, so the keyword layer's own regex state is used bit-for-bit.
Anything else was typed by a person: grammar-constrained parse (nlp_parse.py),
deterministic validation (clean_parse), category candidates + confidence from
the resolver (pipeline.py), then ONE typed object - the team's FastMemoryUpdate
plus the fields it lacks - for Category, Embedder and Long-term Memory to read.

    python3 bolt_on.py "need plain tees for my husband, under 30, he hates plastic"
    python3 bolt_on.py --no-resolver "..."
"""
from __future__ import annotations

import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

LAB = Path(__file__).resolve().parent
WINSTON = LAB.parent
KIT = WINSTON.parent / "techjam-conversational-search"
for p in (WINSTON, WINSTON / "experiments", LAB, KIT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from memory.fast_memory import FastMemoryUpdate  # noqa: E402
from memory.types import ConstraintKind, TypedConstraint  # noqa: E402
from nlp_parse import parse_with_ollama, tier_of  # noqa: E402

PARSER_MODEL = "qwen2.5:7b-instruct"

# ------------------------------------------------------------ post-validation
# Same philosophy as nlp_parse.hard_claim_holds: the model proposes, code disposes.
# Lives here, not in nlp_parse.py, so the benchmark's cached parses stay raw and
# every rule below can be A/B'd offline on parses.jsonl (plan B5 is the dept rule).
_JUNK_VALUES = frozenset({"", "none", "n/a", "na", "null", "unknown", "any", "not specified",
                          "unspecified", "not mentioned", "no preference", "not stated"})
_DECLINE_CUES = ("don't care", "dont care", "doesn't matter", "doesnt matter",
                 "does not matter", "do not care", "not important", "no preference")
# The wearer has to be SAID. The 7B inferred baby-girls from "bathing suit" and
# unisex-child from "baseball cap" (probes 05, 22, 29, 30) - a hard filter on a
# guess. A department survives only if the message names a person or group.
_WEARER_RE = re.compile(
    r"\b(?:wom[ae]n|female|lad(?:y|ies)|girls?|wife|mum|mom|mother|sister|daughter|niece|"
    r"girlfriend|m[ae]n|male|guys?|husband|dad|father|brother|son|nephew|boyfriend|"
    r"boys?|kids?|child(?:ren)?|toddler|bab(?:y|ies)|infant|unisex|anyone|everyone|"
    r"his|hers?|him|she|he)\b", re.I)


# A negated flag is only trusted if a negation cue sits within a few tokens BEFORE
# the value in the message. The 7B flagged "leather" as negated in "i really like
# brown leather" and in "leather that's not too heavy" (the "not" belongs to
# "heavy"); a downstream filter then sank the target. The model proposes, the code
# disposes.
_NEG_CUES = r"(?:not|no|non|don'?t|dont|never|without|avoid|hates?|nothing|isn'?t|aren'?t|can'?t|won'?t)"
_NEG_WINDOW = 6


def negation_supported(utterance: str, value: str) -> bool:
    words = [w for w in re.split(r"[^a-z0-9']+", value.lower()) if len(w) > 2]
    if not words:
        return False
    head = re.escape(words[0])
    gap = r"(?:\s+[a-z0-9'&-]+){0," + str(_NEG_WINDOW) + r"}"
    return re.search(rf"\b{_NEG_CUES}{gap}\s+(?:[a-z0-9'&-]+\s+)?{head}", utterance.lower()) is not None


def clean_parse(pred: dict, utterance: str = "") -> dict:
    """Junk values vanish, decline cues become declines, duplicates collapse, a
    department with no named wearer is a guess -> None, and a negated flag with no
    negation cue near the value in the message is dropped. Idempotent."""
    slots, seen = [], set()
    for s in pred.get("slots") or []:
        value = str(s.get("value") or "").strip()
        low = value.lower()
        declined = bool(s.get("declined"))
        negated = bool(s.get("negated"))
        if negated and utterance:
            core = re.sub(r"^(?:not|no|non|without|never)\s+", "", value, flags=re.I)
            negated = negation_supported(utterance, core)
        if declined or any(cue in low for cue in _DECLINE_CUES):
            declined, value, low = True, "", ""
        elif low in _JUNK_VALUES:
            continue
        key = (s.get("attribute"), low, declined)
        if key in seen:
            continue
        seen.add(key)
        slots.append({"attribute": s.get("attribute"), "value": value,
                      "declined": declined, "negated": negated})
    out = {**pred, "slots": slots}
    if utterance and not _WEARER_RE.search(utterance):
        out["department"] = None
    return out


def to_flat_state(parse: dict) -> tuple[str, list[str]]:
    """The keyword layer's own {category, constraints[]} - bench/score.parsed_state,
    so the bolt-on ships exactly what `parsed_rank` measured."""
    sys.path.insert(0, str(LAB / "bench"))
    from score import parsed_state
    return parsed_state(parse)

# ---------------------------------------------------------------- the fall-through
# The evaluator's initial form is `I'm looking for {coarse category}` + one of
# three tails, and {coarse category} is a catalog bucket - the last two path
# segments, verbatim (exp07: 200/200 exact lookup). The generators write
# "i'm looking for a new strap for my watch. i really like..." for almost every
# case, so a SHAPE test would swallow the benchmark; the bucket lookup is the
# only predicate that separates the simulator from a person.
_INITIAL_RE = re.compile(
    r"^i['’]m looking for (?P<cat>[^.,]{1,80}?)"
    r"(?:, but i['’]m still exploring\.?|\. a key requirement is: .+|\. \S.*)$", re.I | re.S)
_OTHER_RES = tuple(re.compile(pat, re.I | re.S) for pat in (
    r"^for that, what matters is: ",
    r"^actually, (?:please )?ignore my earlier preference",
    r"^i don['’]t have (?:a|an additional) preference for ",
    r"^those options are not quite right yet",
))
_BUCKETS: frozenset[str] | None = None


def bucket_set() -> frozenset[str]:
    """Every coarse category the evaluator can put in an initial message."""
    global _BUCKETS
    if _BUCKETS is None:
        cache = LAB / ".cache" / "buckets.json"
        if cache.exists():
            _BUCKETS = frozenset(json.loads(cache.read_text()))
        else:
            from evaluator.local_evaluator import catalog_index, coarse_category
            _, cats, _ = catalog_index(KIT / "data" / "catalog.jsonl")
            _BUCKETS = frozenset(coarse_category(v).lower() for v in cats.values())
            cache.parent.mkdir(exist_ok=True)
            cache.write_text(json.dumps(sorted(_BUCKETS)))
    return _BUCKETS


def is_template(message: str) -> bool:
    text = message.strip()
    m = _INITIAL_RE.match(text)
    if m:
        return m.group("cat").strip().lower() in bucket_set()
    return any(r.match(text) for r in _OTHER_RES)


# ------------------------------------------------------------------ derived labels
# same regex as lab/bench/covariates.py:MODEL_CODE_RE - a code must never pass
# through an LLM, embeddings and small models both mangle alphanumerics
_MODEL_CODE_RE = re.compile(r"\b([A-Za-z]{1,4}-?\d{3,}[A-Za-z0-9-]*)\b")
_GRADES = frozenset({"316L", "925", "S925", "14K", "18K", "10K", "585", "750", "24K", "9K", "UV400"})
_EXPLORING_RE = re.compile(
    r"\b(?:just browsing|still exploring|not sure (?:yet|what)|open to (?:ideas|suggestions|anything)|"
    r"any (?:ideas|suggestions|recommendations)|no idea|looking around|what do you (?:have|recommend))\b",
    re.I)
_COMPAT_RE = re.compile(
    r"\b(?:i (?:already )?own|to (?:go|fit|match) with my|(?:fits?|for) my (?:watch|bracelet|sneakers|boots|shoes|band))\b",
    re.I)
MESSAGE_TYPES = ("exact", "compatibility", "symptom", "product_type", "use_case", "feature")


def model_code(message: str) -> str | None:
    for m in _MODEL_CODE_RE.finditer(message):
        code = m.group(1)
        if code.upper() in _GRADES or code.isdigit() or not re.search(r"[A-Za-z]", code):
            continue
        return code.upper()
    return None


def n_hard(parse: dict) -> int:
    return (sum(1 for s in parse.get("slots", []) if tier_of(s) == "hard")
            + bool(parse.get("price_max")) + bool(parse.get("price_min")))


def intent_of(parse: dict, message: str) -> str:
    """buying = the user fixed something the catalog can filter on; else browsing.

    ponytail: the LLM's own `exploring` bool was true on 6/30 probes and wrong
    on half - specificity is a count, not a vibe. An explicit cue still wins.
    """
    if _EXPLORING_RE.search(message):
        return "browsing"
    return "buying" if n_hard(parse) or model_code(message) else "browsing"


def message_type_of(parse: dict, message: str) -> str:
    """v0 rule over the parse; the TF-IDF+LR classifier has to beat this."""
    if model_code(message):
        return "exact"
    if _COMPAT_RE.search(message):
        return "compatibility"
    if not (parse.get("category_phrase") or "").strip():
        return "symptom"
    slots = [s for s in parse.get("slots", []) if not s.get("declined")]
    if not slots and not parse.get("price_max") and not parse.get("price_min"):
        return "product_type"
    if {s["attribute"] for s in slots} <= {"use_case"}:
        return "use_case"
    return "feature"


# ------------------------------------------------------------------ the one object
_KIND = {k.value: k for k in ConstraintKind}


@dataclass(frozen=True)
class ParsedTurn(FastMemoryUpdate):
    """FastMemoryUpdate + what it lacks. isinstance(..., FastMemoryUpdate) holds,
    so update_state() consumes it unchanged; the extras are for Category/Embedder."""
    department: str | None = None
    specificity: str | None = None       # the parser's own enum (plan A3), for comparison
    message_type: str = "feature"
    model_code: str | None = None
    category_candidates: tuple[str, ...] = ()
    parse: dict | None = None            # the validated raw parse, for scoring

    def to_json(self) -> str:
        d = asdict(self)
        for k in ("hard_constraints", "soft_preferences", "negatives"):
            d[k] = [{"value": c.value, "kind": c.kind.value, "negated": c.negated} for c in getattr(self, k)]
        return json.dumps(d, indent=2)


def _tc(slot: dict, turn: int, *, hard: bool, negated: bool = False) -> TypedConstraint:
    return TypedConstraint(
        value=slot["value"], kind=_KIND.get(slot["attribute"], ConstraintKind.FEATURE),
        hard=hard, negated=negated, explicit=True,
        strength=1.0 if hard or negated else 0.75, confidence=1.0 if hard or negated else 0.85,
        source_turn=turn, source="bolt_on", intent_epoch=0)


def to_update(parse: dict, message: str, turn: int, candidates: tuple[str, ...] = (),
              confidence: float | None = None) -> ParsedTurn:
    parse = clean_parse(parse, message)
    hard, soft, neg = [], [], []
    priced = bool(parse.get("price_max") or parse.get("price_min"))
    for s in parse.get("slots", []):
        tier = tier_of(s)
        if tier == "decline":
            continue
        if s["attribute"] == "budget" and priced:
            continue                  # the normalised "under $N" below replaces the slot text
        if s.get("negated"):          # a flag, not a tier: nlp_parse.tier_of ignores it
            # the 7B keeps the "not" inside the value even when it sets the flag
            s = {**s, "value": re.sub(r"^(?:not|no|non|without|never)\s+", "", s["value"], flags=re.I)}
            neg.append(_tc(s, turn, hard=True, negated=True))
        elif tier == "hard":
            hard.append(_tc(s, turn, hard=True))
        else:
            soft.append(_tc(s, turn, hard=False))
    for key, word in (("price_max", "under"), ("price_min", "over")):
        v = parse.get(key)
        if v:   # 0 is the model's "no bound"
            hard.append(TypedConstraint(value=f"{word} ${float(v):g}", kind=ConstraintKind.BUDGET,
                                        hard=True, source_turn=turn, source="bolt_on"))
    return ParsedTurn(
        category=(parse.get("category_phrase") or "").strip() or None,
        intent=intent_of(parse, message),
        hard_constraints=tuple(hard), soft_preferences=tuple(soft), negatives=tuple(neg),
        topic_override=False, replace_constraints=False, confidence=confidence,
        department=parse.get("department"), specificity=parse.get("specificity"),
        message_type=message_type_of(parse, message),
        model_code=model_code(message), category_candidates=tuple(candidates), parse=parse)


class BoltOnParser:
    """memory.fast_memory.SemanticParser: None on template turns, ParsedTurn otherwise."""

    def __init__(self, model: str = PARSER_MODEL, host: str = "http://localhost:11434",
                 parse_fn=None, resolver: bool = True) -> None:
        self.model, self.host = model, host
        self._parse_fn = parse_fn                      # injectable: tests, cached parses
        self._resolver = resolver
        self._ix = None

    def _candidates(self, parse: dict) -> tuple[tuple[str, ...], float | None]:
        if not self._resolver:
            return (), None
        if self._ix is None:
            from common import get_index
            from pipeline import content_profiles, resolve, slot_terms
            self._ix, self._resolve, self._slot_terms = get_index(), resolve, slot_terms
            self._profiles = content_profiles(self._ix)
        terms = [parse.get("category_phrase") or "", *self._slot_terms(parse)]
        ranked, conf = self._resolve(terms, self._ix, self._profiles, top_n=3)
        return tuple(ranked), round(conf, 3)

    def parse(self, message: str, turn: int) -> ParsedTurn | None:
        if is_template(message):
            return None
        raw = self._parse_fn(message) if self._parse_fn else parse_with_ollama(message, self.model, self.host)
        parse = clean_parse(raw, message)
        candidates, confidence = self._candidates(parse)
        return to_update(parse, message, turn, candidates, confidence)


# ------------------------------------------- hard constraints, after retrieval
# Three-valued: a product either MATCHES a hard slot, CONTRADICTS it, or is SILENT.
# Only a contradiction counts. 28 % of targets say nothing about material and 79 %
# have no price - a filter that excluded on silence would drop the target more
# often than a wrong product.
from nlp_parse import _MATERIALS, hard_claim_holds, normalize_department  # noqa: E402

_PRICE_SLACK = 1.10          # "like 30 bucks" tolerates $32.99
_UNISEX = {"unisex-adult", "unisex-child", None}


def _mentions(text: str, value: str) -> bool:
    words = [w for w in re.split(r"[^a-z0-9]+", value.lower()) if len(w) > 2]
    return bool(words) and all(re.search(rf"\b{re.escape(w)}s?\b", text) for w in words)


def contradictions(parse: dict, product: dict, text: str) -> list[str]:
    """Which hard constraints / negatives does this product CONTRADICT? Empty = keep.

    `text` is the product's full lowercase text (title, features, details,
    description). Returns reasons, so a re-ranker can log why something sank."""
    out: list[str] = []
    price = product.get("price")
    try:
        price = float(price) if price not in (None, "") else None
    except (TypeError, ValueError):
        price = None
    if price is not None:
        if parse.get("price_max") and price > float(parse["price_max"]) * _PRICE_SLACK:
            out.append(f"price {price} > max {parse['price_max']}")
        if parse.get("price_min") and price < float(parse["price_min"]) / _PRICE_SLACK:
            out.append(f"price {price} < min {parse['price_min']}")
    dept = parse.get("department")
    pdept = normalize_department((product.get("details") or {}).get("Department"))
    if dept and pdept and dept != pdept and dept not in _UNISEX and pdept not in _UNISEX:
        out.append(f"department {pdept} != {dept}")
    store = str(product.get("store") or "").strip().lower()
    product_materials = {m for m in _MATERIALS if re.search(rf"\b{m}\b", text)}
    for s in parse.get("slots", []):
        value = str(s.get("value") or "").strip()
        if not value or s.get("declined"):
            continue
        if s.get("negated"):
            core = re.sub(r"^(?:not|no|non|without|never)\s+", "", value, flags=re.I)
            if _mentions(text, core):
                out.append(f"negated '{core}' present")
            continue
        if tier_of(s) != "hard" or not hard_claim_holds(s["attribute"], value):
            continue
        if s["attribute"] == "brand" and store and value.lower() not in store and store not in value.lower():
            out.append(f"store '{store}' != brand '{value}'")
        elif s["attribute"] == "material":
            wanted = {w for w in re.split(r"[^a-z]+", value.lower()) if w in _MATERIALS}
            if product_materials and wanted and not (wanted & product_materials):
                out.append(f"material {sorted(product_materials)} != {sorted(wanted)}")
        # ponytail: size is not checked - the catalog's size text is too irregular to
        # call a contradiction safely; it stays positive evidence only
    return out


# ------------------------------------------------ attach to the keyword layer as-is
def make_agent(catalog_path, parser: BoltOnParser | None = None):
    """starter.agent.Agent with the bolt-on on the fall-through. Zero edits there:
    super() runs the template regexes; only a message they ignored reaches the parser."""
    from starter.agent import Agent

    parser = parser or BoltOnParser()

    class BoltOnAgent(Agent):
        def _update_state(self, state, user_message, turn):
            super()._update_state(state, user_message, turn)
            if is_template(user_message):
                return
            update = parser.parse(user_message, turn)
            if update is None:
                return
            category, constraints = to_flat_state(update.parse)
            if not state["category"]:                        # ponytail: topic shift on turn>1 is v2
                state["category"] = category
            for value in constraints:                        # negated/declined already dropped
                self._append_constraint(state, value)
            state["bolt_on"] = update

    return BoltOnAgent(catalog_path)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not args:
        raise SystemExit(__doc__)
    bp = BoltOnParser(resolver="--no-resolver" not in sys.argv)
    for msg in args:
        u = bp.parse(msg, 1)
        print(f"\n> {msg}\n" + ("template -> keyword layer" if u is None else u.to_json()))
