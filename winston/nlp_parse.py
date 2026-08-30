"""Structured intent parse for the shop agent, sized for a local LLM.

The reliability mechanism is the SCHEMA, not the prompt: Ollama takes a JSON
schema as `format=` and constrains decoding to it, so an invalid shape is
unrepresentable rather than merely discouraged. Everything here is therefore
built to push work out of the model's judgement and into the grammar.

Four design rules, each one a thing that goes wrong with a 7B otherwise:

  1. ONE flat slot list, not hard/soft/declined containers. The probe-set YAML
     splits a constraint across three keys, which forces the model to make a
     routing decision AND an extraction decision per span. A single list with a
     `stance` enum makes it one decision, and the enum is grammar-enforced.
  2. EVERY key required, null as a value. Optional keys let a small model omit
     the hard cases; a required key with an explicit null makes "I didn't find
     one" an answer it has to actually give.
  3. CLOSED enums wherever the value space is knowable. `attribute` is exactly
     the evaluator's ALLOWED_ATTRIBUTES, so a slot feeds `ask_attribute`
     straight through with no mapping layer.
  4. NO verbatim span extraction. The YAML's `discard` asks the model to quote
     input back; small models paraphrase and drift. A discarded span is simply
     a slot that never gets created, and the spurious-slot rate measures it.

`category` stays a free phrase on purpose. Picking among 1,115 catalog buckets
is not the model's job - EXP06/EXP07 show a lexical resolver plus an encoder
fallback does it better, and a 1,115-value enum would bloat the grammar.

    python3 nlp_parse.py              # self-check + gold conversion report
    python3 nlp_parse.py --model qwen2.5:7b-instruct   # also run Ollama
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROBE_SET = HERE / "probe_set.md"
CATALOG = HERE.parent / "techjam-conversational-search" / "data" / "catalog.jsonl"
CACHE = HERE / ".cache"

# The evaluator's enum, verbatim - a slot's `attribute` IS a legal ask_attribute.
ALLOWED_ATTRIBUTES = [
    "category", "material", "color", "size", "style", "brand",
    "budget", "feature", "use_case", "other",
]
# The catalog's details.Department has 136 spellings over 43,582 products; these
# eight cover 97.7% of them. The model picks from the clean set, and
# normalize_department() absorbs the mess on the filter side - a small model
# should never have to learn that "女士", "ladies" and "women's" are one value.
DEPARTMENTS = ["womens", "mens", "girls", "boys",
               "baby-girls", "baby-boys", "unisex-adult", "unisex-child", None]
# probe_set.md's tier rule: HARD = brand/category/department/material/size/price,
# SOFT = colour, use-case, season, vibe, comfort/fit/durability. Tier is a
# function of the ATTRIBUTE, so it is a lookup, not a model decision - asking a
# 7B to also emit it just gave it a way to disagree with itself.
#
# "category" is deliberately absent: the category lives in `category_phrase`, and
# a category slot is pure duplication. The model emitted three of them on case 01
# alone ("comfortable shoe", "grippy shoe"), each one a soft property glued to the
# category noun. Removing it from the enum makes that unrepresentable.
HARD_ATTRIBUTES = {"brand", "material", "size", "budget"}

# A hard slot FILTERS the catalog. Claiming hard tier for a value the catalog
# cannot express is how a parser destroys recall, so the claim has to be checked
# against the catalog rather than trusted. The model said material:"waterproof",
# size:"long sleeve", brand:"UPF 50" - none of which is the thing it claims to be.
_MATERIALS = frozenset({
    "cotton", "polyester", "nylon", "leather", "wool", "spandex", "silk", "rayon",
    "denim", "suede", "canvas", "mesh", "fleece", "cashmere", "linen", "velvet",
    "satin", "acrylic", "viscose", "elastane", "rubber", "metal", "plastic",
    "gold", "silver", "alloy", "stainless", "sterling", "crystal", "gemstone",
    "wood", "bamboo", "microfiber", "chiffon", "lace", "jersey", "corduroy",
    "tweed", "synthetic", "fabric", "pearl", "diamond", "brass", "copper",
    "titanium", "nickel", "zinc", "resin", "felt", "flannel", "sequin",
})
# "real gold" is gold; "soft leather-looking" is not leather. Modifiers are
# stripped, then EVERY remaining token must be a substance.
_MODIFIERS = frozenset({"real", "genuine", "faux", "natural", "pure", "solid",
                        "blend", "blended", "soft", "fine"})
_NEGATIONS = frozenset({"not", "without", "no", "non"})
_SIZE_WORDS = frozenset({
    "plus", "petite", "tall", "big", "wide", "narrow", "regular", "slim",
    "toddler", "kids", "junior", "infant", "youth", "adult", "short", "long",
    "xs", "s", "m", "l", "xl", "xxl", "xxxl", "small", "medium", "large",
    "size", "width",
})
_SIZE_NUMERIC = re.compile(r"^\d{1,2}(\.\d)?[a-z]{0,3}$|^\d{2}x\d{2}$")

_STORES: frozenset[str] | None = None


def catalog_stores() -> frozenset[str]:
    """Every brand the catalog actually sells - 19,749 of them, 99.4% coverage.

    A brand filter on a value no store matches removes everything, so this is the
    only honest test of whether a brand slot is real.
    """
    global _STORES
    if _STORES is None:
        cached = CACHE / "stores.json"
        if cached.exists():
            _STORES = frozenset(json.loads(cached.read_text()))
        else:
            names = set()
            with CATALOG.open(encoding="utf-8") as fh:
                for line in fh:
                    store = json.loads(line).get("store")
                    if store:
                        names.add(str(store).strip().lower())
            CACHE.mkdir(exist_ok=True)
            cached.write_text(json.dumps(sorted(names)))
            _STORES = frozenset(names)
    return _STORES


def hard_claim_holds(attribute: str, value: str) -> bool:
    """Can the catalog actually filter on this? If not, the slot is soft."""
    tokens = [t for t in re.split(r"[^a-z0-9]+", value.lower()) if t]
    if not tokens or _NEGATIONS & set(tokens):
        return False
    core = [t for t in tokens if t not in _MODIFIERS]
    if not core:
        return False
    if attribute == "brand":
        stores = catalog_stores()
        return value.strip().lower() in stores or " ".join(core) in stores
    if attribute == "budget":
        return any(c.isdigit() for c in value)
    if attribute == "size":
        return len(core) <= 2 and all(
            t in _SIZE_WORDS or _SIZE_NUMERIC.match(t) for t in core)
    if attribute == "material":
        return len(core) <= 2 and all(t in _MATERIALS for t in core)
    return True


def tier_of(slot: dict) -> str:
    """hard = filterable (three-valued). soft = scoreable. decline = user opted out.

    A slot only reaches hard tier if the catalog can verify the claim; otherwise
    it is demoted, never dropped - the VALUE is usually right and the ranker can
    still use it, it is only the type that was wrong.
    """
    if slot.get("declined"):
        return "decline"
    if slot["attribute"] not in HARD_ATTRIBUTES:
        return "soft"
    return "hard" if hard_claim_holds(slot["attribute"], slot["value"]) else "soft"
QUALITY_PRIORS = ["none", "well_rated", "reputable_brand"]

# Passed to Ollama as `format=`; decoding is constrained to it.
SCHEMA = {
    "type": "object",
    "properties": {
        "category_phrase": {
            "type": "string",
            "maxLength": 40,
            "description": "The product type in the user's own words, 1-4 words. Never a brand.",
        },
        "department": {
            "type": ["string", "null"],
            "enum": DEPARTMENTS,
            "description": "null when the user does not say who it is for - that is "
                           "the normal answer. Use unisex-adult or unisex-child ONLY "
                           "if the user actually says unisex, or says it is for anyone. "
                           "Never infer from the product type.",
        },
        "slots": {
            "type": "array",
            "maxItems": 8,
            "items": {
                "type": "object",
                "properties": {
                    # ALLOWED_ATTRIBUTES minus "category": the category is already
                    # `category_phrase`, and a category slot only ever restated it.
                    "attribute": {"type": "string",
                                  "enum": [a for a in ALLOWED_ATTRIBUTES if a != "category"]},
                    # maxLength is load-bearing, not tidiness. A JSON schema
                    # constrains STRUCTURE, not LENGTH: an unbounded "string" is an
                    # unbounded generation, and the grammar has no way to stop a
                    # degenerate loop inside one. Case 09 emitted "not too preppy,"
                    # ~90 times and blew past a 300s timeout twice before this cap.
                    "value": {"type": "string", "maxLength": 40},
                    "declined": {
                        "type": "boolean",
                        "description": "true ONLY if the user said they do not care "
                                       "about this attribute. Otherwise false.",
                    },
                    # The team's TypedConstraint has `negated`; without it "not
                    # leather" becomes a positive leather filter. Top failure mode.
                    "negated": {
                        "type": "boolean",
                        "description": "true ONLY if the user said they do NOT want "
                                       "this value. Otherwise false.",
                    },
                },
                "required": ["attribute", "value", "declined", "negated"],
            },
        },
        "price_max": {"type": ["number", "null"]},
        "price_min": {"type": ["number", "null"]},
        "quality_prior": {
            "type": "string",
            "enum": QUALITY_PRIORS,
            "description": "Almost always 'none'. Use well_rated ONLY if the user "
                           "literally asks about ratings, reviews or popularity, and "
                           "reputable_brand ONLY if they ask for a trusted or known "
                           "make without naming one. Wanting a good product is not a "
                           "request for either. A named brand is a brand slot instead.",
        },
        "exploring": {
            "type": "boolean",
            "description": "true if browsing with no fixed target, false if buying.",
        },
        # Same call as the parse, so it costs nothing. Tested against n_hard and
        # card_hard_said on the benchmark; keep only if it beats them.
        # A 0-1 number came back 1.0 on all 30 probes; a 7B cannot calibrate a
        # rubric. Enum: constrained decoding picks a label instead.
        "specificity": {
            "type": "string",
            "enum": ["scenario_only", "type_with_wishes", "type_with_requirements"],
            "description": "scenario_only: a situation or feeling, no product type named. "
                           "type_with_wishes: a product type plus soft preferences. "
                           "type_with_requirements: a product type plus firm requirements "
                           "such as size, brand, material or a price limit.",
        },
    },
    "required": ["category_phrase", "department", "slots",
                 "price_max", "price_min", "quality_prior", "exploring", "specificity"],
}

PROMPT = """Extract shopping constraints from the message. Output only what the user said.

Rules:
- department: only if stated. "women's shoes" -> womens. A dress does NOT imply womens.
- Do not create a slot for a self-correction, an aside, or a question the user
  answered themselves. If the user drops an idea mid-sentence, it is not a constraint.
- quality_prior is "none" unless the user actually mentions ratings, reviews,
  popularity, or wanting a reputable/trusted brand. Default to "none".
- declined is true only when the user says they do not care about something.
- negated is true only when the user says they do NOT want that value ("no logos",
  "not leather"). Still record the value; mark it negated.
- specificity: scenario_only if no product type is named, type_with_wishes for a
  product type plus soft preferences, type_with_requirements only when there are
  firm requirements (size, brand, material, price limit).

Message: {utterance}"""


# --------------------------------------------------------------------------- gold

_YAML_BLOCK = re.compile(r"```yaml\n(.*?)```", re.S)
_HEADING = re.compile(r"^## (\d+)\. `([A-Z0-9]+)` — stratum ([A-G])", re.M)

_DEPT_CANON = {
    "women": "womens", "woman": "womens", "womens": "womens", "women's": "womens",
    "women\u2019s": "womens", "ladies": "womens", "female": "womens", "daughter": "womens",
    "men": "mens", "mens": "mens", "men's": "mens", "male": "mens", "adult-male": "mens",
    "girls": "girls", "girl": "girls", "teen-girls": "girls",
    "boys": "boys", "boy": "boys", "teen-boys": "boys",
    "baby-girls": "baby-girls", "baby girls": "baby-girls",
    "baby-boys": "baby-boys", "baby boys": "baby-boys",
    "unisex": "unisex-adult", "unisex-adult": "unisex-adult",
    "unisex adult": "unisex-adult", "unisex-adults": "unisex-adult",
    "unisex-child": "unisex-child", "unisex child": "unisex-child",
    "unisex-baby": "unisex-child", "unisex baby": "unisex-child",
    "unisex-kids": "unisex-child", "unisex-youth": "unisex-child", "kids": "unisex-child",
}


def normalize_department(raw: object) -> str | None:
    """Map any catalog Department spelling onto the schema enum, else None.

    None means "this product is SILENT on department" - which is not the same as
    contradicting the user, and the filter has to keep treating it three-valued.
    """
    if raw in (None, ""):
        return None
    text = str(raw).strip().lower()
    if text in _DEPT_CANON:
        return _DEPT_CANON[text]
    text = re.split(r"[;(]", text)[0].strip()      # "teen-boys;mens", "unisex-adult (luggage only)"
    return _DEPT_CANON.get(text)


_DEPT_ALIASES = _DEPT_CANON
# probe-set soft tags / declined names -> the evaluator's attribute enum
_ATTR_ALIASES = {"colour": "color", "price": "budget", "brand": "brand",
                 "material": "material", "size": "size", "department": "other"}


def load_gold() -> tuple[list[dict], list[str]]:
    """Convert probe_set.md's expected_parse blocks into SCHEMA shape.

    Returns (cases, warnings). Warnings name every value that did not map
    cleanly - they are the human-review list, not silent coercions.
    """
    import yaml

    text = PROBE_SET.read_text(encoding="utf-8")
    heads = _HEADING.findall(text)
    blocks = _YAML_BLOCK.findall(text)
    if len(heads) != len(blocks):
        raise SystemExit(f"probe_set.md: {len(heads)} headings vs {len(blocks)} yaml blocks")

    cases, warn = [], []
    for (num, asin, stratum), raw in zip(heads, blocks):
        doc = yaml.safe_load(raw)
        exp = doc.get("expected_parse") or {}
        hard = exp.get("hard") or {}
        tag = f"case {num} ({asin})"

        dept_raw = hard.get("department")
        dept = None
        if dept_raw is not None:
            dept = _DEPT_ALIASES.get(str(dept_raw).strip().lower())
            if dept is None:
                warn.append(f"{tag}: department {dept_raw!r} has no enum mapping")

        slots, price_max, price_min = [], None, None
        for key, value in hard.items():
            if key in ("category", "department") or value is None:
                continue
            if key == "price":
                if isinstance(value, dict):
                    price_max, price_min = value.get("max"), value.get("min")
                else:
                    warn.append(f"{tag}: price {value!r} is not a {{min,max}} map")
                continue
            attr = _ATTR_ALIASES.get(key, key)
            if attr not in ALLOWED_ATTRIBUTES:
                warn.append(f"{tag}: hard slot {key!r} is outside ALLOWED_ATTRIBUTES")
                attr = "other"
            slots.append({"attribute": attr, "value": str(value).lower(), "declined": False})

        for value in exp.get("soft") or []:
            slots.append({"attribute": "feature", "value": str(value).lower(),
                          "declined": False})
        for value in exp.get("declined") or []:
            attr = _ATTR_ALIASES.get(str(value).strip().lower(), str(value).strip().lower())
            if attr not in ALLOWED_ATTRIBUTES:
                warn.append(f"{tag}: declined {value!r} is outside ALLOWED_ATTRIBUTES")
                attr = "other"
            slots.append({"attribute": attr, "value": "", "declined": True})

        quality = exp.get("quality") or {}
        prior = "none"
        if quality.get("well_rated"):
            prior = "well_rated"
        if str(quality.get("brand_reputation", "")).lower() == "high":
            prior = "reputable_brand"

        cases.append({
            "case": int(num), "asin": asin, "stratum": stratum,
            "utterance": doc["utterance"],
            "discard_spans": exp.get("discard") or [],   # kept for scoring only
            "gold": {
                "category_phrase": str(hard.get("category") or ""),
                "department": dept,
                "slots": slots,
                "price_max": price_max, "price_min": price_min,
                "quality_prior": prior,
                "exploring": False,
                "specificity": None,      # not graded; no gold for it
            },
        })
    return cases, warn


# -------------------------------------------------------------------------- score

_STOPWORDS = {"a", "an", "the", "so", "i", "it", "is", "of", "for", "to", "and",
              "my", "me", "in", "on", "at", "or", "if", "do", "you", "know",
              "guess", "idk", "hmm", "really", "quite", "whatever", "want",
              "dont", "don", "t", "that", "this", "them", "with", "what", "not"}


def _price(value: object) -> float | None:
    """0 and null both mean "no bound". The model emits 0 for an absent minimum on
    every priced case; grading that as a miss measures JSON style, not extraction."""
    return None if value in (None, 0, 0.0) else float(value)


def _norm(text: str) -> set[str]:
    return {w for w in re.split(r"[^a-z0-9]+", text.lower()) if w}


def score(pred: dict, gold: dict, discard_spans: list[str] | None = None,
          utterance: str = "") -> dict:
    """Slot-level agreement. A slot matches on (attribute, stance) plus token overlap.

    `discard_spans` are the asides the probe set says a parser must NOT turn into
    constraints. The schema has no `discard` field on purpose - a discarded span
    is simply a slot that never got created - so this is where that gets measured:
    a spurious slot drawn from a discard span is the specific failure of inventing
    a constraint out of an aside, and it is worth counting separately from an
    ordinary false positive.
    """
    def key(slot: dict) -> tuple:
        # Grade the TIER, not the label. The ablation that motivated this: grading
        # `attribute` exactly scored F1 0.084 while the same predictions scored
        # 0.397 on value alone - because the gold puts every soft tag in "feature"
        # (a converter artifact) and the model spreads them over material/style/
        # use_case, all defensible. Tier is what the agent acts on: hard slots
        # filter, soft slots score. Getting THAT wrong is a real error, and this
        # still catches it.
        return (tier_of(slot),)

    matched, used_gold, used = 0, set(), set()
    for j, p in enumerate(pred["slots"]):
        for i, g in enumerate(gold["slots"]):
            if i in used_gold or key(p) != key(g):
                continue
            if not g["value"] or _norm(p["value"]) & _norm(g["value"]):
                matched += 1
                used_gold.add(i)
                used.add(j)
                break
    leaked = 0
    if discard_spans:
        # Words the aside shares with the rest of the message are not evidence of a
        # leak: "does the material of the shoe matter?" shares "shoe" with the real
        # category, and flagging that scored 7 false leaks on one case. Only words
        # UNIQUE to the aside can implicate a slot.
        span_words = set().union(*(_norm(sp) for sp in discard_spans)) - _STOPWORDS
        # Cut the span TEXT out and re-tokenize the remainder. Subtracting token
        # SETS instead loses the count: "shoe" occurs in both the aside and the
        # real query, so set-subtraction deleted it from the remainder and made
        # every slot containing "shoe" look like a leak - 6 false positives on
        # case 01 alone.
        remainder = (utterance or "").lower()
        for sp in discard_spans:
            remainder = remainder.replace(sp.lower(), " ")
        discard_words = span_words - _norm(remainder)
        for j, p in enumerate(pred["slots"]):
            # A DECLINED slot is supposed to name the thing the user waved off;
            # quoting the aside there is correct behaviour, not a leak.
            if j in used or p.get("declined"):
                continue
            if _norm(p["value"]) & discard_words:
                leaked += 1

    # 46% of the probe set's soft tags never appear in the utterance ("quick-dry",
    # "flattering", "hooded") - they are query EXPANSION targets Winston wrote, not
    # extraction targets. Grading recall against them caps any extractor at 0.544,
    # so headline recall uses the extractable subset and the full number is kept
    # beside it rather than quietly averaged in.
    utterance_words = _norm(utterance or "")
    extractable = [g for g in gold["slots"]
                   if not g["value"] or not utterance_words
                   or (_norm(g["value"]) & utterance_words)]
    matched_ext = 0
    used_gold_ext = set()
    for p in pred["slots"]:
        for i, g in enumerate(extractable):
            if i in used_gold_ext or key(p) != key(g):
                continue
            if not g["value"] or _norm(p["value"]) & _norm(g["value"]):
                matched_ext += 1
                used_gold_ext.add(i)
                break

    n_pred, n_gold = len(pred["slots"]), len(gold["slots"])
    precision = matched_ext / n_pred if n_pred else 1.0
    recall = matched_ext / len(extractable) if extractable else 1.0
    return {
        "department_ok": pred["department"] == gold["department"],
        "category_overlap": bool(_norm(pred["category_phrase"]) & _norm(gold["category_phrase"])),
        "price_ok": (_price(pred["price_max"]) == _price(gold["price_max"])
                     and _price(pred["price_min"]) == _price(gold["price_min"])),
        "quality_ok": pred["quality_prior"] == gold["quality_prior"],
        "slot_precision": round(precision, 3),
        "slot_recall": round(recall, 3),
        "slot_f1": round(2 * precision * recall / (precision + recall), 3) if matched_ext else 0.0,
        "recall_incl_inferred": round(matched / n_gold, 3) if n_gold else 1.0,
        "gold_extractable": len(extractable),
        "gold_total": n_gold,
        "spurious_slots": n_pred - matched,
        "discard_leaks": leaked,
    }


# ------------------------------------------------------------------------- ollama

def parse_with_ollama(utterance: str, model: str, host: str = "http://localhost:11434",
                      timeout: int = 300, retries: int = 1) -> dict:
    """One constrained-decode call, retried once on transport failure.

    Raises if it still fails - a parse that silently returns empty would look
    like a model that extracted nothing, which is a different bug entirely.
    """
    import urllib.request

    body = json.dumps({
        "model": model,
        "prompt": PROMPT.format(utterance=utterance),
        "format": SCHEMA,
        "stream": False,
        # A capped decode is the difference between a slow case and a hung one:
        # case 09 ("classy bold and provocative... show off my figure") ran past
        # 300s twice, which is what an unbounded grammar-constrained decode looks
        # like when the model would rather not answer. The schema is ~360 tokens;
        # 512 is ample for any legitimate parse.
        "options": {"temperature": 0, "num_predict": 512},
    }).encode()
    last: Exception | None = None
    for _ in range(retries + 1):
        try:
            request = urllib.request.Request(f"{host}/api/generate", body,
                                             {"Content-Type": "application/json"})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(json.loads(response.read())["response"])
        except Exception as exc:                       # noqa: BLE001 - retried, then raised
            last = exc
    raise RuntimeError(f"ollama call failed after {retries + 1} attempts: {last}")


# --------------------------------------------------------------------------- main

def self_check() -> None:
    """The schema and the scorer both have to survive their own edge cases."""
    def slot(attribute, value, declined=False):
        return {"attribute": attribute, "value": value, "declined": declined}

    empty = {"category_phrase": "", "department": None, "slots": [],
             "price_max": None, "price_min": None, "quality_prior": "none",
             "exploring": False}
    assert score(empty, empty)["slot_f1"] == 0.0
    assert score(empty, empty)["slot_precision"] == 1.0     # nothing predicted, nothing wrong

    # tier is a lookup on attribute, and decline outranks it
    assert tier_of(slot("material", "leather")) == "hard"
    assert tier_of(slot("style", "sexy")) == "soft"
    assert tier_of(slot("material", "", declined=True)) == "decline"

    utt = "mens tennis shoe with cushioning, don't care about material"
    gold = {**empty, "category_phrase": "tennis-shoes", "department": "mens",
            "slots": [slot("feature", "cushioning"), slot("material", "", True)]}
    exact = score(gold, gold, None, utt)
    assert exact["slot_f1"] == 1.0 and exact["department_ok"] and exact["category_overlap"]

    # a declined slot has an empty gold value: it matches on tier alone
    pred = {**gold, "slots": [slot("material", "anything", True)]}
    assert score(pred, gold, None, utt)["slot_recall"] == 0.5

    # THE FIX: a soft tag labelled use_case instead of feature is no longer punished,
    # because both derive to tier "soft" and that is what the agent acts on.
    pred = {**gold, "slots": [slot("use_case", "cushioning")]}
    assert score(pred, gold, None, utt)["spurious_slots"] == 0

    # ...and a soft tag mislabelled as a MATERIAL is now caught by the catalog
    # gate rather than punished: "cushioning" is not a substance, so it is
    # demoted to soft, where it correctly matches the gold soft tag.
    pred = {**gold, "slots": [slot("material", "cushioning")]}
    assert tier_of(pred["slots"][0]) == "soft"
    assert score(pred, gold, None, utt)["spurious_slots"] == 0

    # the gate must not be a blanket demotion - a real substance stays hard
    assert tier_of(slot("material", "leather")) == "hard"
    assert tier_of(slot("material", "denim")) == "hard"        # absent from the
    #   evaluator's own 9-item MATERIALS tuple, but a real fabric the catalog sells
    assert tier_of(slot("material", "soft leather-looking")) == "soft"   # contains
    #   "leather" but is not leather - substring matching would have passed this
    assert tier_of(slot("material", "plated, not real gold")) == "soft"  # negation
    assert tier_of(slot("size", "wide-width")) == "hard"
    assert tier_of(slot("size", "long sleeve")) == "soft"
    assert tier_of(slot("brand", "UPF 50")) == "soft"          # matches no store
    assert tier_of(slot("budget", "under 30")) == "hard"
    assert "category" not in {a for a in SCHEMA["properties"]["slots"]["items"]
                              ["properties"]["attribute"]["enum"]}

    # an inferred gold tag (never in the utterance) must not count against recall
    infer = {**gold, "slots": [slot("feature", "cushioning"), slot("feature", "quick-dry")]}
    got = score({**empty, "slots": [slot("feature", "cushioning")]}, infer, None, utt)
    assert got["gold_extractable"] == 1 and got["gold_total"] == 2
    assert got["slot_recall"] == 1.0 and got["recall_incl_inferred"] == 0.5

    # an aside must not become a constraint, and must be countable when it does
    aside = ["so i don't slip on the court"]
    assert score(gold, gold, aside, utt)["discard_leaks"] == 0
    leaky = {**gold, "slots": [*gold["slots"], slot("feature", "slip resistant court")]}
    assert score(leaky, gold, aside, utt)["discard_leaks"] == 1
    # a word the aside SHARES with the real query is not evidence of a leak
    shared = "mens tennis shoe, does the shoe material matter? cushioning please"
    assert score({**empty, "slots": [slot("category", "tennis shoe")]},
                 empty, ["does the shoe material matter?"], shared)["discard_leaks"] == 0
    assert score(leaky, gold, aside, utt)["spurious_slots"] == 1
    assert score(leaky, gold, ["i want the thing"], utt)["discard_leaks"] == 0

    # wrong department must not be forgiven by a null
    assert not score({**gold, "department": None}, gold, None, utt)["department_ok"]

    # every free-text field must be length-bounded or the grammar cannot halt
    props = SCHEMA["properties"]
    assert props["category_phrase"]["maxLength"] <= 60
    assert props["slots"]["items"]["properties"]["value"]["maxLength"] <= 60
    assert props["slots"]["maxItems"] <= 12

    # every enum the schema advertises is one the scorer can actually receive
    assert set(_ATTR_ALIASES.values()) <= set(ALLOWED_ATTRIBUTES) | {"other"}
    assert set(_DEPT_CANON.values()) <= set(DEPARTMENTS) - {None}
    assert HARD_ATTRIBUTES <= set(ALLOWED_ATTRIBUTES)

    # the normalizer has to survive the catalog's actual spellings, not just tidy ones
    assert normalize_department("Womens") == "womens"
    assert normalize_department("women\u2019s") == "womens"
    assert normalize_department("unisex-adult (luggage only)") == "unisex-adult"
    assert normalize_department("teen-boys;mens") == "boys"
    assert normalize_department(None) is None
    assert normalize_department("watches") is None          # not a department; stay silent
    print("self-check: pass")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="ollama model tag; omit to only convert + self-check")
    ap.add_argument("--host", default="http://localhost:11434")
    args = ap.parse_args()

    self_check()
    cases, warn = load_gold()
    print(f"\ngold: {len(cases)} cases converted from probe_set.md")
    strata: dict[str, int] = {}
    for c in cases:
        strata[c["stratum"]] = strata.get(c["stratum"], 0) + 1
    print("  per stratum:", dict(sorted(strata.items())))
    ext = sum(1 for c in cases for g in c["gold"]["slots"]
              if not g["value"] or (_norm(g["value"]) & _norm(c["utterance"])))
    print(f"  slots: {sum(len(c['gold']['slots']) for c in cases)}"
          f" ({ext} extractable from the utterance,"
          f" {sum(len(c['gold']['slots']) for c in cases) - ext} inferred)"
          f" | declines: {sum(1 for c in cases for s in c['gold']['slots'] if s['declined'])}"
          f" | priced: {sum(1 for c in cases if c['gold']['price_max'] is not None)}"
          f" | quality priors: {sum(1 for c in cases if c['gold']['quality_prior']!='none')}"
          f" | discard spans: {sum(len(c['discard_spans']) for c in cases)}")
    if warn:
        print(f"\n  {len(warn)} value(s) need review:")
        for w in warn:
            print("   -", w)
    else:
        print("  every value mapped cleanly into the schema")

    out = HERE / "probe_gold.json"
    out.write_text(json.dumps(cases, indent=2) + "\n", encoding="utf-8")
    print(f"\n  -> {out.name}")

    if not args.model:
        print("\nno --model given; skipping Ollama run")
        return

    # Cache every prediction. Re-running a 7B over 30 probes costs minutes, and
    # most questions asked afterwards ("why did price regress?") are scorer
    # questions that should never need the model again.
    rows, failures, preds = [], [], []
    for c in cases:
        try:
            pred = parse_with_ollama(c["utterance"], args.model, args.host)
        except Exception as exc:                      # noqa: BLE001 - report, don't mask
            print(f"  case {c['case']:02d}: FAILED {type(exc).__name__}: {exc}")
            failures.append(c["case"])
            continue
        preds.append({"case": c["case"], "stratum": c["stratum"], "pred": pred})
        rows.append({**score(pred, c["gold"], c["discard_spans"], c["utterance"]),
                     "case": c["case"], "stratum": c["stratum"]})
        print(f"  case {c['case']:02d} [{c['stratum']}] f1={rows[-1]['slot_f1']:.2f} "
              f"dept={'ok' if rows[-1]['department_ok'] else 'XX'} "
              f"cat={'ok' if rows[-1]['category_overlap'] else 'XX'}")
    # Never clobber a good cache with a worse run. Ollama dying mid-run once
    # overwrote 30 cached predictions with an empty list, destroying four minutes
    # of model time and every offline diagnostic that depended on it.
    pred_path = HERE / f"preds-{args.model.replace(':', '-').replace('/', '-')}.json"
    existing = 0
    if pred_path.exists():
        try:
            existing = len(json.loads(pred_path.read_text()))
        except (OSError, ValueError):
            existing = 0
    if not preds:
        print(f"\n  every call failed - keeping the existing {existing} cached "
              f"prediction(s), writing nothing")
    elif len(preds) < existing:
        keep = pred_path.with_suffix(".partial.json")
        keep.write_text(json.dumps(preds, indent=2) + "\n", encoding="utf-8")
        print(f"\n  only {len(preds)} of {existing} cached predictions succeeded; "
              f"wrote {keep.name} and left the fuller cache intact")
    else:
        pred_path.write_text(json.dumps(preds, indent=2) + "\n", encoding="utf-8")
        print(f"\n  -> {pred_path.name}")

    n = len(rows)
    if not n:
        print("\nevery case failed; nothing to score")
        return
    print(f"\n{args.model} over {n} probes"
          + (f" ({len(failures)} FAILED: {failures})" if failures else "") + ":")
    for k in ("slot_f1", "slot_precision", "slot_recall", "recall_incl_inferred"):
        print(f"  {k:16s} {sum(r[k] for r in rows)/n:.3f}")
    for k in ("department_ok", "category_overlap", "price_ok", "quality_ok"):
        print(f"  {k:16s} {sum(r[k] for r in rows)/n:.3f}")
    print(f"  spurious/case    {sum(r['spurious_slots'] for r in rows)/n:.2f}")
    print(f"  discard leaks    {sum(r['discard_leaks'] for r in rows)}"
          f" (asides turned into constraints)")


if __name__ == "__main__":
    sys.exit(main())
