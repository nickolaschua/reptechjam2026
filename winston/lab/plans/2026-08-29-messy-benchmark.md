# Messy-Input Benchmark (Phase A) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate ~1,700 messy shopper utterances with free ground truth, stratified by product covariates × the team's search-type taxonomy, and score three retrieval systems plus first-question quality on them with bootstrap CIs.

**Architecture:** Six small stdlib-only scripts under `winston/lab/bench/`, run in sequence: `covariates.py` (per-product scores for the whole catalog, cached) → `sample.py` (~260 products with over-sampled failure pools) → `generate.py` (Ollama shopper, round-robin over three non-parser models, resumable) → `score.py` (three systems, resumable) → `report.py` (CI tables). Prompt text lives in `prompts.py`. One `unittest` file covers every pure function; network calls are behind a `--dry-run` flag so the whole pipeline runs offline for tests.

**Tech Stack:** Python 3 stdlib (`json`, `re`, `random`, `urllib`, `unittest`), `winston/experiments/common.py` (cached catalog index + IDF), `winston/lab/pipeline.py` (resolver), `winston/nlp_parse.py` (parser), `nickolas/experiments/experiment_11_candidate_agent.CleanFTSAgent` (lexical baseline), `techjam-conversational-search/evaluator/local_evaluator.py` (`intent_card`, `behavior_for`). Ollama at `localhost:11434`. `wordfreq` optional.

**Spec:** `winston/lab/specs/2026-08-29-messy-benchmark-design.md`

**Commit note:** `winston/` is currently untracked. The first `git add` in Task 1 adds the folder to the shared repo — confirm with Winston before that first commit. Every later commit is routine.

---

## File map

| file | responsibility | imports |
|---|---|---|
| `winston/lab/bench/prompts.py` | stopwords, `content_words`, forbidden list, style/modifier text, `build_system_prompt`, `relation_for` | stdlib |
| `winston/lab/bench/covariates.py` | §3.2 scores for every catalog product, cached to `.cache/covariates_all.jsonl` | `common.py`, optional `wordfreq` |
| `winston/lab/bench/sample.py` | §3.1 sampling with over-sampled pools → `products.jsonl` | `covariates.py` |
| `winston/lab/bench/generate.py` | §4 case plan + Ollama calls + overlap + resume → `cases.jsonl`, `manifest.json` | `prompts.py`, evaluator, `common.py` |
| `winston/lab/bench/score.py` | §6 three systems → `results.jsonl`, caches parses to `parses.jsonl` | exp11 agent, `nlp_parse.py`, `pipeline.py` |
| `winston/lab/bench/report.py` | §6 bootstrap CIs, slice tables → `report.md` | stdlib |
| `winston/lab/bench/test_bench.py` | one `unittest` file for all pure functions | all of the above |

Shared conventions used in every file:

```python
from pathlib import Path
import sys
BENCH = Path(__file__).resolve().parent          # winston/lab/bench
LAB = BENCH.parent                                # winston/lab
WINSTON = LAB.parent                              # winston
REPO = WINSTON.parent                             # reptechjam2026
KIT = REPO / "techjam-conversational-search"
for p in (WINSTON / "experiments", WINSTON, LAB, str(KIT), REPO / "nickolas" / "experiments"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
```

Run every command from `winston/lab/bench/` unless stated. Tests: `python3 -m unittest test_bench -v`.

---

### Task 0: Environment

**Files:**
- Create: `winston/lab/bench/.gitignore`

- [ ] **Step 1: Create the directory and gitignore**

```bash
mkdir -p "/Users/winstonyang/Desktop/Coding/Hackathons/Techjam 2026/reptechjam2026/winston/lab/bench"
cd "/Users/winstonyang/Desktop/Coding/Hackathons/Techjam 2026/reptechjam2026/winston/lab/bench"
printf '.cache/\n__pycache__/\nparses.jsonl\n' > .gitignore
```

- [ ] **Step 2: Pull the three generator models (≈15 GB, do this in the background while building)**

```bash
ollama pull llama3.1:8b && ollama pull gemma2:9b && ollama pull mistral:7b
```

Expected: three `success` lines. Verify: `ollama list` shows all three plus `qwen2.5:7b-instruct`.

- [ ] **Step 3: Optional jargon dependency**

```bash
pip install wordfreq
python3 -c "import wordfreq; print(wordfreq.zipf_frequency('polyester','en'), wordfreq.zipf_frequency('comfy','en'))"
```

Expected: two numbers, the first lower than the second (e.g. `3.4 4.1`). If install fails, skip — `jargon` will be `null` and everything else works.

---

### Task 1: `prompts.py` — content words, forbidden list, relation

**Files:**
- Create: `winston/lab/bench/prompts.py`
- Create: `winston/lab/bench/test_bench.py`

- [ ] **Step 1: Write the failing tests**

```python
# winston/lab/bench/test_bench.py
import random
import unittest
from types import SimpleNamespace


class TestPrompts(unittest.TestCase):
    def test_content_words_strips_stopwords_and_dedupes(self):
        from prompts import content_words
        self.assertEqual(content_words("The Leather Boots, leather!"), ["leather", "boots"])

    def test_forbidden_list_comes_from_title_and_features_capped(self):
        from prompts import forbidden_list
        product = {"title": "Merrell Vapor Glove Trail Running Shoe",
                   "features": ["100% Textile", "Rubber sole", "Barefoot-style trail runner"]}
        words = forbidden_list(product, cap=5)
        self.assertEqual(len(words), 5)
        self.assertIn("merrell", words)
        self.assertNotIn("100", forbidden_list(product, cap=40))   # pure digits are not vocabulary

    def test_content_words_keeps_alphanumeric_codes(self):
        from prompts import content_words
        self.assertEqual(content_words("BM8242-08E in black"), ["bm8242", "08e", "black"])

    def test_forbidden_list_handles_missing_fields(self):
        from prompts import forbidden_list
        self.assertEqual(forbidden_list({"title": None, "features": None}), [])

    def test_relation_matches_department(self):
        from prompts import relation_for
        rng = random.Random(0)
        self.assertIn(relation_for("womens", rng), {"wife", "mum", "sister", "daughter"})
        self.assertIn(relation_for("mens", rng), {"dad", "husband", "brother", "son"})
        self.assertEqual(relation_for(None, rng), "friend")
        self.assertEqual(relation_for("unisex-adult", rng), "friend")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run to verify they fail**

```bash
python3 -m unittest test_bench -v 2>&1 | tail -5
```

Expected: `ModuleNotFoundError: No module named 'prompts'`

- [ ] **Step 3: Implement**

```python
# winston/lab/bench/prompts.py
"""Utterance-axis text: styles, modifiers, the forbidden list, and the system prompt.

Everything the LLM shopper is told lives here so the benchmark's messiness is
inspectable in one file. Styles follow the team's search-type taxonomy. Spec
section 4.
"""
from __future__ import annotations

import random
import re

TOKEN_RE = re.compile(r"[a-z0-9]+")     # alphanumeric: model codes like bm8242 must survive (spec 4.5)

# Deliberately larger than exp11's list: the forbidden list must not be padded
# with function words, and overlap must not be inflated by them.
STOPWORDS = frozenset("""
a an and are as at be but by for from i in is it me my of on or please some that
the this to want with would you looking need like get one ones also just really
something anything thing things kind sort pair set new good nice great pretty
very quite bit lot more less than then there here what which who how when where
why can could should will do does did have has had not no yes if so up out
""".split())

STYLES = ("exact", "product_type", "feature", "use_case", "symptom",
          "compatibility", "plain", "lay")
GATED = {"exact": "has_model_code", "compatibility": "compat_eligible"}   # style -> product flag it needs
INTENT_LABEL = {st: ("buying" if st == "exact" else "browsing") for st in STYLES}
MODIFIERS = ("negation", "for_other", "vague_budget", "format_noise")
MODIFIER_ONLY_FOR = {"format_noise": "exact"}                             # modifier -> the one style it applies to

STYLE_INSTRUCTIONS = {
    "exact": "You remember this from the listing: {code}. Mention it, plus one other thing you want.",
    "product_type": ("Name only the kind of item you want. Nothing else - no features, "
                     "no occasion, no brand."),
    "feature": "Name the kind of item and two or three things it must have.",
    "use_case": ("Describe the situation, event or task you need this for. Do not name "
                 "the item and do not list its attributes - let the assistant work it out."),
    "symptom": ("Describe the problem you are trying to fix, or how you want to look or "
                "feel. Do not name the item."),
    "compatibility": ("You already own a {anchor} and need this to go with it. Describe "
                      "what you own and what you need it for - do not describe the "
                      "accessory itself."),
    "plain": "Tell the assistant what you're looking for.",
    "lay": ("Describe what you want in everyday words. You must not use any of these "
            "words, or their plurals: {forbidden}."),
}

MODIFIER_INSTRUCTIONS = {
    "negation": "Also say one specific thing you do NOT want.",
    "for_other": "You are buying this for your {relation}.",
    "vague_budget": "Mention that price matters to you, but do not say a number.",
    "format_noise": ("Write the code with different spacing, hyphens or capitalisation "
                     "than the listing."),
}

RELATIONS = {
    "womens": ("wife", "mum", "sister", "daughter"),
    "mens": ("dad", "husband", "brother", "son"),
    "girls": ("daughter", "niece"),
    "baby-girls": ("daughter", "niece"),
    "boys": ("son", "nephew"),
    "baby-boys": ("son", "nephew"),
}


def content_words(text: str) -> list[str]:
    """Lowercase alphanumeric tokens, stopwords and single characters removed,
    first-occurrence order, unique. Shared by the forbidden list and the overlap metric."""
    seen: dict[str, None] = {}
    for tok in TOKEN_RE.findall(str(text or "").lower()):
        if tok not in STOPWORDS and len(tok) > 1:
            seen.setdefault(tok, None)
    return list(seen)


def forbidden_list(product: dict, cap: int = 40) -> list[str]:
    """The listing's own vocabulary, so the `lay` style cannot paraphrase it back."""
    feats = product.get("features") or []
    text = " ".join([str(product.get("title") or ""), *map(str, feats)])
    # "100" from "100% cotton" is not vocabulary; "bm8242" is
    return [w for w in content_words(text) if not w.isdigit()][:cap]


def relation_for(department: str | None, rng: random.Random) -> str:
    """A recipient consistent with the target's department - never a contradiction."""
    options = RELATIONS.get(department or "")
    return rng.choice(options) if options else "friend"


def build_system_prompt(product: dict, card: dict, profile: dict, style: str,
                        modifiers: list[str], code: str | None = None,
                        relation: str | None = None, anchor: str | None = None) -> str:
    """The v2 shopper prompt, single-turn. Relative to experiment_1/shopper_agent.py:
    rule 1 (lead with hard constraints) dropped - it would contradict the use_case /
    symptom styles; rule 3 (paraphrase the listing) dropped - the opposite of messy;
    rule 5 (recognise the product, end the chat) dropped - multi-turn only; the
    Ground Truth ASIN and Category lines dropped - leakage. Rule 5 here is new."""
    if style not in STYLES:
        raise ValueError(f"unknown style {style!r}")
    desc = product.get("description") or ""
    if isinstance(desc, list):
        desc = " ".join(map(str, desc))
    details = product.get("details") or {}
    details_str = " ".join(f"{k}: {v}" for k, v in details.items()) if isinstance(details, dict) else str(details)

    lines = [
        "You are acting as a real customer shopping online. Your target product is:",
        f"Target Product Title: {product.get('title', '')}",
        f"Hard Constraints (Must-Haves): {', '.join(map(str, card.get('hard_constraints', [])))}",
        f"Soft Preferences (Nice-to-Haves): {', '.join(map(str, card.get('soft_preferences', [])))}",
        f"Target Product Details: {details_str}",
        f"Target Product Description: {str(desc)[:300]}",
        "",
        "Your Shopping Profile:",
        f"Purchase Frequency: {profile.get('purchase_frequency', 'Regular')}",
        f"Preference Tags: {', '.join(profile.get('preference_tags', []))}",
        f"Summary: {profile.get('summary', '')}",
        "",
        "Rules:",
        "1. Do NOT state the exact product title or any product code unless told to. Speak like a real human.",
        "2. Use synonyms and subjective phrases instead of copying the listing.",
        "3. Keep it short, informal and conversational.",
        "4. Write in lowercase with simple punctuation. No capital letters.",
        "5. Write only the opening message. One to three sentences.",
        "",
        "How to describe what you want:",
    ]
    instr = STYLE_INSTRUCTIONS[style]
    if style == "lay":
        instr = instr.format(forbidden=", ".join(forbidden_list(product)))
    elif style == "exact":
        if not code:
            raise ValueError("exact style needs a code")
        instr = instr.format(code=code)
    elif style == "compatibility":
        if not anchor:
            raise ValueError("compatibility style needs an anchor")
        instr = instr.format(anchor=anchor)
    lines.append(instr)
    for m in modifiers:
        if m not in MODIFIERS:
            raise ValueError(f"unknown modifier {m!r}")
        if m in MODIFIER_ONLY_FOR and MODIFIER_ONLY_FOR[m] != style:
            raise ValueError(f"modifier {m!r} only applies to style {MODIFIER_ONLY_FOR[m]!r}")
        text = MODIFIER_INSTRUCTIONS[m]
        if m == "for_other":
            text = text.format(relation=relation or "friend")
        lines.append(text)
    return "\n".join(lines)
```

- [ ] **Step 4: Run tests**

```bash
python3 -m unittest test_bench -v 2>&1 | tail -6
```

Expected: `Ran 5 tests ... OK`

- [ ] **Step 5: Commit** (first commit adds the untracked folder — confirm with Winston)

```bash
cd "/Users/winstonyang/Desktop/Coding/Hackathons/Techjam 2026/reptechjam2026"
git add winston/lab/bench/prompts.py winston/lab/bench/test_bench.py winston/lab/bench/.gitignore
git commit -m "bench: prompt text, forbidden list, relation picker"
```

---

### Task 2: `prompts.build_system_prompt` tests

**Files:**
- Modify: `winston/lab/bench/test_bench.py`

- [ ] **Step 1: Add the failing tests**

Append inside `TestPrompts`:

```python
    def _product(self):
        return {"title": "Crocs Classic Clog", "features": ["Croslite foam", "Ventilation ports"],
                "details": {"Department": "unisex-adult"}, "description": ["Iconic clog."]}

    def test_lay_prompt_contains_forbidden_words_and_no_rule3(self):
        from prompts import build_system_prompt
        p = build_system_prompt(self._product(), {"hard_constraints": ["foam"], "soft_preferences": []},
                                {"preference_tags": ["comfort"]}, "lay", [])
        self.assertIn("croslite", p)
        self.assertIn("You must not use any of these words", p)
        self.assertNotIn("Drop descriptive hints", p)

    def test_exact_requires_code_and_uses_it(self):
        from prompts import build_system_prompt
        with self.assertRaises(ValueError):
            build_system_prompt(self._product(), {}, {}, "exact", [])
        p = build_system_prompt(self._product(), {}, {}, "exact", [], code="WA1200")
        self.assertIn("WA1200", p)

    def test_compatibility_requires_anchor_and_uses_it(self):
        from prompts import build_system_prompt
        with self.assertRaises(ValueError):
            build_system_prompt(self._product(), {}, {}, "compatibility", [])
        p = build_system_prompt(self._product(), {}, {}, "compatibility", [], anchor="watch")
        self.assertIn("already own a watch", p)

    def test_modifiers_append_relation_fills_and_format_noise_is_exact_only(self):
        from prompts import build_system_prompt
        p = build_system_prompt(self._product(), {}, {}, "plain", ["negation", "for_other"], relation="dad")
        self.assertIn("do NOT want", p)
        self.assertIn("for your dad", p)
        with self.assertRaises(ValueError):
            build_system_prompt(self._product(), {}, {}, "plain", ["format_noise"])
        p = build_system_prompt(self._product(), {}, {}, "exact", ["format_noise"], code="WA1200")
        self.assertIn("different spacing", p)

    def test_context_block_renders_card_and_profile(self):
        # if the card or profile silently stopped rendering, every utterance would be
        # generated without its constraints and the run would still "succeed"
        from prompts import build_system_prompt
        p = build_system_prompt(self._product(), {"hard_constraints": ["foam"], "soft_preferences": ["ventilated"]},
                                {"preference_tags": ["comfort"], "summary": "buys clogs"}, "plain", [])
        for needle in ("foam", "ventilated", "comfort", "buys clogs", "Crocs Classic Clog", "unisex-adult"):
            self.assertIn(needle, p)

    def test_description_truncated_to_300(self):
        from prompts import build_system_prompt
        product = {**self._product(), "description": ["x" * 400 + "TAIL"]}
        p = build_system_prompt(product, {}, {}, "plain", [])
        self.assertNotIn("TAIL", p)
        self.assertIn("x" * 300, p)
```

- [ ] **Step 2: Run tests**

```bash
python3 -m unittest test_bench.TestPrompts -v 2>&1 | tail -8
```

Expected: `Ran 11 tests ... OK` (implementation from Task 1 already satisfies them; this task exists so the prompt contract is pinned before anything depends on it).

- [ ] **Step 3: Commit**

```bash
git add winston/lab/bench/test_bench.py
git commit -m "bench: pin system-prompt contract"
```

---

### Task 3: `covariates.py` — pure scoring functions

**Files:**
- Create: `winston/lab/bench/covariates.py`
- Modify: `winston/lab/bench/test_bench.py`

- [ ] **Step 1: Write the failing tests**

Append a new class:

```python
def _fake_ix():
    """Four products, two buckets. Enough to exercise every covariate branch."""
    import math
    products = {
        "A1": {"title": "Asics E760Y-0143 Gel Tennis Shoe", "features": ["Rubber sole", "GEL cushioning system"],
               "details": {}, "description": "", "price": 80.0, "rating_number": 500, "categories": ["Clothing, Shoes & Jewelry", "Men", "Shoes"]},
        "A2": {"title": "Asics Gel Tennis Shoe Blue", "features": ["Rubber sole"],
               "details": {}, "description": "", "price": None, "rating_number": 3, "categories": ["Clothing, Shoes & Jewelry", "Men", "Shoes"]},
        "B1": {"title": "Sterling Silver 925 Pendant", "features": ["100% Cotton cord"],
               "details": {"Material": "silver"}, "description": "", "price": 12.0, "rating_number": 40, "categories": ["Clothing, Shoes & Jewelry", "Westlake"]},
        "B2": {"title": "Plain Hoodie", "features": [],
               "details": {}, "description": "", "price": 20.0, "rating_number": 10, "categories": ["Clothing, Shoes & Jewelry", "Women", "Clothing", "Hoodies"]},
    }
    text = {a: " ".join([p["title"], *p["features"], str(p["details"]), str(p["description"])]).lower() for a, p in products.items()}
    fields = {a: {"title": p["title"].lower(), "features": " ".join(p["features"]).lower()} for a, p in products.items()}
    df = {}
    for t in text.values():
        for tok in set(t.split()):
            df[tok] = df.get(tok, 0) + 1
    idf = {tok: math.log(4 / c) for tok, c in df.items()}
    bucket_of = {"A1": "men shoes", "A2": "men shoes", "B1": "watches watch bands", "B2": "clothing hoodies"}
    buckets = {}
    for a, b in bucket_of.items():
        buckets.setdefault(b, []).append(a)
    return SimpleNamespace(products=products, text=text, fields=fields, idf=idf, buckets=buckets, bucket_of=bucket_of,
                           categories={a: p["categories"] for a, p in products.items()})


class TestCovariates(unittest.TestCase):
    def test_model_code_accepts_real_codes_and_rejects_grades(self):
        from covariates import model_code
        self.assertEqual(model_code("Asics E760Y-0143 Gel"), "E760Y-0143")
        self.assertEqual(model_code("VICTONY WA1200 Extender"), "WA1200")
        self.assertIsNone(model_code("Sterling Silver 925 Pendant"))
        self.assertIsNone(model_code("316L Surgical Steel Ring"))
        self.assertIsNone(model_code("14K Gold Chain"))
        self.assertIsNone(model_code("Plain Hoodie"))
        self.assertIsNone(model_code("Sterling S925 Silver Ring"))                  # purity mark, not a code
        self.assertEqual(model_code("Shades UV400 Protection B2599"), "B2599")      # standard skipped, real code found

    def test_near_duplicates_by_title_jaccard_within_bucket(self):
        from covariates import near_duplicates
        dups = near_duplicates(_fake_ix())
        self.assertEqual(dups, {"A1", "A2"})          # B1 and B2 are alone in their buckets

    def test_covariates_for_fields(self):
        from covariates import covariates_for, near_duplicates
        ix = _fake_ix()
        c = covariates_for("B1", ix, near_duplicates(ix))
        self.assertTrue(c["promo_bucket"])             # Westlake path
        self.assertTrue(c["compat_eligible"])          # bucket is a watch-band bucket
        self.assertEqual(c["compat_anchor"], "watch")
        self.assertFalse(c["silent_on_material"])      # "cotton" (silver is not in MATERIAL_RE)
        self.assertFalse(c["has_model_code"])
        self.assertFalse(c["has_near_duplicate"])
        self.assertTrue(c["price_present"])
        self.assertEqual(c["bucket_size"], 1)
        self.assertEqual(c["category_depth"], 2)
        self.assertGreater(c["descriptiveness"], 0.0)
        self.assertTrue(c["jargon"] is None or 0.0 <= c["jargon"] <= 1.0)   # None only without wordfreq
        c2 = covariates_for("B2", ix, set())
        self.assertFalse(c2["compat_eligible"])
        self.assertIsNone(c2["compat_anchor"])
        self.assertIsNone(c2["department"])
        self.assertTrue(c2["silent_on_material"])
        self.assertEqual(c2["descriptiveness"], 0.0)   # no features at all
        self.assertFalse(c2["price_present"] is None)
```

- [ ] **Step 2: Run to verify they fail**

```bash
python3 -m unittest test_bench.TestCovariates -v 2>&1 | tail -4
```

Expected: `ModuleNotFoundError: No module named 'covariates'`

- [ ] **Step 3: Implement**

```python
# winston/lab/bench/covariates.py
"""Product-axis scores (spec section 3.2), computed for the whole catalog once.

Every score answers "for this product, which retrieval signal cannot be trusted?"
All are corpus statistics; the only external resource is an optional general-
English word-frequency list for `jargon`.

    python3 covariates.py        # -> .cache/covariates_all.jsonl (50,000 rows)
"""
from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

BENCH = Path(__file__).resolve().parent
LAB = BENCH.parent
WINSTON = LAB.parent
for p in (WINSTON / "experiments", WINSTON, LAB):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

CACHE = BENCH / ".cache"
ALL_PATH = CACHE / "covariates_all.jsonl"

MATERIAL_RE = re.compile(r"\b(cotton|polyester|nylon|leather|wool|spandex|silk|rayon|denim|suede)\b", re.I)
# letters then digits, e.g. E760Y-0143, WA1200, J03918. Pure digits never match.
MODEL_CODE_RE = re.compile(r"\b([A-Z]{1,4}-?\d{3,}[A-Z0-9-]*)\b")
# purity marks and steel grades that look like codes and are not
GRADE_EXCLUDE = frozenset({"316L", "925", "S925", "14K", "18K", "10K", "585", "750", "24K", "9K",
                           "UV400"})   # S925 = sterling mark, UV400 = lens standard; both looked like codes
CANON_DEPARTMENTS = frozenset({
    "Women", "Men", "Girls", "Boys", "Baby", "Novelty & More", "Sport Specific Clothing",
    "Luggage & Travel Gear", "Costumes & Accessories", "Shoe, Jewelry & Watch Accessories",
    "Uniforms, Work & Safety", "Shoe Care & Accessories",
})
# Things bought FOR an item the shopper already owns. Bucket names are the
# evaluator's coarse_category, lowercased. ~350 products.
COMPAT_ANCHORS = {
    "watches watch bands": "watch",
    "shoe care & accessories shoelaces": "sneakers or boots",
    "charms bead": "charm bracelet",
    "charms & charm bracelets charms": "charm bracelet",
    "shoe care & accessories shoe decoration charms": "charm bracelet",
}
TOK = re.compile(r"[a-z0-9]+")

try:  # optional
    from wordfreq import zipf_frequency
except ImportError:  # pragma: no cover
    zipf_frequency = None


def model_code(title: str) -> str | None:
    for m in MODEL_CODE_RE.finditer(str(title or "")):
        code = m.group(1)
        if code.upper() in GRADE_EXCLUDE or code.isdigit():
            continue
        # reject "925" glued to a word: the letters must be part of the code
        if not re.search(r"[A-Z]", code):
            continue
        return code
    return None


def _title_tokens(ix, asin: str) -> set[str]:
    return set(str(ix.products[asin].get("title") or "").lower().split())


def near_duplicates(ix) -> set[str]:
    """ASINs that have a same-bucket sibling with title-token Jaccard > 0.6.

    O(bucket^2) but the largest bucket is ~1,350, so the whole catalog takes
    tens of seconds. Both members of a pair are flagged.
    """
    flagged: set[str] = set()
    for members in ix.buckets.values():
        toks = {a: _title_tokens(ix, a) for a in members}
        for i, a in enumerate(members):
            if a in flagged and all(b in flagged for b in members[i + 1:]):
                continue
            ta = toks[a]
            if not ta:
                continue
            for b in members[i + 1:]:
                tb = toks[b]
                if tb and len(ta & tb) / len(ta | tb) > 0.6:
                    flagged.add(a)
                    flagged.add(b)
    return flagged


def _idf_mass(ix, text: str) -> float:
    return round(sum(ix.idf.get(t, 0.0) for t in set(TOK.findall(text.lower()))), 3)


def jargon_score(ix, tokens: set[str]) -> float | None:
    """Fraction of tokens common in the catalog but rare in general English."""
    if zipf_frequency is None or not tokens:
        return None
    n = len(ix.products)
    hits = 0
    for t in tokens:
        df = n / math.exp(ix.idf[t]) if t in ix.idf else 0
        if df >= 10 and zipf_frequency(t, "en") < 3.0:
            hits += 1
    return round(hits / len(tokens), 3)


def covariates_for(asin: str, ix, dup_asins: set[str]) -> dict:
    from nlp_parse import normalize_department
    p = ix.products[asin]
    title = str(p.get("title") or "")
    feats = " ".join(map(str, p.get("features") or []))
    cats = ix.categories.get(asin) or p.get("categories") or []
    code = model_code(title)
    bucket = ix.bucket_of[asin]
    return {
        "compat_eligible": bucket in COMPAT_ANCHORS,
        "compat_anchor": COMPAT_ANCHORS.get(bucket),
        "department": normalize_department((p.get("details") or {}).get("Department")),
        "asin": asin,
        "descriptiveness": _idf_mass(ix, feats),
        "title_richness": _idf_mass(ix, title),
        "jargon": jargon_score(ix, set(TOK.findall((title + " " + feats).lower()))),
        "bucket_size": len(ix.buckets[ix.bucket_of[asin]]),
        "popularity": int(p.get("rating_number") or 0),
        "has_model_code": code is not None,
        "model_code": code,
        "silent_on_material": MATERIAL_RE.search(ix.text[asin]) is None,
        "has_near_duplicate": asin in dup_asins,
        "price_present": p.get("price") is not None,
        "promo_bucket": not (len(cats) >= 2 and cats[1] in CANON_DEPARTMENTS),
        "category_depth": len(cats),
    }


def load_all(ix=None) -> dict[str, dict]:
    """Every product's covariates, computed once and cached."""
    if ALL_PATH.exists():
        return {r["asin"]: r for r in map(json.loads, ALL_PATH.open())}
    if ix is None:
        from common import get_index
        ix = get_index()
    dups = near_duplicates(ix)
    rows = {a: covariates_for(a, ix, dups) for a in ix.products}
    CACHE.mkdir(exist_ok=True)
    with ALL_PATH.open("w") as fh:
        for r in rows.values():
            fh.write(json.dumps(r) + "\n")
    return rows


if __name__ == "__main__":
    rows = load_all()
    n = len(rows)
    print(f"{n} products -> {ALL_PATH}")
    for k in ("silent_on_material", "has_near_duplicate", "has_model_code", "compat_eligible", "promo_bucket", "price_present"):
        c = sum(1 for r in rows.values() if r[k])
        print(f"  {k:20s} {c:6d}  ({c / n * 100:4.1f}%)")
    print(f"  jargon available: {any(r['jargon'] is not None for r in rows.values())}")
```

- [ ] **Step 4: Run tests**

```bash
python3 -m unittest test_bench.TestCovariates -v 2>&1 | tail -6
```

Expected: `Ran 3 tests ... OK`

- [ ] **Step 5: Build the full cache and sanity-check the rates**

```bash
python3 covariates.py
```

Expected (measured 2026-08-29): `silent_on_material` 44.7%, `has_near_duplicate` 7.9%, `has_model_code` ~3.1%, `compat_eligible` 0.7% (346), `promo_bucket` 5.9%, `price_present` 21.1%, `jargon available: True`. Takes ~1–2 min the first time.

- [ ] **Step 6: Commit**

```bash
git add winston/lab/bench/covariates.py winston/lab/bench/test_bench.py
git commit -m "bench: per-product covariates with cached full-catalog table"
```

---

### Task 4: `sample.py` — product sampling with over-sampled pools

**Files:**
- Create: `winston/lab/bench/sample.py`
- Modify: `winston/lab/bench/test_bench.py`

- [ ] **Step 1: Write the failing tests**

```python
class TestSample(unittest.TestCase):
    def _cov(self, n=400):
        rows = {}
        for i in range(n):
            a = f"P{i:04d}"
            rows[a] = {"asin": a, "popularity": i, "silent_on_material": i % 7 == 0,
                       "has_near_duplicate": i % 11 == 0, "has_model_code": i % 13 == 0,
                       "model_code": "X100" if i % 13 == 0 else None, "promo_bucket": False,
                       "compat_eligible": i % 17 == 0}
        return rows

    def test_pools_are_tagged_and_deduped(self):
        from sample import sample_products
        rows = self._cov()
        out = sample_products(rows, seed=1, base=50, per_pool=10)
        asins = [r["asin"] for r in out]
        self.assertEqual(len(asins), len(set(asins)))
        pools = {}
        for r in out:
            for pl in r["pools"]:
                pools[pl] = pools.get(pl, 0) + 1
        for pl in ("base", "silent_on_material", "has_near_duplicate", "low_popularity", "has_model_code", "compat_eligible"):
            self.assertGreaterEqual(pools[pl], 10, pl)
        low = [r for r in out if "low_popularity" in r["pools"]]
        self.assertTrue(all(r["popularity"] < 100 for r in low))   # bottom quartile of 0..399

    def test_seed_reproduces(self):
        from sample import sample_products
        rows = self._cov()
        a = [r["asin"] for r in sample_products(rows, seed=7, base=30, per_pool=5)]
        b = [r["asin"] for r in sample_products(rows, seed=7, base=30, per_pool=5)]
        self.assertEqual(a, b)
```

- [ ] **Step 2: Run to verify they fail**

```bash
python3 -m unittest test_bench.TestSample -v 2>&1 | tail -4
```

Expected: `ModuleNotFoundError: No module named 'sample'`

- [ ] **Step 3: Implement**

```python
# winston/lab/bench/sample.py
"""Spec section 3.1: ~260 products, with the failure and gated-style pools over-sampled.

    python3 sample.py            # -> products.jsonl
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

BENCH = Path(__file__).resolve().parent
sys.path.insert(0, str(BENCH))
from covariates import load_all  # noqa: E402

SEED = 20260829
PRODUCTS_PATH = BENCH / "products.jsonl"


def sample_products(cov: dict[str, dict], seed: int = SEED, base: int = 120, per_pool: int = 30) -> list[dict]:
    rng = random.Random(seed)
    rows = [r for r in cov.values() if r.get("title_ok", True)]
    pops = sorted(r["popularity"] for r in rows)
    q1 = pops[len(pops) // 4]

    pools = {
        "silent_on_material": [r for r in rows if r["silent_on_material"]],
        "has_near_duplicate": [r for r in rows if r["has_near_duplicate"]],
        "low_popularity": [r for r in rows if r["popularity"] < q1],
        "has_model_code": [r for r in rows if r["has_model_code"]],
        "compat_eligible": [r for r in rows if r["compat_eligible"]],
    }
    chosen: dict[str, dict] = {}

    def take(pool: list[dict], n: int, tag: str) -> None:
        # sort first so rng.sample is deterministic regardless of dict order
        for r in rng.sample(sorted(pool, key=lambda x: x["asin"]), min(n, len(pool))):
            entry = chosen.setdefault(r["asin"], {**r, "pools": []})
            entry["pools"].append(tag)

    take(rows, base, "base")
    for tag, pool in pools.items():
        take(pool, per_pool, tag)
    return sorted(chosen.values(), key=lambda r: r["asin"])


def main() -> None:
    from common import get_index
    ix = get_index()
    cov = load_all(ix)
    for a, r in cov.items():
        r["title_ok"] = bool(str(ix.products[a].get("title") or "").strip())
    out = sample_products(cov)
    with PRODUCTS_PATH.open("w") as fh:
        for r in out:
            r.pop("title_ok", None)
            fh.write(json.dumps(r) + "\n")
    print(f"{len(out)} products -> {PRODUCTS_PATH.name}")
    tally: dict[str, int] = {}
    for r in out:
        for p in r["pools"]:
            tally[p] = tally.get(p, 0) + 1
    print("  pools:", tally)


if __name__ == "__main__":
    sys.path.insert(0, str(BENCH.parent.parent / "experiments"))
    main()
```

- [ ] **Step 4: Run tests**

```bash
python3 -m unittest test_bench.TestSample -v 2>&1 | tail -5
```

Expected: `Ran 2 tests ... OK`

- [ ] **Step 5: Generate the real sample**

```bash
python3 sample.py
```

Expected: `~2xx products -> products.jsonl` and a pools tally with every pool ≥ 25 (spec §7).

- [ ] **Step 6: Commit**

```bash
git add winston/lab/bench/sample.py winston/lab/bench/test_bench.py winston/lab/bench/products.jsonl
git commit -m "bench: seeded product sample with over-sampled failure pools"
```

---

### Task 5: `generate.py` — overlap and the case plan (no network yet)

**Files:**
- Create: `winston/lab/bench/generate.py`
- Modify: `winston/lab/bench/test_bench.py`

- [ ] **Step 1: Write the failing tests**

```python
class TestGenerate(unittest.TestCase):
    def test_overlap_is_fraction_of_utterance_content_words_in_listing(self):
        from generate import overlap
        product = {"title": "Merrell Vapor Glove Trail Running Shoe", "features": ["Rubber sole"]}
        self.assertAlmostEqual(overlap("i want trail running shoes from merrell", product), 3 / 4, places=3)
        self.assertEqual(overlap("something for the mountains", product), 0.0)
        self.assertEqual(overlap("", product), 0.0)

    def test_plan_cases_gates_styles_labels_intent_and_is_deterministic(self):
        from generate import plan_cases, GENERATORS
        products = [{"asin": "A", "has_model_code": True, "model_code": "WA1200", "compat_eligible": False,
                     "compat_anchor": None, "department": "mens"},
                    {"asin": "B", "has_model_code": False, "model_code": None, "compat_eligible": True,
                     "compat_anchor": "watch", "department": None}]
        plan = plan_cases(products, seed=3)
        self.assertEqual(plan, plan_cases(products, seed=3))
        styles = [c["style"] for c in plan]
        self.assertEqual(styles.count("exact"), 1)             # only A has a code
        self.assertEqual(styles.count("compatibility"), 1)     # only B is eligible
        for st in ("product_type", "feature", "use_case", "symptom", "plain", "lay"):
            self.assertEqual(styles.count(st), 2)
        self.assertTrue(all(c["generator"] in GENERATORS for c in plan))
        exact = next(c for c in plan if c["style"] == "exact")
        self.assertEqual(exact["code"], "WA1200")
        self.assertEqual(exact["intent_label"], "buying")
        compat = next(c for c in plan if c["style"] == "compatibility")
        self.assertEqual(compat["anchor"], "watch")
        self.assertEqual(compat["intent_label"], "browsing")
        for c in plan:
            if "for_other" in c["modifiers"]:
                self.assertTrue(c["relation"])
            if "format_noise" in c["modifiers"]:
                self.assertEqual(c["style"], "exact")
```

- [ ] **Step 2: Run to verify they fail**

```bash
python3 -m unittest test_bench.TestGenerate -v 2>&1 | tail -4
```

Expected: `ModuleNotFoundError: No module named 'generate'`

- [ ] **Step 3: Implement the pure half**

```python
# winston/lab/bench/generate.py
"""Spec section 4: turn sampled products into messy utterances.

    python3 generate.py --dry-run --limit 6    # offline smoke test, canned text
    python3 generate.py                        # full run, resumable

Resumable: cases already present in cases.jsonl are skipped, so a crash or an
Ollama restart costs nothing. Ground truth is the asin; nothing else is gold.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
import urllib.request
from pathlib import Path

BENCH = Path(__file__).resolve().parent
LAB = BENCH.parent
WINSTON = LAB.parent
REPO = WINSTON.parent
KIT = REPO / "techjam-conversational-search"
for p in (WINSTON / "experiments", WINSTON, LAB, BENCH, KIT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from prompts import (STYLES, GATED, INTENT_LABEL, MODIFIERS, MODIFIER_ONLY_FOR,  # noqa: E402
                     content_words, build_system_prompt, relation_for)

SEED = 20260829
GENERATORS = ("llama3.1:8b", "gemma2:9b", "mistral:7b")     # never the parser's qwen2.5
MODIFIER_P = {"negation": 0.2, "for_other": 0.2, "vague_budget": 0.2, "format_noise": 0.3}
OVERLAP_LIMIT = 0.5
USER_TURN = "Start the conversation by telling the assistant what you are looking for."
PRODUCTS_PATH = BENCH / "products.jsonl"
CASES_PATH = BENCH / "cases.jsonl"
MANIFEST_PATH = BENCH / "manifest.json"


def overlap(utterance: str, product: dict) -> float:
    """Share of the utterance's content words that appear in title+features."""
    u = content_words(utterance)
    if not u:
        return 0.0
    listing = set(content_words(" ".join([str(product.get("title") or ""),
                                          *map(str, product.get("features") or [])])))
    return round(sum(1 for w in u if w in listing) / len(u), 3)


def plan_cases(products: list[dict], seed: int = SEED) -> list[dict]:
    """Deterministic case plan: which product x style x modifiers x generator.

    The LLM output is not reproducible (temperature 0.7); the PLAN is, so a
    partial run can be resumed and two runs can be compared case-for-case.
    """
    rng = random.Random(seed)
    plan: list[dict] = []
    gi = 0
    for prod in sorted(products, key=lambda r: r["asin"]):
        styles = [st for st in STYLES if st not in GATED or prod.get(GATED[st])]
        for style in styles:
            mods = [m for m in MODIFIERS
                    if MODIFIER_ONLY_FOR.get(m, style) == style and rng.random() < MODIFIER_P[m]]
            relation = relation_for(prod.get("department"), rng) if "for_other" in mods else None
            plan.append({
                "case_id": f"c{len(plan) + 1:04d}",
                "asin": prod["asin"],
                "style": style,
                "intent_label": INTENT_LABEL[style],
                "modifiers": mods,
                "relation": relation,
                "code": prod.get("model_code") if style == "exact" else None,
                "anchor": prod.get("compat_anchor") if style == "compatibility" else None,
                "generator": GENERATORS[gi % len(GENERATORS)],
            })
            gi += 1
    return plan


def ollama_chat(model: str, system: str, user: str, temperature: float = 0.7,
                timeout: int = 120, host: str = "http://localhost:11434") -> str:
    body = json.dumps({
        "model": model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "stream": False,
        "options": {"temperature": temperature, "num_predict": 160},
    }).encode()
    req = urllib.request.Request(f"{host}/api/chat", body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())["message"]["content"].strip()


def _canned(case: dict, product: dict) -> str:
    """Offline stand-in so the whole pipeline can be exercised without Ollama."""
    return f"[dry-run {case['style']}] looking for something like {str(product.get('title') or '')[:30].lower()}"


def run(products: list[dict], plan: list[dict], ix, samples_profile: dict, *, dry_run: bool,
        limit: int | None, out_path: Path = CASES_PATH) -> dict:
    from evaluator.local_evaluator import intent_card

    done = set()
    if out_path.exists():
        done = {json.loads(l)["case_id"] for l in out_path.open() if l.strip()}
    by_asin = {p["asin"]: p for p in products}
    todo = [c for c in plan if c["case_id"] not in done][:limit]
    print(f"plan {len(plan)} | done {len(done)} | this run {len(todo)}")

    started = time.time()
    counts: dict[str, int] = {}
    with out_path.open("a") as fh:
        for i, case in enumerate(todo, 1):
            product = ix.products[case["asin"]]
            card = intent_card(product)
            system = build_system_prompt(product, card, samples_profile, case["style"],
                                         case["modifiers"], code=case["code"],
                                         relation=case["relation"], anchor=case["anchor"])
            if dry_run:
                text = _canned(case, product)
            else:
                text = ollama_chat(case["generator"], system, USER_TURN)
                if overlap(text, product) > OVERLAP_LIMIT:
                    text = ollama_chat(case["generator"], system, USER_TURN)
            ov = overlap(text, product)
            row = {**case, "utterance": text, "overlap": ov, "overlap_flag": ov > OVERLAP_LIMIT}
            row.pop("code", None)
            row.pop("relation", None)
            row.pop("anchor", None)
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            counts[case["style"]] = counts.get(case["style"], 0) + 1
            if i % 10 == 0 or i == len(todo):
                el = time.time() - started
                print(f"  {i}/{len(todo)}  {el / i:.1f}s/case  eta {(len(todo) - i) * el / i / 60:.0f} min", flush=True)
    return {"generated_this_run": len(todo), "seconds": round(time.time() - started, 1), "by_style": counts}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="canned utterances, no Ollama")
    ap.add_argument("--limit", type=int, help="stop after N new cases")
    ap.add_argument("--out", type=Path, default=CASES_PATH)
    args = ap.parse_args()

    from common import get_index
    ix = get_index()
    products = [json.loads(l) for l in PRODUCTS_PATH.open()]
    # profile text is only flavour for the shopper; keep it constant and neutral
    profile = {"purchase_frequency": "a few prior purchases", "preference_tags": ["fit", "comfort"],
               "summary": "Prior purchases emphasize fit and comfort."}
    plan = plan_cases(products)
    stats = run(products, plan, ix, profile, dry_run=args.dry_run, limit=args.limit, out_path=args.out)

    rows = [json.loads(l) for l in args.out.open() if l.strip()]
    manifest = {
        "seed": SEED, "generators": GENERATORS, "modifier_p": MODIFIER_P, "overlap_limit": OVERLAP_LIMIT,
        "planned": len(plan), "written": len(rows), "dry_run": args.dry_run,
        "by_style": {s: sum(1 for r in rows if r["style"] == s) for s in STYLES},
        "by_generator": {g: sum(1 for r in rows if r["generator"] == g) for g in GENERATORS},
        "by_modifier": {m: sum(1 for r in rows if m in r["modifiers"]) for m in MODIFIERS},
        "by_intent": {lab: sum(1 for r in rows if r["intent_label"] == lab) for lab in ("buying", "browsing")},
        "overlap_flagged": sum(1 for r in rows if r["overlap_flag"]),
        "last_run": stats,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest["by_style"]), "->", MANIFEST_PATH.name)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests**

```bash
python3 -m unittest test_bench.TestGenerate -v 2>&1 | tail -5
```

Expected: `Ran 2 tests ... OK`

- [ ] **Step 5: Offline smoke run**

```bash
python3 generate.py --dry-run --limit 6 --out /tmp/cases_dry.jsonl && head -2 /tmp/cases_dry.jsonl
```

Expected: `plan 1xxx | done 0 | this run 6`, two JSON rows with `"utterance": "[dry-run ...`, `overlap` numeric, `manifest.json` written.

- [ ] **Step 6: Commit**

```bash
git add winston/lab/bench/generate.py winston/lab/bench/test_bench.py
git commit -m "bench: case plan, overlap, resumable generator with dry-run"
```

---

### Task 6: `generate.py` — live check on two cases

**Files:** none new.

- [ ] **Step 1: Confirm Ollama and models**

```bash
curl -s localhost:11434/api/tags | python3 -c "import json,sys; print([m['name'] for m in json.load(sys.stdin)['models']])"
```

Expected: list containing `llama3.1:8b`, `gemma2:9b`, `mistral:7b`.

- [ ] **Step 2: Generate two real cases to a scratch file**

```bash
python3 generate.py --limit 2 --out /tmp/cases_live.jsonl && python3 -c "
import json
for l in open('/tmp/cases_live.jsonl'):
    r=json.loads(l); print(r['style'], r['generator'], round(r['overlap'],2), '|', r['utterance'][:120])"
```

Expected: two lowercase, informal utterances; `overlap` well below 0.5; ~8–15 s each. If a call times out, raise `timeout` in `ollama_chat` to 180 — the first call to a freshly loaded model includes load time.

- [ ] **Step 3: No commit** (scratch output only).

---

### Task 7: `score.py` — three systems

**Files:**
- Create: `winston/lab/bench/score.py`
- Modify: `winston/lab/bench/test_bench.py`

- [ ] **Step 1: Write the failing tests**

```python
class TestScore(unittest.TestCase):
    def test_rank_of_helper(self):
        from score import rank_of
        self.assertEqual(rank_of(["a", "b", "c"], "b"), 2)
        self.assertIsNone(rank_of(["a", "b"], "z"))
        self.assertIsNone(rank_of([], "a"))

    def test_question_hit_matches_evaluator_classification(self):
        from score import question_hit
        product = {"title": "Leather boots", "features": ["100% Leather", "Rubber sole"], "details": {},
                   "description": "", "price": None}
        # intent_card puts the material first -> "leather" -> classify_constraint -> "material"
        self.assertTrue(question_hit("material", product))
        self.assertFalse(question_hit("budget", product))
        self.assertFalse(question_hit(None, product))

    def test_resolver_rank_uses_phrase_and_soft_slots(self):
        from score import resolver_query
        parse = {"category_phrase": "shoe", "slots": [
            {"attribute": "use_case", "value": "running", "declined": False},
            {"attribute": "material", "value": "leather", "declined": False},   # hard -> excluded
            {"attribute": "color", "value": "red", "declined": True},           # declined -> excluded
        ]}
        self.assertEqual(resolver_query(parse), ["shoe", "running"])
```

- [ ] **Step 2: Run to verify they fail**

```bash
python3 -m unittest test_bench.TestScore -v 2>&1 | tail -4
```

Expected: `ModuleNotFoundError: No module named 'score'`

- [ ] **Step 3: Implement**

```python
# winston/lab/bench/score.py
"""Spec section 6: where does the target rank, for each of three systems?

  template   exp11 as-is. Its turn-1 parser is regex-only, so a messy message
             leaves the query empty. Expected near zero; documents the limitation.
             Also records ask_attribute and question_hit: did the first question
             target an attribute the shopper's hidden intent card actually has?
  lexical    exp11's own FTS5 + reranker with the raw utterance as the query.
             The FAIR baseline - same retrieval, no template dependence.
  resolver   nlp_parse -> pipeline.resolve. Rank of the TRUE BUCKET among 1,115.
             Needs Ollama (qwen); parses are cached to parses.jsonl.

    python3 score.py --skip-resolver     # fast: the two lexical systems only
    python3 score.py                     # everything, resumable
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

BENCH = Path(__file__).resolve().parent
LAB = BENCH.parent
WINSTON = LAB.parent
REPO = WINSTON.parent
KIT = REPO / "techjam-conversational-search"
for p in (WINSTON / "experiments", WINSTON, LAB, BENCH, KIT, REPO / "nickolas" / "experiments"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

CASES_PATH = BENCH / "cases.jsonl"
RESULTS_PATH = BENCH / "results.jsonl"
PARSES_PATH = BENCH / "parses.jsonl"
PARSER_MODEL = "qwen2.5:7b-instruct"
TOP_K = 50


def rank_of(ranked: list[str], target: str) -> int | None:
    try:
        return ranked.index(target) + 1
    except ValueError:
        return None


def resolver_query(parse: dict) -> list[str]:
    """category_phrase + soft slots - the arm that won in pipeline.py (B)."""
    from nlp_parse import tier_of
    terms = [parse.get("category_phrase") or ""]
    terms += [s["value"] for s in parse.get("slots", []) if tier_of(s) == "soft"]
    return [t for t in terms if t]


def template_rank(agent, sid: str, utterance: str, asin: str) -> tuple[int | None, str | None]:
    """(rank, ask_attribute) from exp11 as shipped."""
    agent.reset(sid, {})
    resp = agent.respond(sid, utterance, 1, TOP_K)
    return rank_of([r["parent_asin"] for r in resp["recommendations"]], asin), resp.get("ask_attribute")


def question_hit(ask_attribute: str | None, product: dict) -> bool:
    """Would this question have elicited a real disclosure from the simulator?"""
    from evaluator.local_evaluator import intent_card, classify_constraint
    if not ask_attribute:
        return False
    return ask_attribute in {classify_constraint(str(c)) for c in intent_card(product).get("hard_constraints", [])}


def lexical_rank(agent, sid: str, utterance: str, asin: str) -> int | None:
    """Bypass the template regex: hand the utterance to exp11's own query path."""
    agent.reset(sid, {})
    state = agent.sessions[sid]
    state["category"] = utterance
    return rank_of(agent._rank(state, TOP_K), asin)


def load_parses() -> dict[str, dict]:
    if not PARSES_PATH.exists():
        return {}
    return {json.loads(l)["case_id"]: json.loads(l)["parse"] for l in PARSES_PATH.open() if l.strip()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-resolver", action="store_true")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--cases", type=Path, default=CASES_PATH)
    ap.add_argument("--out", type=Path, default=RESULTS_PATH)
    args = ap.parse_args()

    from common import get_index
    from experiment_11_candidate_agent import CleanFTSAgent
    ix = get_index()
    agent = CleanFTSAgent(KIT / "data" / "catalog.jsonl", pagination_mode="none")
    print(f"exp11 index built in {agent.index_build_seconds}s")

    profiles = None
    resolve = None
    parse_fn = None
    if not args.skip_resolver:
        from pipeline import content_profiles, resolve as _resolve
        from nlp_parse import parse_with_ollama
        profiles = content_profiles(ix)
        resolve = _resolve
        parse_fn = parse_with_ollama

    cases = [json.loads(l) for l in args.cases.open() if l.strip()][:args.limit]
    done = {json.loads(l)["case_id"] for l in args.out.open()} if args.out.exists() else set()
    parses = load_parses()
    todo = [c for c in cases if c["case_id"] not in done]
    print(f"cases {len(cases)} | scored {len(done)} | this run {len(todo)}")

    t0 = time.time()
    with args.out.open("a") as out, PARSES_PATH.open("a") as pf:
        for i, c in enumerate(todo, 1):
            sid = c["case_id"]
            t_rank, asked = template_rank(agent, sid, c["utterance"], c["asin"])
            row = {"case_id": sid, "asin": c["asin"],
                   "template_rank": t_rank,
                   "ask_attribute": asked,
                   "question_hit": question_hit(asked, ix.products[c["asin"]]),
                   "lexical_rank": lexical_rank(agent, sid, c["utterance"], c["asin"])}
            if not args.skip_resolver:
                parse = parses.get(sid)
                if parse is None:
                    parse = parse_fn(c["utterance"], PARSER_MODEL)
                    pf.write(json.dumps({"case_id": sid, "parse": parse}) + "\n")
                    pf.flush()
                ranked, conf = resolve(resolver_query(parse), ix, profiles, top_n=len(ix.buckets))
                row["bucket_rank"] = rank_of(ranked, ix.bucket_of[c["asin"]])
                row["resolver_confidence"] = round(conf, 3)
                row["category_phrase"] = parse.get("category_phrase")
            out.write(json.dumps(row) + "\n")
            out.flush()
            if i % 25 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)}  {(time.time() - t0) / i:.1f}s/case", flush=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests**

```bash
python3 -m unittest test_bench.TestScore -v 2>&1 | tail -5
```

Expected: `Ran 3 tests ... OK`

- [ ] **Step 5: Regression check — the fair baseline must reproduce exp07's template result**

A templated message must land the target at rank 1 through `lexical_rank` on a known public target:

```bash
python3 - <<'EOF'
import sys, json
from pathlib import Path
B = Path("."); sys.path.insert(0, str(B))
import score
from experiment_11_candidate_agent import CleanFTSAgent
KIT = score.KIT
agent = CleanFTSAgent(KIT / "data" / "catalog.jsonl", pagination_mode="none")
s = json.loads(open(KIT / "data" / "public_set.jsonl").readline())
asin = s["ground_truth"]["parent_asin"]
from evaluator.local_evaluator import catalog_index, coarse_category
_, cats, _ = catalog_index(KIT / "data" / "catalog.jsonl")
msg = f"I'm looking for {coarse_category(cats[asin])}."
print("template_rank, asked:", score.template_rank(agent, "t", msg, asin))
print("lexical_rank :", score.lexical_rank(agent, "l", msg, asin))
print("messy template_rank:", score.template_rank(agent, "m", "need something comfy for a beach trip", asin)[0])
EOF
```

Expected: `template_rank` and `lexical_rank` both small integers (≤ 10) for the templated message; the messy `template_rank` is `None` or large — that is the documented limitation, not a bug.

- [ ] **Step 6: Score the dry-run cases end to end (offline)**

```bash
python3 score.py --skip-resolver --cases /tmp/cases_dry.jsonl --out /tmp/results_dry.jsonl && head -2 /tmp/results_dry.jsonl
```

Expected: 6 rows with `template_rank`, `ask_attribute`, `question_hit`, `lexical_rank` keys.

- [ ] **Step 7: Commit**

```bash
git add winston/lab/bench/score.py winston/lab/bench/test_bench.py
git commit -m "bench: score template, fair-lexical and resolver systems, resumable"
```

---

### Task 8: `report.py` — bootstrap CIs and slice tables

**Files:**
- Create: `winston/lab/bench/report.py`
- Modify: `winston/lab/bench/test_bench.py`

- [ ] **Step 1: Write the failing tests**

```python
class TestReport(unittest.TestCase):
    def test_metrics(self):
        from report import hit10, mrr
        ranks = [1, 5, 11, None, 2]
        self.assertAlmostEqual(hit10(ranks), 3 / 5)
        self.assertAlmostEqual(mrr(ranks), (1 + 0.2 + 0 + 0 + 0.5) / 5)

    def test_bootstrap_ci_brackets_the_point_estimate_and_is_seeded(self):
        from report import bootstrap_ci, hit10
        ranks = [1, 2, 3, 15, None, 4, 1, 30, 2, 9] * 5
        lo, hi = bootstrap_ci(ranks, hit10, n_boot=500, seed=1)
        self.assertLessEqual(lo, hit10(ranks))
        self.assertGreaterEqual(hi, hit10(ranks))
        self.assertEqual((lo, hi), bootstrap_ci(ranks, hit10, n_boot=500, seed=1))

    def test_quartile_label(self):
        from report import quartile_label
        vals = list(range(100))
        self.assertEqual(quartile_label(0, vals), "Q1")
        self.assertEqual(quartile_label(99, vals), "Q4")
        self.assertEqual(quartile_label(None, vals), "n/a")
```

- [ ] **Step 2: Run to verify they fail**

```bash
python3 -m unittest test_bench.TestReport -v 2>&1 | tail -4
```

Expected: `ModuleNotFoundError: No module named 'report'`

- [ ] **Step 3: Implement**

```python
# winston/lab/bench/report.py
"""Spec section 6 reporting: every cell carries a bootstrap 95% CI.

    python3 report.py            # -> report.md
"""
from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Callable

BENCH = Path(__file__).resolve().parent
CASES_PATH = BENCH / "cases.jsonl"
PRODUCTS_PATH = BENCH / "products.jsonl"
RESULTS_PATH = BENCH / "results.jsonl"
REPORT_PATH = BENCH / "report.md"
MIN_N = 20
COVARIATES = ("descriptiveness", "title_richness", "jargon", "bucket_size", "popularity")
FLAGS = ("silent_on_material", "has_near_duplicate", "has_model_code", "compat_eligible", "promo_bucket", "price_present")


def hit10(ranks: list[int | None]) -> float:
    return sum(1 for r in ranks if r is not None and r <= 10) / len(ranks) if ranks else 0.0


def mrr(ranks: list[int | None]) -> float:
    return sum(1.0 / r for r in ranks if r) / len(ranks) if ranks else 0.0


def bootstrap_ci(values: list, metric: Callable, n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng = random.Random(seed)
    stats = sorted(metric(rng.choices(values, k=len(values))) for _ in range(n_boot))
    return round(stats[int(0.025 * n_boot)], 3), round(stats[int(0.975 * n_boot)], 3)


def quartile_label(value, population: list) -> str:
    if value is None:
        return "n/a"
    pop = sorted(v for v in population if v is not None)
    if not pop:
        return "n/a"
    cuts = [pop[len(pop) * q // 4] for q in (1, 2, 3)]
    return "Q" + str(1 + sum(value >= c for c in cuts))


def cell(ranks: list, metric: Callable) -> str:
    if not ranks:
        return "—"
    lo, hi = bootstrap_ci(ranks, metric)
    flag = " *" if len(ranks) < MIN_N else ""
    return f"{metric(ranks):.3f} [{lo:.3f}, {hi:.3f}] n={len(ranks)}{flag}"


def table(rows: list[dict], key: Callable[[dict], str], rank_field: str, title: str) -> list[str]:
    groups: dict[str, list] = defaultdict(list)
    for r in rows:
        groups[key(r)].append(r.get(rank_field))
    out = [f"### {title} — `{rank_field}`", "", "| slice | HitRate@10 | MRR |", "|---|---|---|"]
    for g in sorted(groups):
        out.append(f"| {g} | {cell(groups[g], hit10)} | {cell(groups[g], mrr)} |")
    return out + [""]


def main() -> None:
    cases = {r["case_id"]: r for r in map(json.loads, CASES_PATH.open())}
    products = {r["asin"]: r for r in map(json.loads, PRODUCTS_PATH.open())}
    results = [json.loads(l) for l in RESULTS_PATH.open() if l.strip()]
    rows = [{**cases[r["case_id"]], **products[r["asin"]], **r} for r in results if r["case_id"] in cases]
    if not rows:
        raise SystemExit("no scored rows")

    rank_fields = [f for f in ("template_rank", "lexical_rank", "bucket_rank") if any(f in r for r in rows)]
    md = [f"# Messy benchmark report", "", f"{len(rows)} scored cases. `*` = n < {MIN_N}.", ""]
    if any("question_hit" in r for r in rows):
        md += ["## First-question quality (`question_hit`)", "",
               "| slice | hit rate |", "|---|---|"]
        for key, lab in (("all", lambda r: "all"), ("style", lambda r: r["style"]),
                         ("intent", lambda r: r["intent_label"])):
            groups: dict[str, list] = defaultdict(list)
            for r in rows:
                groups[lab(r)].append(1 if r.get("question_hit") else None)   # rank 1 = hit, None = miss
            for g in sorted(groups):
                md.append(f"| {key}={g} | {cell(groups[g], hit10)} |")
        md.append("")
    for rf in rank_fields:
        md += [f"## {rf}", ""]
        md += table(rows, lambda r: "all", rf, "Overall")
        md += table(rows, lambda r: r["style"], rf, "By style")
        md += table(rows, lambda r: r["intent_label"], rf, "By intent label")
        md += table(rows, lambda r: r["generator"], rf, "By generator")
        for m in ("negation", "for_other", "vague_budget", "format_noise"):
            md += table(rows, lambda r, m=m: f"{m}={m in r['modifiers']}", rf, f"By modifier {m}")
        md += table(rows, lambda r: f"overlap {quartile_label(r['overlap'], [x['overlap'] for x in rows])}", rf, "By listing overlap quartile")
        for cv in COVARIATES:
            pop = [x.get(cv) for x in rows]
            md += table(rows, lambda r, cv=cv, pop=pop: f"{cv} {quartile_label(r.get(cv), pop)}", rf, f"By {cv} quartile")
        for fl in FLAGS:
            md += table(rows, lambda r, fl=fl: f"{fl}={bool(r.get(fl))}", rf, f"By {fl}")
    REPORT_PATH.write_text("\n".join(md))
    print(f"{len(rows)} rows -> {REPORT_PATH.name}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests**

```bash
python3 -m unittest test_bench -v 2>&1 | tail -6
```

Expected: `Ran 24 tests ... OK`

- [ ] **Step 5: Report on the dry-run data**

```bash
cp /tmp/cases_dry.jsonl cases.jsonl && cp /tmp/results_dry.jsonl results.jsonl && python3 report.py && head -20 report.md && rm cases.jsonl results.jsonl
```

Expected: `6 rows -> report.md`; tables render with `*` on every cell (n < 20).

- [ ] **Step 6: Commit**

```bash
git add winston/lab/bench/report.py winston/lab/bench/test_bench.py
git commit -m "bench: CI tables by style, generator, modifier, covariate quartile"
```

---

### Task 9: Full generation run

**Files:** produces `winston/lab/bench/cases.jsonl`, `manifest.json`.

- [ ] **Step 1: Launch in the background (~4–6 h, resumable)**

```bash
cd "/Users/winstonyang/Desktop/Coding/Hackathons/Techjam 2026/reptechjam2026/winston/lab/bench"
nohup python3 -u generate.py > generate.log 2>&1 &
echo $!
```

- [ ] **Step 2: Check progress any time**

```bash
tail -3 generate.log; wc -l cases.jsonl
```

If Ollama dies (`Connection refused`), `ollama serve &` and relaunch Step 1 — it resumes.

- [ ] **Step 3: Verify §7 success criteria when done**

```bash
python3 - <<'EOF'
import json, statistics
rows = [json.loads(l) for l in open("cases.jsonl")]
m = json.load(open("manifest.json"))
print("cases:", len(rows), "| by style:", m["by_style"], "| by intent:", m["by_intent"])
OPEN = ("product_type", "feature", "use_case", "symptom", "plain", "lay")
for s in OPEN:
    ov = [r["overlap"] for r in rows if r["style"] == s]
    print(f"  {s:13s} n={len(ov):4d} median overlap {statistics.median(ov):.2f}")
from prompts import content_words
pt_len = [len(content_words(r["utterance"])) for r in rows if r["style"] == "product_type"]
assert len(rows) >= 1000
assert all(m["by_style"][s] >= 200 for s in OPEN)
assert m["by_style"]["exact"] >= 30 and m["by_style"]["compatibility"] >= 30
assert statistics.median([r["overlap"] for r in rows if r["style"] == "lay"]) < 0.30
assert statistics.median([r["overlap"] for r in rows if r["style"] in ("use_case", "symptom")]) < 0.35
assert statistics.median(pt_len) < 8, f"product_type not broad enough: median {statistics.median(pt_len)} words"
print("all section-7 criteria pass")
EOF
```

Expected: `all section-7 criteria pass`. If `lay` median overlap ≥ 0.30, the forbidden list is not biting — inspect 10 `lay` utterances, and if the LLM is ignoring the list, lower `cap` in `forbidden_list` to 25 and regenerate `lay` only (`--out` to a scratch file, then merge).

- [ ] **Step 4: Commit the dataset**

```bash
git add winston/lab/bench/cases.jsonl winston/lab/bench/manifest.json
git commit -m "bench: generated messy utterance dataset"
```

---

### Task 10: Full scoring and first report

- [ ] **Step 1: Fast pass — the two lexical systems (minutes)**

```bash
python3 score.py --skip-resolver && python3 report.py && sed -n '1,40p' report.md
```

Expected: overall `template_rank` HitRate near 0 (documented limitation); `lexical_rank` HitRate somewhere between exp08's paraphrase floor (0.72 on template-derived paraphrases) and much lower — this is the first honest messy-input number for the current retrieval.

- [ ] **Step 2: Resolver pass (~2 h, Ollama qwen)**

```bash
rm results.jsonl                # rescore with the resolver column included
nohup python3 -u score.py > score.log 2>&1 &
```

Then `python3 report.py`. Expected: `bucket_rank` tables appear.

- [ ] **Step 3: Commit results**

```bash
git add winston/lab/bench/results.jsonl winston/lab/bench/report.md
git commit -m "bench: first full scoring run"
```

---

## Self-review

**Spec coverage:** §1 taxonomy → `STYLES`/`INTENT_LABEL` in Task 1. §2 reuse → Task 5 imports `intent_card` directly (shopper_agent needs `requests`, which is absent; the skeleton is reproduced in `build_system_prompt`). §3.1 → Task 4. §3.2 → Task 3 (all 14 covariates incl. `compat_eligible`, `compat_anchor`, `department`; `jargon` optional). §4.1–4.3 → Tasks 1–2, 5. §4.4 → `GENERATORS` in Task 5. §4.5 → `overlap`/regenerate-once in Task 5. §5 output → Tasks 5, 9. §6 incl. `question_hit` → Tasks 7, 8, 10. §7 criteria → Task 9 Step 3. §8 out of scope → nothing here touches the gate, v2's override, or `nlp_parse.py` (`department` is read via `normalize_department`, not written).

**Placeholder scan:** none.

**Type consistency:** `content_words` returns `list[str]` (Task 1) and is consumed as such in `overlap` (Task 5). `resolve(query_terms, ix, profiles, top_n)` matches `pipeline.py`. `CleanFTSAgent(catalog_path, *, pagination_mode=...)`, `.reset`, `.sessions[sid]["category"]`, `._rank(state, k)`, `.respond(...)` match the source read on 2026-08-29. `tier_of` and `normalize_department` are from `nlp_parse.py` as it exists now.
