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
