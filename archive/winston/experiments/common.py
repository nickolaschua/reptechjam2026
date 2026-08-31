"""Shared catalog loading, indexes and scoring primitives for the winston experiments.

Every experiment imports from here so the numbers are comparable across runs.
Indexes are cached to .cache/ because building IDF over 50k products takes ~30s.
"""
from __future__ import annotations

import collections
import json
import math
import pickle
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]          # repository root after archival
KIT = ROOT / "techjam-conversational-search"
CATALOG = KIT / "data" / "catalog.jsonl"
PUBLIC_SET = KIT / "data" / "public_set.jsonl"
RESULTS = Path(__file__).resolve().parent / "results"
CACHE = Path(__file__).resolve().parent / ".cache"
sys.path.insert(0, str(KIT))

from evaluator.local_evaluator import (  # noqa: E402
    catalog_index, coarse_category, load_jsonl, _flatten_values, intent_card,
)

TOKEN_RE = re.compile(r"[a-z0-9]+")
FIELDS = ("title", "categories", "store", "features", "description", "details")
N_CATALOG = 50_000


def field_texts(product: dict) -> dict[str, str]:
    return {
        "title": str(product.get("title") or "").lower(),
        "categories": " ".join(_flatten_values(product.get("categories"))).lower(),
        "store": str(product.get("store") or "").lower(),
        "features": " ".join(_flatten_values(product.get("features"))).lower(),
        "description": " ".join(_flatten_values(product.get("description"))).lower(),
        "details": " ".join(_flatten_values(product.get("details"))).lower(),
    }


class Index:
    """Catalog plus the derived structures every experiment needs."""

    def __init__(self, catalog_path: Path = CATALOG) -> None:
        self.ids, self.categories, self.products = catalog_index(catalog_path)
        self.fields = {a: field_texts(p) for a, p in self.products.items()}
        self.text = {a: " | ".join(f[k] for k in FIELDS) for a, f in self.fields.items()}
        self.tokens = {a: set(TOKEN_RE.findall(t)) for a, t in self.text.items()}
        doc_freq: collections.Counter = collections.Counter()
        for toks in self.tokens.values():
            doc_freq.update(toks)
        self.doc_freq = doc_freq
        self.idf = {t: math.log(N_CATALOG / c) for t, c in doc_freq.items()}
        self.popularity = {
            a: math.log1p(p.get("rating_number") or 0) for a, p in self.products.items()
        }
        self.buckets: dict[str, list[str]] = collections.defaultdict(list)
        for a in self.products:
            self.buckets[coarse_category(self.categories[a]).lower()].append(a)
        self.buckets = dict(self.buckets)
        self.bucket_of = {
            a: coarse_category(self.categories[a]).lower() for a in self.products
        }
        self.bucket_tokens = {k: set(TOKEN_RE.findall(k)) for k in self.buckets}

    def samples(self) -> list[dict]:
        return load_jsonl(PUBLIC_SET)

    def idf_of(self, phrase: str) -> float:
        return sum(self.idf.get(t, 0.0) for t in TOKEN_RE.findall(phrase.lower()))

    def resolve_bucket(self, phrase: str, top_n: int = 3) -> list[str]:
        """Score every bucket by IDF-weighted overlap. No anchor phrase required."""
        q = set(TOKEN_RE.findall(phrase.lower()))
        if not q:
            return []
        scored = sorted(
            self.buckets,
            key=lambda k: -sum(self.idf.get(t, 0.0) for t in self.bucket_tokens[k] & q)
            / (1 + 0.15 * len(self.bucket_tokens[k] - q)),
        )
        return scored[:top_n]


_INDEX: Index | None = None


def get_index() -> Index:
    """Build once per process; cache to disk across processes."""
    global _INDEX
    if _INDEX is not None:
        return _INDEX
    CACHE.mkdir(exist_ok=True)
    cached = CACHE / f"index-{CATALOG.stat().st_size}.pkl"
    if cached.exists():
        with cached.open("rb") as fh:
            _INDEX = pickle.load(fh)
    else:
        _INDEX = Index()
        with cached.open("wb") as fh:
            pickle.dump(_INDEX, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return _INDEX


def write_result(name: str, payload: dict) -> Path:
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"  -> {path.relative_to(ROOT)}")
    return path
