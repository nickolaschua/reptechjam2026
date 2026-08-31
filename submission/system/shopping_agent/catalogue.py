"""Catalogue loading, exact FTS5 routing, and hard eligibility masks."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
import sqlite3
from threading import RLock
from typing import Any, Iterable

import numpy as np


FTS_AND_LIMIT = 1_000
FTS_OR_THRESHOLD = 15
KEYWORD_ROUTE_THRESHOLD = 10
BUYING_FTS_OR_THRESHOLD = 15
BUYING_KEYWORD_ROUTE_THRESHOLD = 10
BROWSING_FTS_OR_THRESHOLD = 30
BROWSING_KEYWORD_ROUTE_THRESHOLD = 15
VECTOR_FALLBACK_LIMIT = 150
FTS_BM25_WEIGHTS = (0.0, 6.0, 4.0, 2.5, 2.5, 1.5, 1.0)
SEARCH_FIELDS = ("title", "categories", "features", "details", "store", "description")
GENERIC_NEGATIVE_TERMS = frozenset({"clothing", "shoes", "jewelry"})


def contains_phrase(text: object, phrase: object) -> bool:
    """Match a normalized phrase on Unicode word boundaries."""

    normalized = _normalize(phrase)
    if not normalized:
        return False
    return re.search(rf"(?<!\w){re.escape(normalized)}(?!\w)", _normalize(text)) is not None


def _text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(f"{key} {item}" for key, item in value.items())
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return str(value)


def _normalize(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "").lower()).strip()


def standardize_department(dept_val: Any) -> str:
    """Return the canonical demographic department bucket."""

    if dept_val is None:
        return "unspecified"
    val = str(dept_val).strip()
    if not val or val.lower() in {"unspecified", '""', "none", "nan", "null"}:
        return "unspecified"
    val_lower = val.lower()
    if (
        any(sep in val_lower for sep in [",", ";", " and ", " & "])
        and any(word in val_lower for word in ["men", "women", "girl", "boy"])
    ):
        return "multi-demographic"
    if any(key in val_lower for key in ["baby", "infant", "toddler", "男嬰", "ç”·å©´"]):
        if "girl" in val_lower:
            return "baby-girls"
        if "boy" in val_lower:
            return "baby-boys"
        return "baby"
    if "unisex" in val_lower:
        if any(key in val_lower for key in ["child", "kid", "youth", "baby"]):
            return "unisex-kids"
        return "unisex-adult"
    if any(key in val_lower for key in ["girl", "daughter"]):
        return "girls"
    if any(key in val_lower for key in ["boy", "son"]):
        return "boys"
    if any(key in val_lower for key in ["women", "woman", "female", "lady", "ladies", "mom", "miss", "girlfriend", "女士", "å¥³å£«"]):
        return "women"
    if any(key in val_lower for key in ["men", "man", "male", "husband", "dad", "bridegroom"]):
        return "men"
    if any(key in val_lower for key in ["kid", "child"]):
        return "unisex-kids"
    if any(key in val_lower for key in ["adult", "teen"]):
        return "unisex-adult"
    return "unspecified"


def allowed_departments(target: str) -> set[str]:
    target_dept = _normalize(target)
    allowed = {target_dept, "unspecified", "multi-demographic"}
    if target_dept in {"men", "women"}:
        allowed.add("unisex-adult")
    elif target_dept == "boys":
        allowed.update(["unisex-kids", "baby-boys", "baby"])
    elif target_dept == "girls":
        allowed.update(["unisex-kids", "baby-girls", "baby"])
    elif target_dept in {"baby", "toddler"}:
        allowed.update(["baby", "baby-girls", "baby-boys", "unisex-kids"])
    elif target_dept == "kids":
        allowed.update(["unisex-kids", "girls", "boys", "baby", "baby-girls", "baby-boys"])
    return allowed


@dataclass(frozen=True)
class FTSRoute:
    row_indices: tuple[int, ...]
    and_count: int
    or_count: int


@dataclass(frozen=True)
class Eligibility:
    mask: np.ndarray
    hard_mask: np.ndarray
    negative_mask: np.ndarray
    hard_eligible_count: int
    negative_filtered_count: int


class Catalogue:
    """A single in-memory representation of catalogue rows and its FTS index."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.products: list[dict[str, Any]] = []
        self.ids: list[str] = []
        self.row_by_asin: dict[str, int] = {}
        self.metadata: dict[str, dict[str, Any]] = {}
        prices: list[float] = []
        departments: list[str] = []
        ratings: list[float] = []
        review_counts: list[int] = []
        brands: list[str] = []
        categories: list[set[str]] = []
        searchable: list[str] = []

        with self.path.open(encoding="utf-8") as handle:
            for row, line in enumerate(handle):
                product = json.loads(line)
                asin = str(product["parent_asin"])
                details = product.get("details") if isinstance(product.get("details"), dict) else {}
                cats = [str(value) for value in product.get("categories") or []]
                features = [str(value) for value in product.get("features") or []]
                brand = _normalize(product.get("store") or details.get("Manufacturer") or "")
                price = self._float(product.get("price"), default=np.nan)
                rating = self._float(product.get("average_rating"), default=0.0)
                reviews = self._int(product.get("rating_number"), default=0)
                search_text = _normalize(" ".join(_text(product.get(field)) for field in SEARCH_FIELDS))
                full = dict(product)
                full["parent_asin"] = asin
                self.products.append(full)
                self.ids.append(asin)
                self.row_by_asin[asin] = row
                prices.append(price)
                departments.append(standardize_department(details.get("Department")))
                ratings.append(rating)
                review_counts.append(reviews)
                brands.append(brand)
                categories.append({_normalize(value) for value in cats})
                searchable.append(search_text)
                self.metadata[asin] = {
                    "title": str(product.get("title") or ""),
                    "categories": cats,
                    "features": features,
                    "details": details,
                    "description": product.get("description") or "",
                    "store": str(product.get("store") or ""),
                    "brand": brand,
                    "price": None if np.isnan(price) else float(price),
                    "department": departments[-1],
                    "average_rating": rating,
                    "rating_number": reviews,
                    "searchable_bag": search_text,
                }

        self.ids_array = np.asarray(self.ids)
        self.prices = np.asarray(prices, dtype=np.float64)
        self.departments = np.asarray(departments)
        self.avg_ratings = np.asarray(ratings, dtype=np.float64)
        self.rating_numbers = np.asarray(review_counts, dtype=np.int64)
        self.brands = np.asarray(brands)
        self.categories = categories
        self.searchable_texts = searchable
        self._lock = RLock()
        self._closed = False
        self.connection = sqlite3.connect(":memory:", check_same_thread=False)
        self._build_fts5()

    @staticmethod
    def _float(value: object, *, default: float) -> float:
        try:
            if value in (None, ""):
                return default
            return float(str(value).replace("$", "").replace(",", "").strip())
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _int(value: object, *, default: int) -> int:
        try:
            if value in (None, ""):
                return default
            return int(value)
        except (TypeError, ValueError):
            return default

    def _build_fts5(self) -> None:
        with self._lock:
            cursor = self.connection.cursor()
            cursor.execute(
            "CREATE VIRTUAL TABLE products USING fts5("
            "parent_asin UNINDEXED, title, categories, features, details, store, description, "
            "tokenize='unicode61 remove_diacritics 2')"
        )
            batch: list[tuple[str, ...]] = []
            for product in self.products:
                batch.append((
                str(product["parent_asin"]),
                _text(product.get("title")),
                _text(product.get("categories")),
                _text(product.get("features")),
                _text(product.get("details")),
                _text(product.get("store")),
                _text(product.get("description")),
                ))
                if len(batch) >= 1_000:
                    cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
                    batch.clear()
            if batch:
                cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
            self.connection.commit()

    @staticmethod
    def _quoted_terms(terms: Iterable[object]) -> list[str]:
        result: list[str] = []
        for value in terms:
            normalized = _normalize(value).replace('"', '""')
            if normalized and normalized not in result:
                result.append(normalized)
        return result[:45]

    def fts_route(
        self,
        terms: Iterable[object],
        *,
        or_threshold: int = FTS_OR_THRESHOLD,
    ) -> FTSRoute:
        normalized = self._quoted_terms(terms)
        if not normalized:
            return FTSRoute((), 0, 0)
        if isinstance(or_threshold, bool) or int(or_threshold) < 0:
            raise ValueError("or_threshold must be a non-negative integer")
        and_expression = " AND ".join(f'"{term}"' for term in normalized)
        with self._lock:
            if self._closed:
                raise RuntimeError("catalogue is closed")
            and_rows = self.connection.execute(
                "SELECT parent_asin FROM products WHERE products MATCH ? LIMIT 1000",
                (and_expression,),
            ).fetchall()
        ids = [str(row[0]) for row in and_rows]
        or_count = 0
        if len(ids) < int(or_threshold):
            or_expression = " OR ".join(f'"{term}"' for term in normalized)
            weights = ", ".join(str(value) for value in FTS_BM25_WEIGHTS)
            with self._lock:
                or_rows = self.connection.execute(
                    f"SELECT parent_asin, bm25(products, {weights}) AS score "
                    "FROM products WHERE products MATCH ? ORDER BY score LIMIT 1000",
                    (or_expression,),
                ).fetchall()
            or_count = len(or_rows)
            seen = set(ids)
            for row in or_rows:
                asin = str(row[0])
                if asin not in seen:
                    ids.append(asin)
                    seen.add(asin)
        return FTSRoute(tuple(self.row_by_asin[asin] for asin in ids), len(and_rows), or_count)

    def eligibility(self, state: dict[str, Any]) -> Eligibility:
        hard = np.ones(len(self.ids), dtype=bool)
        price_max = float(state.get("price_max", 9999.0))
        if price_max < 9999.0:
            hard &= np.isfinite(self.prices) & (self.prices <= price_max)
        target_department = _normalize(state.get("target_department"))
        if target_department:
            hard &= np.isin(self.departments, sorted(allowed_departments(target_department)))
        min_rating = float(state.get("min_avg_rating", 0.0))
        if min_rating > 0.0:
            hard &= (self.avg_ratings >= min_rating) | (self.avg_ratings == 0.0)
        min_reviews = int(state.get("min_rating_number", 0))
        if min_reviews > 0:
            hard &= (self.rating_numbers >= min_reviews) | (self.rating_numbers == 0)
        raw_store = state.get("store")
        store_values = raw_store if isinstance(raw_store, (set, list, tuple, frozenset)) else [raw_store]
        target_stores = sorted({_normalize(value) for value in store_values if _normalize(value)})
        if target_stores:
            hard &= np.asarray(
                [any(target_store in brand for target_store in target_stores) for brand in self.brands],
                dtype=bool,
            )

        negatives = sorted({
            term
            for value in state.get("negated_terms", set())
            if (term := _normalize(value)) and term not in GENERIC_NEGATIVE_TERMS
        })
        negative_mask = np.ones(len(self.ids), dtype=bool)
        if negatives:
            negative_mask = np.asarray(
                [not any(contains_phrase(bag, term) for term in negatives) for bag in self.searchable_texts],
                dtype=bool,
            )
        combined = hard & negative_mask
        return Eligibility(
            mask=combined,
            hard_mask=hard,
            negative_mask=negative_mask,
            hard_eligible_count=int(np.count_nonzero(hard)),
            negative_filtered_count=int(np.count_nonzero(hard & ~negative_mask)),
        )

    def close(self) -> None:
        """Close the in-memory SQLite index; repeated calls are harmless."""

        with self._lock:
            if not self._closed:
                self.connection.close()
                self._closed = True

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


__all__ = [
    "BROWSING_FTS_OR_THRESHOLD", "BROWSING_KEYWORD_ROUTE_THRESHOLD",
    "BUYING_FTS_OR_THRESHOLD", "BUYING_KEYWORD_ROUTE_THRESHOLD",
    "Catalogue", "Eligibility", "FTSRoute", "FTS_AND_LIMIT", "FTS_BM25_WEIGHTS",
    "FTS_OR_THRESHOLD", "GENERIC_NEGATIVE_TERMS", "KEYWORD_ROUTE_THRESHOLD", "VECTOR_FALLBACK_LIMIT",
    "allowed_departments", "contains_phrase", "standardize_department",
]
