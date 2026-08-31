"""Winston's completed lexical category resolver over the live catalogue.

The resolver is deliberately advisory: it supplies candidate buckets and a
relative top-two margin for ambiguity handling.  It never filters or reranks
products.  All indexes are derived once from the already-loaded ``Catalogue``;
no second catalogue read or experiment cache is used.
"""

from __future__ import annotations

import collections
import math
import re
from types import MappingProxyType
from typing import Iterable

try:
    from .catalogue import Catalogue
except ImportError:  # pragma: no cover - direct script compatibility
    from catalogue import Catalogue


TOKEN_RE = re.compile(r"[a-z0-9]+")
_PROFILE_MIN_FRACTION = 0.02
_FALLBACK_IDF = 4.0


def coarse_category(values: Iterable[object]) -> str:
    """Match the evaluator's bucket naming without importing evaluator code."""

    excluded = {
        "clothing",
        "clothing shoes & jewelry",
        "clothing, shoes & jewelry",
    }
    cleaned: list[str] = []
    for value in values:
        for part in str(value).split(","):
            part = part.strip()
            if part and part.lower() not in excluded:
                cleaned.append(part)
    return " ".join(cleaned[-2:]).lower() if cleaned else "clothing item"


def catalog_bucket_set(catalogue: Catalogue) -> frozenset[str]:
    return frozenset(
        coarse_category(product.get("categories") or ())
        for product in catalogue.products
    )


def _stem(token: str) -> str:
    return token[:-1] if len(token) > 3 and token.endswith("s") else token


class CategoryResolver:
    """IDF label/content resolver measured by Winston's soft-slot arm."""

    def __init__(self, catalogue: Catalogue) -> None:
        buckets: dict[str, list[int]] = collections.defaultdict(list)
        title_tokens: list[set[str]] = []

        for row, product in enumerate(catalogue.products):
            bucket = coarse_category(product.get("categories") or ())
            buckets[bucket].append(row)
            title_tokens.append(set(TOKEN_RE.findall(str(product.get("title") or "").lower())))

        self._catalogue = catalogue
        self._document_count = max(1, len(catalogue.products))
        self._idf: dict[str, float] = {}
        self._missing_idf: set[str] = set()
        self._buckets = {key: tuple(rows) for key, rows in buckets.items()}
        self._bucket_tokens = {
            key: frozenset(_stem(token) for token in TOKEN_RE.findall(key))
            for key in self._buckets
        }

        profiles: dict[str, MappingProxyType] = {}
        for key, members in self._buckets.items():
            counts: collections.Counter[str] = collections.Counter()
            for row in members:
                counts.update(title_tokens[row])
            size = len(members)
            profile: dict[str, float] = {}
            for token, count in counts.items():
                if count >= max(2, _PROFILE_MIN_FRACTION * size):
                    stemmed = _stem(token)
                    profile[stemmed] = max(profile.get(stemmed, 0.0), count / size)
            profiles[key] = MappingProxyType(profile)
        self._profiles = MappingProxyType(profiles)
        self.bucket_names = frozenset(self._buckets)
        self.catalog_stores = frozenset(
            str(product.get("store") or "").strip().lower()
            for product in catalogue.products
            if str(product.get("store") or "").strip()
        )

    def _idf_of(self, token: str) -> float:
        return self._idf.get(token, self._idf.get(token + "s", _FALLBACK_IDF))

    def _prepare_idf(self, tokens: Iterable[str]) -> None:
        requested = {
            candidate
            for token in tokens
            for candidate in (token, token + "s")
            if candidate not in self._idf and candidate not in self._missing_idf
        }
        frequencies = self._catalogue.document_frequencies(requested)
        for token in requested:
            count = frequencies.get(token)
            if count:
                self._idf[token] = math.log(self._document_count / count)
            else:
                self._missing_idf.add(token)

    def resolve(
        self,
        query_terms: Iterable[object],
        *,
        top_n: int = 3,
    ) -> tuple[tuple[str, ...], float]:
        """Return ranked bucket names and the unrounded relative top-two margin."""

        query = [
            _stem(token)
            for token in TOKEN_RE.findall(" ".join(str(term) for term in query_terms).lower())
        ]
        if not query or top_n <= 0:
            return (), 0.0
        self._prepare_idf(query)

        query_set = set(query)
        scored: list[tuple[float, str]] = []
        for key in self._buckets:
            bucket_tokens = self._bucket_tokens[key]
            label = sum(self._idf_of(token) for token in query if token in bucket_tokens) / (
                1 + 0.15 * len(bucket_tokens - query_set)
            )
            profile = self._profiles[key]
            content = sum(
                profile.get(token, 0.0) * self._idf_of(token)
                for token in query
            )
            scored.append((label + content, key))
        scored.sort(reverse=True)

        top = scored[0][0]
        runner_up = scored[1][0] if len(scored) > 1 else 0.0
        confidence = 0.0 if top <= 0 else (top - runner_up) / top
        return tuple(key for _, key in scored[:top_n]), confidence


__all__ = [
    "CategoryResolver",
    "TOKEN_RE",
    "catalog_bucket_set",
    "coarse_category",
]
