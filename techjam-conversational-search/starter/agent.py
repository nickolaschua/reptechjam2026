from __future__ import annotations

import json
import re
import sqlite3
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Any, Sequence


try:
    import numpy as np
    from scipy import sparse
    from sklearn.feature_extraction.text import CountVectorizer
except ImportError:  # A smaller stateful SQLite fallback remains available.
    np = None
    sparse = None
    CountVectorizer = None


FIELDS = ("title", "features", "details", "description", "categories", "store", "price")
RRF_K = 60
RRF_DEPTH = 1000
TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
WS_RE = re.compile(r"\s+")
INITIAL_PREFIX = "i'm looking for "
BUYING_PREFIX = "a key requirement is:"
DISCLOSURE_PREFIX = "for that, what matters is:"
OVERRIDE_PREFIX = "actually, ignore my earlier preference. what i need is:"
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "i", "in", "is", "it", "me", "my", "of", "on", "or", "please", "some",
    "that", "the", "this", "to", "want", "with", "would", "you", "looking",
}


def _normalize(value: object) -> str:
    return WS_RE.sub(" ", str(value or "").lower()).strip()


def _text(value: object, *, details: bool = False) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        separator = ": " if details else " "
        return " ".join(f"{key}{separator}{item}" for key, item in value.items())
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return str(value)


def _product_field_text(product: dict[str, Any], field_name: str) -> str:
    if field_name == "price":
        value = product.get("price")
        return "" if value in (None, "") else f"budget around ${value} price {value}"
    return _text(product.get(field_name), details=field_name == "details")


def _terms(text: str) -> list[str]:
    return [
        token.lower()
        for token in TOKEN_RE.findall(text)
        if len(token) > 1 and token.lower() not in STOPWORDS
    ]


def _clean_constraint(value: str) -> str:
    return WS_RE.sub(" ", value).strip().rstrip(".").strip()


class Agent:
    """Stateful exact-match + BM25 retrieval agent selected by Experiment 7.

    The primary ranker implements the preregistered ``exact_stateful_bm25_rrf``
    policy. It uses only the category and constraints visible in the dialogue;
    it never reads public labels or target identifiers. If scientific Python
    packages are unavailable, the same dialogue state falls back to SQLite FTS5.
    """

    def __init__(self, catalog_path: str | Path = "data/catalog.jsonl") -> None:
        self.catalog_path = Path(catalog_path)
        self.sessions: dict[str, dict[str, Any]] = {}
        self.ids: list[str] = []
        self.products: list[dict[str, Any]] = []
        self._phrase_cache: OrderedDict[str, tuple[Any, Any]] = OrderedDict()
        self._exact_cache: OrderedDict[tuple[str, ...], tuple[Any, int, int]] = OrderedDict()
        self._bm25_cache: OrderedDict[str, Any] = OrderedDict()
        self._load_catalog()
        if np is not None and sparse is not None and CountVectorizer is not None:
            self.mode = "exact_stateful_bm25_rrf"
            self._build_research_index()
        else:
            self.mode = "stateful_sqlite_fallback"
            self._build_sqlite_index()

    def _load_catalog(self) -> None:
        with self.catalog_path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                product = json.loads(line)
                self.ids.append(str(product["parent_asin"]))
                self.products.append(product)

    def _build_research_index(self) -> None:
        assert np is not None and sparse is not None and CountVectorizer is not None
        self.id_array = np.asarray(self.ids)
        field_texts = {
            field: [_product_field_text(product, field) for product in self.products]
            for field in FIELDS
        }
        self.corpus = [
            " ".join(field_texts[field][index] for field in FIELDS)
            for index in range(len(self.products))
        ]
        self.normalized_corpus = np.asarray([_normalize(value) for value in self.corpus], dtype=object)
        self.vectorizer = CountVectorizer(
            token_pattern=r"(?u)\b[a-zA-Z0-9][a-zA-Z0-9]+\b",
            lowercase=True,
            min_df=1,
        )
        self.total_counts = self.vectorizer.fit_transform(self.corpus).tocsr()
        self.bm25 = self._bm25(self.total_counts)

    @staticmethod
    def _bm25(counts: Any, k1: float = 1.2, b: float = 0.75) -> Any:
        assert np is not None
        matrix = counts.astype(np.float32, copy=True)
        document_count = matrix.shape[0]
        document_frequency = np.diff(matrix.tocsc().indptr).astype(np.float32)
        inverse_document_frequency = np.log1p(
            (document_count - document_frequency + 0.5) / (document_frequency + 0.5)
        ).astype(np.float32)
        lengths = np.asarray(counts.sum(axis=1)).ravel().astype(np.float32)
        average_length = float(lengths.mean()) or 1.0
        rows = np.repeat(np.arange(document_count), np.diff(matrix.indptr))
        frequencies = matrix.data
        matrix.data = inverse_document_frequency[matrix.indices] * (
            frequencies * (k1 + 1.0)
        ) / (
            frequencies + k1 * (1.0 - b + b * lengths[rows] / average_length)
        )
        return matrix

    def _build_sqlite_index(self) -> None:
        self.connection = sqlite3.connect(":memory:")
        cursor = self.connection.cursor()
        cursor.execute(
            "CREATE VIRTUAL TABLE products USING fts5("
            "parent_asin UNINDEXED, title, categories, features, details, store, description, "
            "tokenize='unicode61 remove_diacritics 2')"
        )
        batch: list[tuple[str, str, str, str, str, str, str]] = []
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
            if len(batch) >= 1000:
                cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
                batch.clear()
        if batch:
            cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
        self.connection.commit()

    @staticmethod
    def _remember(cache: OrderedDict, key: object, value: object, limit: int) -> object:
        cache[key] = value
        cache.move_to_end(key)
        while len(cache) > limit:
            cache.popitem(last=False)
        return value

    def _phrase_statistics(self, phrase: str) -> tuple[Any, Any]:
        assert np is not None
        normalized = _normalize(phrase)
        cached = self._phrase_cache.get(normalized)
        if cached is not None:
            self._phrase_cache.move_to_end(normalized)
            return cached
        exact = np.fromiter(
            (normalized in document for document in self.normalized_corpus),
            dtype=np.int8,
            count=len(self.normalized_corpus),
        )
        query = self.vectorizer.transform([normalized]).tocsr()
        if query.nnz:
            query.data[:] = 1.0
            counts = (self.total_counts @ query.T).toarray().ravel()
            overlap = (counts / query.nnz).astype(np.float32)
        else:
            overlap = np.zeros(len(self.products), dtype=np.float32)
        return self._remember(self._phrase_cache, normalized, (exact, overlap), 128)  # type: ignore[return-value]

    def _exact_ranked(self, phrases: Sequence[str]) -> tuple[Any, int, int]:
        assert np is not None
        key = tuple(_normalize(value) for value in phrases if _normalize(value))
        cached = self._exact_cache.get(key)
        if cached is not None:
            self._exact_cache.move_to_end(key)
            return cached
        exact_counts = np.zeros(len(self.products), dtype=np.int16)
        overlaps = np.zeros(len(self.products), dtype=np.float32)
        for phrase in key:
            exact, overlap = self._phrase_statistics(phrase)
            exact_counts += exact
            overlaps += overlap
        scores = exact_counts.astype(np.float32) * 1000.0 + overlaps
        positive = np.flatnonzero(scores > 0)
        order = positive[np.lexsort((self.id_array[positive], -scores[positive]))][:RRF_DEPTH]
        highest = int(exact_counts.max()) if exact_counts.size else 0
        highest_tier_count = int(np.count_nonzero(exact_counts == highest)) if highest else 0
        all_phrases_count = int(np.count_nonzero(exact_counts == len(key))) if key else 0
        result = (order, all_phrases_count, highest_tier_count)
        return self._remember(self._exact_cache, key, result, 256)  # type: ignore[return-value]

    def _bm25_ranked(self, query_text: str) -> Any:
        assert np is not None
        key = _normalize(query_text)
        cached = self._bm25_cache.get(key)
        if cached is not None:
            self._bm25_cache.move_to_end(key)
            return cached
        query = self.vectorizer.transform([query_text]).tocsr()
        if query.nnz:
            query.data[:] = 1.0
            scores = (self.bm25 @ query.T).toarray().ravel().astype(np.float32)
            positive = np.flatnonzero(scores > 0)
            order = positive[np.lexsort((self.id_array[positive], -scores[positive]))][:RRF_DEPTH]
        else:
            order = np.asarray([], dtype=np.int64)
        return self._remember(self._bm25_cache, key, order, 256)

    def _rrf(self, rankings: Sequence[Sequence[int]]) -> list[int]:
        scores: dict[int, float] = defaultdict(float)
        for ranking in rankings:
            seen: set[int] = set()
            for rank, raw_index in enumerate(ranking[:RRF_DEPTH], start=1):
                index = int(raw_index)
                if index in seen:
                    continue
                seen.add(index)
                scores[index] += 1.0 / (RRF_K + rank)
        return sorted(scores, key=lambda index: (-scores[index], self.ids[index]))

    def _research_recommendations(self, category: str, constraints: list[str], top_k: int) -> list[str]:
        phrases = (category, *constraints)
        exact, all_phrases_count, highest_tier_count = self._exact_ranked(phrases)
        use_fallback = (
            not constraints
            or all_phrases_count == 0
            or highest_tier_count > top_k
        )
        if use_fallback:
            query = " ".join(phrases).strip()
            ranked = self._rrf((exact, self._bm25_ranked(query)))
        else:
            ranked = [int(index) for index in exact]
        return [self.ids[index] for index in ranked[:top_k]]

    def _sqlite_recommendations(self, category: str, constraints: list[str], top_k: int) -> list[str]:
        query = " ".join((category, *constraints)).strip()
        unique_terms = list(dict.fromkeys(_terms(query)))[:80]
        if not unique_terms:
            return []
        expression = " OR ".join(f'"{term}"' for term in unique_terms)
        rows = self.connection.execute(
            "SELECT parent_asin FROM products WHERE products MATCH ? "
            "ORDER BY bm25(products, 0.0, 6.0, 4.0, 2.5, 2.5, 1.5, 1.0) LIMIT ?",
            (expression, top_k),
        ).fetchall()
        return [str(row[0]) for row in rows]

    def reset(self, session_id: str, user_profile: dict) -> None:
        self.sessions[session_id] = {
            "category": "",
            "constraints": [],
            "override_seed": None,
            "history": [],
            "profile": user_profile,
        }

    @staticmethod
    def _append_constraint(state: dict[str, Any], value: str) -> None:
        cleaned = _clean_constraint(value)
        if not cleaned:
            return
        normalized = _normalize(cleaned)
        if all(_normalize(existing) != normalized for existing in state["constraints"]):
            state["constraints"].append(cleaned)

    def _update_state(self, state: dict[str, Any], user_message: str, turn: int) -> None:
        normalized = _normalize(user_message)
        if turn == 1 and normalized.startswith(INITIAL_PREFIX):
            body = user_message[len("I'm looking for "):]
            if ", but I'm still exploring" in body:
                category = body.split(", but I'm still exploring", 1)[0]
                remainder = ""
            else:
                category, separator, remainder = body.partition(".")
                if not separator:
                    remainder = ""
            state["category"] = _clean_constraint(category)
            remainder = remainder.strip()
            if _normalize(remainder).startswith(BUYING_PREFIX):
                self._append_constraint(state, remainder.split(":", 1)[1])
            elif remainder:
                initial_preference = _clean_constraint(remainder)
                self._append_constraint(state, initial_preference)
                state["override_seed"] = initial_preference
            return

        if normalized.startswith(DISCLOSURE_PREFIX):
            payload = user_message.split(":", 1)[1].strip().rstrip(".")
            for value in payload.split("; "):
                self._append_constraint(state, value)
            return

        if normalized.startswith(OVERRIDE_PREFIX):
            seed = state.get("override_seed")
            if seed:
                seed_normalized = _normalize(seed)
                state["constraints"] = [
                    value for value in state["constraints"]
                    if _normalize(value) != seed_normalized
                ]
            new_value = user_message.split(":", 1)[1]
            self._append_constraint(state, new_value)
            state["override_seed"] = None

    def respond(
        self,
        session_id: str,
        user_message: str,
        turn: int,
        top_k: int,
    ) -> dict:
        if session_id not in self.sessions:
            raise RuntimeError("reset must be called before respond")
        state = self.sessions[session_id]
        state["history"].append(user_message)
        self._update_state(state, user_message, turn)
        category = state["category"] or "clothing item"
        constraints = state["constraints"]
        if self.mode == "exact_stateful_bm25_rrf":
            identifiers = self._research_recommendations(category, constraints, top_k)
        else:
            identifiers = self._sqlite_recommendations(category, constraints, top_k)

        if constraints:
            message = (
                f"I am using {len(constraints)} stated preference"
                f"{'s' if len(constraints) != 1 else ''}. What other requirement should I consider?"
            )
        else:
            message = "What other requirement matters most for this product?"
        return {
            "message": message,
            "ask_attribute": "other",
            "recommendations": [{"parent_asin": identifier} for identifier in identifiers],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }
