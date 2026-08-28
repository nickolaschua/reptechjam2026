from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import logging
import math
import os
import platform
import random
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer

from .config import (
    BM25_RELATIVE_CUTOFF,
    DENSE_MAX_SEQ_LENGTH,
    DENSE_ABSOLUTE_CUTOFF,
    DENSE_RELATIVE_CUTOFF,
    MAX_TURNS,
    MODEL_ID,
    RRF_DEPTH,
    RRF_K,
    SEED,
    TOP_K,
)

FIELDS = ("title", "features", "details", "description", "categories", "store", "price")
TEXT_FIELDS = FIELDS[:-1]
TOKEN_RE = re.compile(r"[a-z0-9]+", re.I)
WS_RE = re.compile(r"\s+")
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from", "i", "in", "is",
    "it", "me", "my", "of", "on", "or", "please", "some", "that", "the", "this", "to", "want",
    "with", "would", "you", "looking", "what", "matters", "key", "requirement", "around",
}


def _load_official(repo: Path):
    kit = repo / "techjam-conversational-search-participant-kit"
    sys.path.insert(0, str(kit))
    spec = importlib.util.spec_from_file_location("techjam_official_evaluator", kit / "evaluator" / "local_evaluator.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load official evaluator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def normalize(text: object) -> str:
    return WS_RE.sub(" ", str(text or "").lower()).strip()


def tokens(text: object) -> list[str]:
    return [t.lower() for t in TOKEN_RE.findall(str(text or "")) if len(t) > 1 and t.lower() not in STOPWORDS]


def flatten(value: object, *, details: bool = False) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        sep = ": " if details else " "
        return " ".join(f"{k}{sep}{v}" for k, v in value.items())
    if isinstance(value, list):
        return " ".join(str(v) for v in value)
    return str(value)


def product_field_text(product: dict, field_name: str) -> str:
    if field_name == "price":
        value = product.get("price")
        return "" if value in (None, "") else f"budget around ${value} price {value}"
    return flatten(product.get(field_name), details=field_name == "details")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_json(data: object) -> str:
    return json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def write_json(path: Path, data: object) -> None:
    path.write_text(stable_json(data), encoding="utf-8")


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, sort_keys=True) if isinstance(v, (list, dict)) else v for k, v in row.items()})


def package_versions() -> dict[str, str | None]:
    names = ["numpy", "scipy", "scikit-learn", "pandas", "matplotlib", "sentence-transformers", "transformers", "torch"]
    result: dict[str, str | None] = {}
    for name in names:
        try:
            result[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            result[name] = None
    return result


def percentile_summary(values: Sequence[float | int]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "p25": None, "p75": None, "p90": None, "p95": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": round(float(array.mean()), 6),
        "median": round(float(np.median(array)), 6),
        "p25": round(float(np.percentile(array, 25)), 6),
        "p75": round(float(np.percentile(array, 75)), 6),
        "p90": round(float(np.percentile(array, 90)), 6),
        "p95": round(float(np.percentile(array, 95)), 6),
    }


@dataclass
class TurnState:
    sample_id: str
    scenario_type: str
    target_asin: str
    turn: int
    message: str
    category: str
    active_constraints: tuple[str, ...]
    disclosed_constraints: tuple[str, ...]
    override_applied: bool
    oracle_card: dict

    @property
    def state_query(self) -> str:
        return " ".join((self.category, *self.active_constraints)).strip()

    @property
    def phrases(self) -> tuple[str, ...]:
        return (self.category, *self.active_constraints)


class LexicalIndex:
    def __init__(self, ids: Sequence[str], field_texts: dict[str, list[str]], cache_dir: Path, input_hash: str, logger: logging.Logger):
        self.ids = np.asarray(ids)
        self.id_to_idx = {value: i for i, value in enumerate(ids)}
        self.field_texts = field_texts
        self.cache_dir = cache_dir
        self.logger = logger
        self.cache_key = input_hash[:16]
        self.vectorizer: CountVectorizer
        self.field_counts: dict[str, sparse.csr_matrix]
        self.field_bm25: dict[str, sparse.csr_matrix]
        self.bm25: sparse.csr_matrix
        self._result_cache: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
        self._build_or_load()

    def _build_or_load(self) -> None:
        import joblib

        target = self.cache_dir / f"lexical_{self.cache_key}.joblib"
        if target.exists():
            started = time.perf_counter()
            payload = joblib.load(target)
            self.vectorizer = payload["vectorizer"]
            self.field_counts = payload["field_counts"]
            self.field_bm25 = payload["field_bm25"]
            self.bm25 = payload["bm25"]
            self.field_aware = payload["field_aware"]
            self.logger.info("Loaded lexical cache %s in %.2fs", target, time.perf_counter() - started)
            return
        started = time.perf_counter()
        combined = [" ".join(self.field_texts[field][i] for field in FIELDS) for i in range(len(self.ids))]
        self.vectorizer = CountVectorizer(token_pattern=r"(?u)\b[a-zA-Z0-9][a-zA-Z0-9]+\b", lowercase=True, min_df=1)
        total_counts = self.vectorizer.fit_transform(combined).tocsr()
        self.field_counts = {field: self.vectorizer.transform(self.field_texts[field]).tocsr() for field in FIELDS}
        weights = {"title": 6.0, "categories": 4.0, "features": 2.5, "details": 2.5, "store": 1.5, "description": 1.0, "price": 1.0}
        self.field_bm25 = {field: self._bm25(matrix) for field, matrix in self.field_counts.items()}
        self.bm25 = self._bm25(total_counts)
        self.field_aware = sum((self.field_bm25[field] * weights[field] for field in FIELDS), start=sparse.csr_matrix(self.bm25.shape))
        payload = {
            "vectorizer": self.vectorizer, "field_counts": self.field_counts,
            "field_bm25": self.field_bm25, "bm25": self.bm25, "field_aware": self.field_aware,
        }
        joblib.dump(payload, target, compress=3)
        self.logger.info("Built lexical cache %s in %.2fs", target, time.perf_counter() - started)

    def _bm25(self, counts: sparse.csr_matrix, k1: float = 1.2, b: float = 0.75) -> sparse.csr_matrix:
        matrix = counts.astype(np.float32, copy=True)
        n_docs = matrix.shape[0]
        df = np.diff(matrix.tocsc().indptr).astype(np.float32)
        idf = np.log1p((n_docs - df + 0.5) / (df + 0.5)).astype(np.float32)
        lengths = np.asarray(counts.sum(axis=1)).ravel().astype(np.float32)
        avgdl = float(lengths.mean()) or 1.0
        rows = np.repeat(np.arange(n_docs), np.diff(matrix.indptr))
        tf = matrix.data
        matrix.data = idf[matrix.indices] * (tf * (k1 + 1.0)) / (tf + k1 * (1.0 - b + b * lengths[rows] / avgdl))
        return matrix

    def _query_vector(self, text: str) -> sparse.csr_matrix:
        q = self.vectorizer.transform([text]).tocsr()
        if q.nnz:
            q.data[:] = 1.0
        return q

    def scores(self, text: str, method: str = "bm25") -> np.ndarray:
        matrix = self.bm25 if method == "bm25" else self.field_aware if method == "field_aware" else self.field_bm25[method.removeprefix("field:")]
        q = self._query_vector(text)
        return (matrix @ q.T).toarray().ravel().astype(np.float32)

    def ranked(self, text: str, method: str = "bm25", depth: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        key = (method, normalize(text))
        if key not in self._result_cache:
            scores = self.scores(text, method)
            positive = np.flatnonzero(scores > 0)
            order = positive[np.lexsort((self.ids[positive], -scores[positive]))]
            self._result_cache[key] = order, scores[order]
        order, scores = self._result_cache[key]
        return (order[:depth], scores[:depth]) if depth else (order, scores)

    def target_rank(self, text: str, target_asin: str, method: str = "bm25") -> int | None:
        order, _ = self.ranked(text, method)
        location = np.flatnonzero(order == self.id_to_idx[target_asin])
        return int(location[0] + 1) if location.size else None

    def cutoff_count(self, text: str, method: str = "bm25", relative: float = BM25_RELATIVE_CUTOFF) -> int:
        _, scores = self.ranked(text, method)
        if not scores.size:
            return 0
        return int(np.count_nonzero(scores >= scores[0] * relative))

    def all_token_mask(self, phrases: Sequence[str]) -> np.ndarray:
        mask = np.ones(len(self.ids), dtype=bool)
        for phrase in phrases:
            q = self._query_vector(phrase)
            if not q.nnz:
                return np.zeros(len(self.ids), dtype=bool)
            matches = (self.field_counts_total @ q.T).toarray().ravel() if hasattr(self, "field_counts_total") else None
            if matches is None:
                total = sum(self.field_counts.values(), start=sparse.csr_matrix(self.bm25.shape))
                self.field_counts_total = total
                matches = (total @ q.T).toarray().ravel()
            mask &= matches >= q.nnz
        return mask


class DenseIndex:
    def __init__(self, ids: Sequence[str], texts: Sequence[str], cache_dir: Path, input_hash: str, logger: logging.Logger):
        self.ids = np.asarray(ids)
        self.id_to_idx = {value: i for i, value in enumerate(ids)}
        self.texts = texts
        self.cache_dir = cache_dir
        self.input_hash = input_hash
        self.logger = logger
        self.model = None
        self.embeddings: np.ndarray | None = None
        self._cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._query_embeddings: dict[str, np.ndarray] = {}

    def _load_model(self):
        if self.model is not None:
            return self.model
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as exc:
            raise RuntimeError(
                "sentence-transformers could not import; repair the installed torch/torchvision combination"
            ) from exc
        self.model = SentenceTransformer(MODEL_ID, device="cpu", cache_folder=str(self.cache_dir / "models"))
        self.model.max_seq_length = DENSE_MAX_SEQ_LENGTH
        return self.model

    def ensure(self) -> None:
        if self.embeddings is not None:
            return
        cache = self.cache_dir / f"dense_{self.input_hash[:16]}_minilm_seq{DENSE_MAX_SEQ_LENGTH}.npy"
        if cache.exists():
            self.embeddings = np.load(cache, mmap_mode="r")
            self.logger.info("Loaded dense embeddings from %s", cache)
            return
        model = self._load_model()
        started = time.perf_counter()
        values = model.encode(
            list(self.texts), batch_size=256, show_progress_bar=True,
            convert_to_numpy=True, normalize_embeddings=True,
        ).astype(np.float32)
        np.save(cache, values)
        self.embeddings = np.load(cache, mmap_mode="r")
        self.logger.info("Encoded %d products in %.2fs", len(self.texts), time.perf_counter() - started)

    def ranked(self, text: str, depth: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        key = normalize(text)
        if key not in self._cache:
            self.ensure()
            query = self._query_embeddings.get(key)
            if query is None:
                query = self._load_model().encode([text], convert_to_numpy=True, normalize_embeddings=True)[0].astype(np.float32)
                self._query_embeddings[key] = query
            scores = np.asarray(self.embeddings @ query)
            order = np.lexsort((self.ids, -scores))
            self._cache[key] = order, scores[order]
        order, scores = self._cache[key]
        return (order[:depth], scores[:depth]) if depth else (order, scores)

    def preload_queries(self, texts: Iterable[str]) -> None:
        unique: dict[str, str] = {}
        for value in texts:
            unique.setdefault(normalize(value), value)
        missing = [(key, value) for key, value in unique.items() if key not in self._query_embeddings]
        if not missing:
            return
        self.ensure()
        vectors = self._load_model().encode(
            [value for _, value in missing], batch_size=128, show_progress_bar=False,
            convert_to_numpy=True, normalize_embeddings=True,
        ).astype(np.float32)
        for (key, _), vector in zip(missing, vectors):
            self._query_embeddings[key] = vector

    def target_rank(self, text: str, target_asin: str) -> int:
        order, _ = self.ranked(text)
        return int(np.flatnonzero(order == self.id_to_idx[target_asin])[0] + 1)

    def cutoff_count(self, text: str) -> int:
        _, scores = self.ranked(text)
        threshold = max(DENSE_ABSOLUTE_CUTOFF, float(scores[0]) * DENSE_RELATIVE_CUTOFF)
        return int(np.count_nonzero(scores >= threshold))


@dataclass
class Harness:
    repo: Path
    catalog_path: Path
    public_path: Path
    results_dir: Path
    logger: logging.Logger
    official: Any = field(init=False)
    products: list[dict] = field(init=False)
    samples: list[dict] = field(init=False)
    ids: list[str] = field(init=False)
    id_to_idx: dict[str, int] = field(init=False)
    product_by_id: dict[str, dict] = field(init=False)
    categories: dict[str, list[str]] = field(init=False)
    cards: dict[str, dict] = field(init=False)
    behaviors: dict[str, dict] = field(init=False)
    traces: list[TurnState] = field(init=False)
    catalog_hash: str = field(init=False)
    public_hash: str = field(init=False)
    field_texts: dict[str, list[str]] = field(init=False)
    corpus: list[str] = field(init=False)
    lexical: LexicalIndex = field(init=False)
    dense: DenseIndex = field(init=False)
    _hybrid_cache: dict[str, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        random.seed(SEED)
        np.random.seed(SEED)
        self.official = _load_official(self.repo)
        self.catalog_hash, self.public_hash = sha256(self.catalog_path), sha256(self.public_path)
        self.products = self.official.load_jsonl(self.catalog_path)
        self.samples = self.official.load_jsonl(self.public_path)
        self.ids = [str(p["parent_asin"]) for p in self.products]
        self.id_to_idx = {value: i for i, value in enumerate(self.ids)}
        self.product_by_id = {str(p["parent_asin"]): p for p in self.products}
        self.categories = {str(p["parent_asin"]): [str(v) for v in p.get("categories") or []] for p in self.products}
        self.cards, self.behaviors = {}, {}
        for sample in self.samples:
            target = str(sample["ground_truth"]["parent_asin"])
            card, behavior = self.official.materialize_hidden_fields(sample, self.product_by_id)
            self.cards[sample["sample_id"]] = card
            self.behaviors[sample["sample_id"]] = behavior
        self.field_texts = {field: [product_field_text(p, field) for p in self.products] for field in FIELDS}
        self.corpus = [" ".join(self.field_texts[f][i] for f in FIELDS) for i in range(len(self.products))]
        cache_dir = self.results_dir / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        (cache_dir / "README.md").write_text(
            "# Reproducibility cache\n\nGenerated lexical indexes, model files, and dense embeddings live here. "
            "They are inputs to reruns, not analytical results.\n", encoding="utf-8",
        )
        self.lexical = LexicalIndex(self.ids, self.field_texts, cache_dir, self.catalog_hash, self.logger)
        dense_corpus = [" ".join(self.field_texts[f][i] for f in ("title", "categories", "store", "features", "details", "description", "price")) for i in range(len(self.products))]
        self.dense = DenseIndex(self.ids, dense_corpus, cache_dir, self.catalog_hash, self.logger)
        self.traces = self._make_traces()
        self.validate_inputs()

    def _make_traces(self) -> list[TurnState]:
        rows: list[TurnState] = []
        for sample in self.samples:
            sid = sample["sample_id"]
            target = str(sample["ground_truth"]["parent_asin"])
            card, behavior = self.cards[sid], self.behaviors[sid]
            effective = {**sample, "intent_card": card, "behavior": behavior}
            category = self.official.coarse_category(self.categories[target])
            disclosed: set[str] = set()
            ordered_disclosed: list[str] = []
            active: list[str] = []
            boundary_used = False
            override_applied = sample["scenario_type"] != "intent_override"
            message = self.official.initial_message(effective, category, disclosed)
            ordered_disclosed.extend(value for value in [*card.get("hard_constraints", []), *card.get("soft_preferences", [])] if value in disclosed)
            if sample["scenario_type"] == "buying" and ordered_disclosed:
                active.extend(ordered_disclosed)
            elif sample["scenario_type"] == "intent_override":
                old = str(behavior["override"]["old_value"])
                active.append(old)
            for turn in range(1, MAX_TURNS + 1):
                rows.append(TurnState(
                    sid, str(sample["scenario_type"]), target, turn, message, category,
                    tuple(active), tuple(ordered_disclosed), override_applied, card,
                ))
                if turn == MAX_TURNS:
                    continue
                override = behavior.get("override", {})
                if not override_applied and turn + 1 == int(override.get("turn", 3)):
                    override_applied = True
                    old, new = str(override.get("old_value", "")), str(override.get("new_value", ""))
                    active = [v for v in active if v != old]
                    if new:
                        disclosed.add(new)
                        if new not in ordered_disclosed:
                            ordered_disclosed.append(new)
                        if new not in active:
                            active.append(new)
                    message = str(override.get("message", "Actually, please ignore my earlier preference."))
                else:
                    before = set(disclosed)
                    message, boundary_used = self.official.customer_reply(effective, "other", disclosed, boundary_used)
                    additions = [v for v in [*card.get("hard_constraints", []), *card.get("soft_preferences", [])] if v in disclosed and v not in before]
                    for value in additions:
                        if value not in ordered_disclosed:
                            ordered_disclosed.append(value)
                        if value not in active:
                            active.append(value)
        return rows

    def validate_inputs(self) -> None:
        scenarios = Counter(s["scenario_type"] for s in self.samples)
        if len(self.products) != 50_000 or len(self.samples) != 200:
            raise ValueError(f"Expected 50,000 products and 200 sessions, got {len(self.products)} and {len(self.samples)}")
        if set(scenarios) != {"buying", "browsing", "intent_override", "boundary"}:
            raise ValueError(f"Scenario coverage mismatch: {scenarios}")
        if len(self.traces) != len(self.samples) * MAX_TURNS:
            raise ValueError("Full diagnostic traces are incomplete")

    def hybrid_ranked(self, text: str, depth: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        key = normalize(text)
        if key not in self._hybrid_cache:
            lexical, _ = self.lexical.ranked(text, "field_aware", RRF_DEPTH)
            dense, _ = self.dense.ranked(text, RRF_DEPTH)
            scores: dict[int, float] = defaultdict(float)
            for rank, idx in enumerate(lexical, 1):
                scores[int(idx)] += 1.0 / (RRF_K + rank)
            for rank, idx in enumerate(dense, 1):
                scores[int(idx)] += 1.0 / (RRF_K + rank)
            candidates = np.fromiter(scores, dtype=np.int64)
            values = np.fromiter((scores[int(idx)] for idx in candidates), dtype=np.float64)
            order = np.lexsort((self.lexical.ids[candidates], -values))
            self._hybrid_cache[key] = candidates[order], values[order]
        indices, scores = self._hybrid_cache[key]
        return (indices[:depth], scores[:depth]) if depth else (indices, scores)

    def hybrid_target_rank(self, text: str, target_asin: str) -> int | None:
        order, _ = self.hybrid_ranked(text)
        location = np.flatnonzero(order == self.id_to_idx[target_asin])
        return int(location[0] + 1) if location.size else None

    def exact_scores(self, phrases: Sequence[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        normalized_phrases = [normalize(p) for p in phrases if normalize(p)]
        exact = np.zeros(len(self.products), dtype=np.int16)
        overlap = np.zeros(len(self.products), dtype=np.float32)
        all_tokens = np.ones(len(self.products), dtype=bool)
        norm_corpus = getattr(self, "_norm_corpus", None)
        if norm_corpus is None:
            self._norm_corpus = np.asarray([normalize(v) for v in self.corpus], dtype=object)
            norm_corpus = self._norm_corpus
        total = getattr(self, "_total_counts", None)
        if total is None:
            self._total_counts = sum(self.lexical.field_counts.values(), start=sparse.csr_matrix(self.lexical.bm25.shape))
            total = self._total_counts
        cache = getattr(self, "_phrase_stats", None)
        if cache is None:
            self._phrase_stats = {}
            cache = self._phrase_stats
        for phrase in normalized_phrases:
            if phrase not in cache:
                phrase_exact = np.fromiter((phrase in doc for doc in norm_corpus), dtype=np.int8, count=len(norm_corpus))
                q = self.lexical._query_vector(phrase)
                if q.nnz:
                    counts = (total @ q.T).toarray().ravel()
                    phrase_overlap = (counts / q.nnz).astype(np.float32)
                    phrase_all = counts >= q.nnz
                else:
                    phrase_overlap = np.zeros(len(self.products), dtype=np.float32)
                    phrase_all = np.zeros(len(self.products), dtype=bool)
                cache[phrase] = phrase_exact, phrase_overlap, phrase_all
            phrase_exact, phrase_overlap, phrase_all = cache[phrase]
            exact += phrase_exact
            overlap += phrase_overlap
            all_tokens &= phrase_all
        return exact, overlap, all_tokens

    def exact_ranked(self, phrases: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
        key = tuple(normalize(value) for value in phrases)
        cache = getattr(self, "_exact_rank_cache", None)
        if cache is None:
            self._exact_rank_cache = {}
            cache = self._exact_rank_cache
        if key in cache:
            return cache[key]
        exact, overlap, _ = self.exact_scores(phrases)
        scores = exact.astype(np.float32) * 1000.0 + overlap
        positive = np.flatnonzero(scores > 0)
        order = positive[np.lexsort((self.lexical.ids[positive], -scores[positive]))]
        cache[key] = order, scores[order]
        return cache[key]

    def exact_rank(self, phrases: Sequence[str], target_asin: str) -> int | None:
        exact, overlap, _ = self.exact_scores(phrases)
        scores = exact.astype(np.float32) * 1000.0 + overlap
        positive = np.flatnonzero(scores > 0)
        if self.id_to_idx[target_asin] not in positive:
            return None
        order = positive[np.lexsort((self.lexical.ids[positive], -scores[positive]))]
        return int(np.flatnonzero(order == self.id_to_idx[target_asin])[0] + 1)

    def trace_by_session(self) -> dict[str, list[TurnState]]:
        grouped: dict[str, list[TurnState]] = defaultdict(list)
        for row in self.traces:
            grouped[row.sample_id].append(row)
        return grouped


def make_logger(results_dir: Path) -> logging.Logger:
    results_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("techjam_experiments")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()
    handler = logging.FileHandler(results_dir / "run_all.log", mode="a", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    logger.addHandler(logging.StreamHandler(sys.stdout))
    return logger


def experiment_logger(root_logger: logging.Logger, directory: Path) -> logging.Logger:
    directory.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"techjam_experiments.{directory.name}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()
    file_handler = logging.FileHandler(directory / "run.log", mode="w", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(file_handler)
    for handler in root_logger.handlers:
        logger.addHandler(handler)
    logger.info("Environment: python=%s platform=%s packages=%s model=%s device=cpu seed=%d", sys.version.split()[0], platform.platform(), package_versions(), MODEL_ID, SEED)
    return logger


def rank_metrics(session_rows: Sequence[dict]) -> dict:
    n = len(session_rows)
    if not n:
        return {"sample_count": 0, "hit_rate_at_10": 0.0, "mrr": 0.0, "mttc": None, "efficiency": 0.0, "technical_score": 0.0}
    hit_rate = sum(bool(r["hit"]) for r in session_rows) / n
    mrr = sum(float(r["reciprocal_rank"]) for r in session_rows) / n
    mttc = sum(r["first_hit_turn"] if r["first_hit_turn"] is not None else MAX_TURNS + 1 for r in session_rows) / n
    efficiency = max(0.0, min(1.0, (11.0 - mttc) / 10.0))
    score = 0.50 * hit_rate + 0.30 * mrr + 0.20 * efficiency
    return {
        "sample_count": n, "hit_rate_at_10": round(hit_rate, 6), "mrr": round(mrr, 6),
        "mttc": round(mttc, 6), "efficiency": round(efficiency, 6), "technical_score": round(score, 6),
    }


def replay_policy(harness: Harness, ranker, width: int = TOP_K, abstain_below: float | None = None) -> tuple[list[dict], dict]:
    sessions: list[dict] = []
    for sid, turns in harness.trace_by_session().items():
        first_hit, best_rank = None, None
        for state in turns:
            indices, scores = ranker(state)
            if abstain_below is not None and (not scores.size or float(scores[0]) < abstain_below):
                shown = np.asarray([], dtype=np.int64)
            else:
                shown = indices[:width]
            location = np.flatnonzero(shown == harness.id_to_idx[state.target_asin])
            if state.override_applied and location.size:
                first_hit, best_rank = state.turn, int(location[0] + 1)
                break
        sessions.append({
            "sample_id": sid, "scenario_type": turns[0].scenario_type, "hit": first_hit is not None,
            "first_hit_turn": first_hit, "best_rank": best_rank,
            "reciprocal_rank": 0.0 if best_rank is None else 1.0 / best_rank,
        })
    overall = rank_metrics(sessions)
    overall["scenario_metrics"] = {
        scenario: rank_metrics([row for row in sessions if row["scenario_type"] == scenario])
        for scenario in sorted({row["scenario_type"] for row in sessions})
    }
    return sessions, overall


def manifest_base(harness: Harness, command: str) -> dict:
    def git(*args: str) -> str:
        try:
            return subprocess.check_output(["git", *args], cwd=harness.repo, text=True, stderr=subprocess.STDOUT).strip()
        except Exception as exc:
            return f"unavailable: {exc}"
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "inputs": {
            str(harness.catalog_path.relative_to(harness.repo)): {"sha256": harness.catalog_hash, "rows": len(harness.products)},
            str(harness.public_path.relative_to(harness.repo)): {"sha256": harness.public_hash, "rows": len(harness.samples)},
        },
        "git": {"commit": git("rev-parse", "HEAD"), "status_porcelain": git("status", "--porcelain")},
        "environment": {
            "python": sys.version, "platform": platform.platform(), "processor": platform.processor(),
            "cpu_count": os.cpu_count(), "packages": package_versions(), "model_id": MODEL_ID,
            "dense_max_seq_length": DENSE_MAX_SEQ_LENGTH, "device": "cpu",
        },
        "commands": [command],
    }
