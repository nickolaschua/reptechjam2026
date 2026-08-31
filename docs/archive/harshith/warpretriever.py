from __future__ import annotations

import hashlib
import json
import math
import os
import pickle
import re
from collections import Counter, defaultdict
from pathlib import Path


TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
CLAUSE_SPLIT_RE = re.compile(r"\s*;\s*|\s{2,}")
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "i", "in", "is", "it", "me", "my", "of", "on", "or", "please", "some",
    "that", "the", "this", "to", "want", "with", "would", "you", "looking",
}
RERANK_POOL = 250
SEARCH_FIELDS = ("title", "categories", "features", "details", "store", "description")
FIELD_WEIGHTS = {
    "title": 6.0,
    "categories": 4.0,
    "features": 2.5,
    "details": 2.5,
    "store": 1.5,
    "description": 1.0,
}


def _text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(f"{key} {item}" for key, item in value.items())
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return str(value)


def terms(text: str) -> list[str]:
    return [
        token.lower()
        for token in TOKEN_RE.findall(text)
        if len(token) > 1 and token.lower() not in STOPWORDS
    ]


def product_text(product: dict) -> str:
    return " ".join(_text(product.get(field)) for field in SEARCH_FIELDS).strip()


def query_clauses(query: str) -> list[str]:
    clauses: list[str] = []
    for clause in CLAUSE_SPLIT_RE.split(query):
        cleaned = " ".join(clause.strip(" :;,.-").split())
        if cleaned:
            clauses.append(cleaned.lower())
    return list(dict.fromkeys(clauses))


def lexical_rerank_score(query: str, text: str, base_rank: int) -> float:
    text_lower = text.lower()
    q_terms = terms(query)
    if not q_terms:
        return -float(base_rank)
    term_counts = Counter(q_terms)
    matched_terms = sum(count for term, count in term_counts.items() if term in text_lower)
    rare_matches = sum(1.0 / math.sqrt(count) for term, count in term_counts.items() if term in text_lower)
    phrase_score = 0.0
    for clause in query_clauses(query):
        clause_terms = terms(clause)
        if not clause_terms:
            continue
        if clause in text_lower:
            phrase_score += 16.0 + 2.0 * len(clause_terms)
        else:
            coverage = sum(1 for term in set(clause_terms) if term in text_lower) / len(set(clause_terms))
            phrase_score += coverage
    coverage = matched_terms / max(len(q_terms), 1)
    return 10.0 * coverage + 2.0 * rare_matches + phrase_score - 0.03 * base_rank


def _catalog_fingerprint(catalog_path: Path) -> str:
    stat = catalog_path.stat()
    value = f"{catalog_path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


class WARPRetriever:
    """Best local retriever so far: weighted lexical WARP-style index + rerank."""

    def __init__(self, catalog_path: str | Path, index_dir: str | Path | None = None) -> None:
        self.catalog_path = Path(catalog_path)
        self.index_dir = Path(index_dir or os.getenv("WARP_INDEX_DIR", "data/warp_index"))
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.index_dir / f"{_catalog_fingerprint(self.catalog_path)}.pkl"
        self.parent_asins: list[str] = []
        self.doc_lengths: list[float] = []
        self.postings: dict[str, list[tuple[int, float]]] = {}
        self.product_texts: list[str] = []
        self.idf: dict[str, float] = {}
        self.avg_doc_length = 1.0
        if not self._load_index():
            self._build_index()
            self._save_index()

    def _load_index(self) -> bool:
        if not self.index_path.exists():
            return False
        with self.index_path.open("rb") as handle:
            payload = pickle.load(handle)
        self.parent_asins = payload["parent_asins"]
        self.doc_lengths = payload["doc_lengths"]
        self.postings = payload["postings"]
        self.product_texts = payload.get("product_texts") or self._load_product_texts()
        self.idf = payload["idf"]
        self.avg_doc_length = payload["avg_doc_length"]
        return True

    def _save_index(self) -> None:
        payload = {
            "parent_asins": self.parent_asins,
            "doc_lengths": self.doc_lengths,
            "postings": self.postings,
            "product_texts": self.product_texts,
            "idf": self.idf,
            "avg_doc_length": self.avg_doc_length,
        }
        with self.index_path.open("wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def _build_index(self) -> None:
        raw_postings: dict[str, list[tuple[int, float]]] = defaultdict(list)
        doc_freq: Counter[str] = Counter()
        with self.catalog_path.open(encoding="utf-8") as handle:
            for doc_id, line in enumerate(handle):
                product = json.loads(line)
                self.parent_asins.append(str(product["parent_asin"]))
                self.product_texts.append(product_text(product))
                weighted_terms: Counter[str] = Counter()
                for field, weight in FIELD_WEIGHTS.items():
                    field_terms = terms(_text(product.get(field)))
                    weighted_terms.update({term: weight * count for term, count in Counter(field_terms).items()})
                doc_length = float(sum(weighted_terms.values())) or 1.0
                self.doc_lengths.append(doc_length)
                for term, weight in weighted_terms.items():
                    raw_postings[term].append((doc_id, float(weight)))
                    doc_freq[term] += 1
        doc_count = len(self.parent_asins) or 1
        self.avg_doc_length = sum(self.doc_lengths) / doc_count
        self.idf = {
            term: math.log(1.0 + (doc_count - freq + 0.5) / (freq + 0.5))
            for term, freq in doc_freq.items()
        }
        self.postings = dict(raw_postings)

    def search(self, query: str, top_k: int) -> list[dict]:
        query_terms = list(dict.fromkeys(terms(query)))[:40]
        if not query_terms:
            return []
        scores: dict[int, float] = defaultdict(float)
        k1 = 1.2
        b = 0.75
        for term in query_terms:
            idf = self.idf.get(term)
            if idf is None:
                continue
            for doc_id, term_weight in self.postings.get(term, []):
                length_norm = 1.0 - b + b * (self.doc_lengths[doc_id] / self.avg_doc_length)
                scores[doc_id] += idf * ((term_weight * (k1 + 1.0)) / (term_weight + k1 * length_norm))
        ranked = sorted(scores.items(), key=lambda item: (-item[1], self.parent_asins[item[0]]))[:max(top_k, RERANK_POOL)]
        reranked = sorted(
            ranked,
            key=lambda item: lexical_rerank_score(query, self.product_texts[item[0]], ranked.index(item)),
            reverse=True,
        )[:top_k]
        return [{"parent_asin": self.parent_asins[doc_id]} for doc_id, _score in reranked]

    def _load_product_texts(self) -> list[str]:
        texts: list[str] = []
        with self.catalog_path.open(encoding="utf-8") as handle:
            for line in handle:
                texts.append(product_text(json.loads(line)))
        return texts
