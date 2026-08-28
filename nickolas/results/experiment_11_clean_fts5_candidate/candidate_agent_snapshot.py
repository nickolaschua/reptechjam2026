"""Clean, deterministic FTS5 candidate derived from the Experiment 11 audit."""

from __future__ import annotations

import json
import re
import sqlite3
import time
from pathlib import Path
from typing import Any


TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
WS_RE = re.compile(r"\s+")
INITIAL_RE = re.compile(
    r"^\s*i(?:'|\u2018|\u2019)m\s+looking\s+for\s+(.+?)(?:,\s*but\s+i(?:'|\u2018|\u2019)m\s+still\s+exploring\.?|\.\s*(.*))$",
    re.IGNORECASE,
)
BUYING_RE = re.compile(r"^\s*a\s+key\s+requirement\s+is\s*:\s*(.+?)\.?\s*$", re.IGNORECASE)
DISCLOSURE_RE = re.compile(r"^\s*for\s+that,?\s+what\s+matters\s+is\s*:\s*(.+?)\.?\s*$", re.IGNORECASE)
OVERRIDE_RE = re.compile(
    r"^\s*actually,?\s+ignore\s+my\s+earlier\s+preference\.?\s+what\s+i\s+need\s+is\s*:\s*(.+?)\.?\s*$",
    re.IGNORECASE,
)
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "i", "in", "is", "it", "me", "my", "of", "on", "or", "please", "some",
    "that", "the", "this", "to", "want", "with", "would", "you", "looking",
    "color", "budget", "around",
}
QUESTION_SEQUENCE = ("feature", "material", "color", "size", "style", "use_case", "budget", "brand", "category")


def normalize(value: object) -> str:
    return WS_RE.sub(" ", str(value or "").lower()).strip().rstrip(".").strip()


def text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(f"{key} {item}" for key, item in value.items())
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return str(value)


def terms(value: str) -> list[str]:
    return [
        token.lower()
        for token in TOKEN_RE.findall(value)
        if len(token) > 1 and token.lower() not in STOPWORDS
    ]


class CleanFTSAgent:
    """Stateful field-weighted FTS5 agent with correct override invalidation.

    Only observable messages and catalog metadata enter retrieval. Pagination can
    be global, query-local, or disabled; all ranking ties end with parent ASIN.
    """

    def __init__(
        self,
        catalog_path: str | Path,
        *,
        question_policy: str = "specific",
        pagination_mode: str = "query",
        popularity_weight: float = 0.02,
    ) -> None:
        if question_policy not in {"specific", "other"}:
            raise ValueError(f"Unknown question policy: {question_policy}")
        if pagination_mode not in {"global", "query", "none"}:
            raise ValueError(f"Unknown pagination mode: {pagination_mode}")
        self.catalog_path = Path(catalog_path)
        self.question_policy = question_policy
        self.pagination_mode = pagination_mode
        self.popularity_weight = float(popularity_weight)
        self.sessions: dict[str, dict[str, Any]] = {}
        self.response_latencies_ms: list[float] = []
        started = time.perf_counter()
        self.connection = sqlite3.connect(":memory:")
        self.catalog_ids: list[str] = []
        self.metadata: dict[str, dict[str, Any]] = {}
        self._build_index()
        self.index_build_seconds = round(time.perf_counter() - started, 6)

    def configure(self, *, question_policy: str, pagination_mode: str) -> None:
        if question_policy not in {"specific", "other"} or pagination_mode not in {"global", "query", "none"}:
            raise ValueError("Invalid clean FTS configuration")
        self.question_policy = question_policy
        self.pagination_mode = pagination_mode
        self.sessions.clear()
        self.response_latencies_ms.clear()

    def _build_index(self) -> None:
        cursor = self.connection.cursor()
        cursor.execute(
            "CREATE VIRTUAL TABLE products USING fts5("
            "parent_asin UNINDEXED, title, categories, features, details, store, description, "
            "tokenize='unicode61 remove_diacritics 2')"
        )
        batch: list[tuple[str, str, str, str, str, str, str]] = []
        with self.catalog_path.open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                product = json.loads(line)
                pid = str(product["parent_asin"])
                title = text(product.get("title"))
                categories = text(product.get("categories"))
                features = product.get("features") or []
                searchable_bag = " ".join((title, categories, text(features[:3]))).lower()
                self.catalog_ids.append(pid)
                self.metadata[pid] = {
                    "title": title,
                    "brand": text(product.get("store")).strip().lower(),
                    "searchable_bag": searchable_bag,
                    "rating_number": float(product.get("rating_number") or 0.0),
                }
                batch.append((
                    pid,
                    title,
                    categories,
                    text(features),
                    text(product.get("details")),
                    text(product.get("store")),
                    text(product.get("description")),
                ))
                if len(batch) >= 1_000:
                    cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
                    batch.clear()
        if batch:
            cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
        self.connection.commit()

    def reset(self, session_id: str, user_profile: dict) -> None:
        self.sessions[session_id] = {
            "category": "",
            "constraints": [],
            "override_seed": None,
            "shown_global": set(),
            "shown_by_query": {},
            "history": [],
            "profile_present": bool(user_profile),
        }

    @staticmethod
    def _append_constraint(state: dict[str, Any], value: str) -> None:
        cleaned = WS_RE.sub(" ", str(value or "")).strip().rstrip(".").strip()
        if cleaned and all(normalize(existing) != normalize(cleaned) for existing in state["constraints"]):
            state["constraints"].append(cleaned)

    def _update_state(self, state: dict[str, Any], message: str, turn: int) -> None:
        if turn == 1:
            initial = INITIAL_RE.match(message)
            if initial:
                state["category"] = WS_RE.sub(" ", initial.group(1)).strip()
                remainder = (initial.group(2) or "").strip()
                buying = BUYING_RE.match(remainder)
                if buying:
                    self._append_constraint(state, buying.group(1))
                elif remainder:
                    seed = remainder.rstrip(".").strip()
                    self._append_constraint(state, seed)
                    state["override_seed"] = seed
                return

        disclosure = DISCLOSURE_RE.match(message)
        if disclosure:
            for value in re.split(r"\s*;\s*", disclosure.group(1)):
                self._append_constraint(state, value)
            return

        override = OVERRIDE_RE.match(message)
        if override:
            seed = normalize(state.get("override_seed"))
            if seed:
                state["constraints"] = [value for value in state["constraints"] if normalize(value) != seed]
            self._append_constraint(state, override.group(1))
            state["override_seed"] = None
            state["shown_global"].clear()
            state["shown_by_query"].clear()

    def _candidate_ids(self, query_terms: list[str]) -> list[str]:
        if not query_terms:
            return list(self.catalog_ids)
        expression_and = " AND ".join(f'"{term}"' for term in query_terms)
        rows = self.connection.execute(
            "SELECT parent_asin FROM products WHERE products MATCH ? ORDER BY rowid LIMIT 1000",
            (expression_and,),
        ).fetchall()
        candidate_ids = [str(row[0]) for row in rows]
        if len(candidate_ids) < 30:
            expression_or = " OR ".join(f'"{term}"' for term in query_terms)
            rows = self.connection.execute(
                "SELECT parent_asin, bm25(products, 0.0, 6.0, 4.0, 2.5, 2.5, 1.5, 1.0) AS score "
                "FROM products WHERE products MATCH ? ORDER BY score, parent_asin LIMIT 1000",
                (expression_or,),
            ).fetchall()
            present = set(candidate_ids)
            for row in rows:
                pid = str(row[0])
                if pid not in present:
                    candidate_ids.append(pid)
                    present.add(pid)
        return candidate_ids or list(self.catalog_ids)

    def _shown_set(self, state: dict[str, Any], query_key: str) -> set[str]:
        if self.pagination_mode == "global":
            return state["shown_global"]
        if self.pagination_mode == "query":
            return state["shown_by_query"].setdefault(query_key, set())
        return set()

    def _rank(self, state: dict[str, Any], top_k: int) -> list[str]:
        query = " ".join((state["category"] or "clothing item", *state["constraints"])).strip()
        query_terms = list(dict.fromkeys(terms(query)))[:45]
        query_key = normalize(query)
        shown = self._shown_set(state, query_key)
        scored: list[tuple[float, str]] = []
        for base_rank, pid in enumerate(self._candidate_ids(query_terms)):
            if pid in shown:
                continue
            metadata = self.metadata[pid]
            score = -0.001 * base_rank
            score += sum(0.3 for term in query_terms if term in metadata["searchable_bag"])
            score += self.popularity_weight * (metadata["rating_number"] ** 0.1)
            scored.append((score, pid))
        scored.sort(key=lambda item: (-item[0], item[1]))

        recommendations: list[str] = []
        selected_brands: dict[str, int] = {}
        selected_title_terms: list[set[str]] = []
        for _score, pid in scored:
            metadata = self.metadata[pid]
            brand = metadata["brand"]
            if brand and selected_brands.get(brand, 0) >= 2:
                continue
            title_terms = set(metadata["title"].lower().split())
            if any(
                len(title_terms & previous) / len(title_terms | previous) > 0.6
                for previous in selected_title_terms
                if title_terms and previous
            ):
                continue
            recommendations.append(pid)
            if brand:
                selected_brands[brand] = selected_brands.get(brand, 0) + 1
            selected_title_terms.append(title_terms)
            if len(recommendations) >= top_k:
                break
        if len(recommendations) < top_k:
            for _score, pid in scored:
                if pid not in recommendations:
                    recommendations.append(pid)
                    if len(recommendations) >= top_k:
                        break
        if self.pagination_mode != "none":
            shown.update(recommendations)
            state["shown_global"].update(recommendations)
        return recommendations

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        started = time.perf_counter()
        if session_id not in self.sessions:
            raise RuntimeError("reset must be called before respond")
        state = self.sessions[session_id]
        state["history"].append(user_message)
        self._update_state(state, user_message, turn)
        recommendations = self._rank(state, top_k)
        ask_attribute = "other" if self.question_policy == "other" else QUESTION_SEQUENCE[(turn - 1) % len(QUESTION_SEQUENCE)]
        self.response_latencies_ms.append((time.perf_counter() - started) * 1_000.0)
        return {
            "message": "Here are the strongest matches. What should I refine next?",
            "ask_attribute": ask_attribute,
            "recommendations": [{"parent_asin": pid} for pid in recommendations],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }
