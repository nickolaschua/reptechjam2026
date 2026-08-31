from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass
import os
import re
import json
import time
import requests
import numpy as np
from pathlib import Path
from typing import Any, Sequence
import builtins

try:
    from .embedding_backends import (
        CacheExpectation,
        CacheValidationError,
        CatalogCacheMissError,
        EmbeddingBackend,
        PRODUCT_TEXT_VERSION,
        cache_filename,
        embedding_backend_for_mode,
        fingerprint_file,
        fingerprint_texts,
        load_embedding_cache,
        save_embedding_cache,
    )
    from .config import CATALOG_PATH, EMBEDDING_CACHE_DIR, TEST_MODE
    from .catalogue import (
        BROWSING_FTS_OR_THRESHOLD, BROWSING_KEYWORD_ROUTE_THRESHOLD,
        BUYING_FTS_OR_THRESHOLD, BUYING_KEYWORD_ROUTE_THRESHOLD,
        Catalogue, GENERIC_NEGATIVE_TERMS, VECTOR_FALLBACK_LIMIT, contains_phrase,
    )
    from .clarification import (
        ATTRIBUTE_ORDER, BROWSING_ATTRIBUTE_ORDER, BUYING_ATTRIBUTE_ORDER,
        select_best_attributes,
    )
    from .memory_store import InMemoryUserMemoryStore, SNAPSHOT_VERSION
    from .vector_memory import (
        BuyerMode,
        DEFAULT_VECTOR_MEMORY_CONFIG,
        positive_slot_text,
        score_catalog,
    )
except ImportError:
    from embedding_backends import (
        CacheExpectation,
        CacheValidationError,
        CatalogCacheMissError,
        EmbeddingBackend,
        PRODUCT_TEXT_VERSION,
        cache_filename,
        embedding_backend_for_mode,
        fingerprint_file,
        fingerprint_texts,
        load_embedding_cache,
        save_embedding_cache,
    )
    from config import CATALOG_PATH, EMBEDDING_CACHE_DIR, TEST_MODE
    from catalogue import (
        BROWSING_FTS_OR_THRESHOLD, BROWSING_KEYWORD_ROUTE_THRESHOLD,
        BUYING_FTS_OR_THRESHOLD, BUYING_KEYWORD_ROUTE_THRESHOLD,
        Catalogue, GENERIC_NEGATIVE_TERMS, VECTOR_FALLBACK_LIMIT, contains_phrase,
    )
    from clarification import (
        ATTRIBUTE_ORDER, BROWSING_ATTRIBUTE_ORDER, BUYING_ATTRIBUTE_ORDER,
        select_best_attributes,
    )
    from memory_store import InMemoryUserMemoryStore, SNAPSHOT_VERSION
    from vector_memory import (
        BuyerMode,
        DEFAULT_VECTOR_MEMORY_CONFIG,
        positive_slot_text,
        score_catalog,
    )

try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    openai = None

try:
    import google.generativeai as genai
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False
    genai = None

def _normalize(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "").lower()).strip()

builtins._normalize = _normalize

SESSION_ONLY_HARD_ATTRIBUTES = {"budget", "gender", "rating", "reviews", "brand", "store"}


class DenseRetrievalError(ValueError):
    """Raised when a precomputed query cannot be scored in the M0 space."""


@dataclass(frozen=True)
class DenseRetrievalResult:
    """Aligned output from the canonical M0 catalogue dot-product scorer."""

    query_embedding: np.ndarray
    row_indices: np.ndarray
    product_ids: tuple[str, ...]
    scores: np.ndarray
    product_embeddings: np.ndarray

    def __post_init__(self) -> None:
        count = len(self.row_indices)
        if not (
            len(self.product_ids)
            == len(self.scores)
            == len(self.product_embeddings)
            == count
        ):
            raise DenseRetrievalError("Dense retrieval result fields are not aligned")


@dataclass(frozen=True)
class DenseQuerySnapshot:
    """One replayable current-query input; target ID is evaluation metadata only."""

    example_id: str
    raw_user_message: str
    effective_query_text: str
    query_embedding: np.ndarray
    target_product_id: str | None = None
    current_scope: str | None = None
    current_category: str | None = None
    user_id: str | None = None
    session_id: str | None = None
    embedding_space_id: str | None = None

    def __post_init__(self) -> None:
        for name in ("example_id", "raw_user_message", "effective_query_text"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise DenseRetrievalError(f"{name} must be a non-empty string")
        for name in (
            "target_product_id",
            "current_scope",
            "current_category",
            "user_id",
            "session_id",
            "embedding_space_id",
        ):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise DenseRetrievalError(f"{name} must be None or a non-empty string")

        query = np.array(self.query_embedding, dtype=np.float32, copy=True)
        if query.ndim != 1 or query.size == 0:
            raise DenseRetrievalError(
                "query_embedding must be a non-empty one-dimensional vector"
            )
        if not np.all(np.isfinite(query)):
            raise DenseRetrievalError("query_embedding must contain only finite values")
        norm = float(np.linalg.norm(query))
        if not np.isclose(norm, 1.0, rtol=1e-5, atol=1e-6):
            raise DenseRetrievalError(
                f"query_embedding must already be L2-normalized; got norm {norm:.8g}"
            )
        query.setflags(write=False)
        object.__setattr__(self, "query_embedding", query)

    @property
    def fixture_id(self) -> str:
        """Evaluator-facing name for the stable replay identifier."""

        return self.example_id

    @property
    def q_m0(self) -> np.ndarray:
        """The owned, read-only canonical float32 M0 query."""

        return self.query_embedding

    @property
    def query_scope(self) -> str | None:
        return self.current_scope


@dataclass(frozen=True)
class ForensicRankingSnapshot:
    """Evaluator-only, immutable evidence for paired M0/M3 reconstruction."""

    session_id: str
    turn: int
    canonical_state: dict[str, Any]
    v1: np.ndarray
    v2: np.ndarray | None
    s1: np.ndarray
    s2: np.ndarray | None
    s3: np.ndarray
    price_mask: np.ndarray
    negative_mask: np.ndarray
    eligibility_mask: np.ndarray
    m0_ranked_rows: np.ndarray
    m3_ranked_rows: np.ndarray
    gate_cosine: float | None
    gate_passed: bool
    current_weight: float
    memory_weight: float

    def __post_init__(self) -> None:
        for name in (
            "v1", "s1", "s3", "price_mask", "negative_mask",
            "eligibility_mask", "m0_ranked_rows", "m3_ranked_rows",
        ):
            value = np.array(getattr(self, name), copy=True)
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        for name in ("v2", "s2"):
            raw = getattr(self, name)
            if raw is not None:
                value = np.array(raw, copy=True)
                value.setflags(write=False)
                object.__setattr__(self, name, value)
        object.__setattr__(self, "canonical_state", deepcopy(self.canonical_state))



# Standard keyword lists for local parsing
COLORS = {"black", "white", "blue", "red", "pink", "green", "brown", "gray", "grey", "purple", "yellow", "orange"}
MATERIALS = {"cotton", "polyester", "nylon", "leather", "wool", "spandex", "silk", "rayon", "fabric"}
TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "i", "in", "is", "it", "me", "my", "of", "on", "or", "please", "some",
    "that", "the", "this", "to", "want", "with", "would", "you", "looking",
    "color", "budget", "around", "think", "mistake", "options", "items",
    "guess", "sure", "actually", "sorry", "thanks", "yes", "no", "look",
    "need", "prefer", "like", "find", "search", "show", "give", "bring",
    "tell", "ask", "reapply", "filters", "results", "suggestions", "pairs",
    "original", "um", "umm", "uh", "ignore", "earlier", "preference",
    "preferences", "character", "buy", "see", "seem", "seems", "get", "got",
    "make", "made", "done", "run", "doing"
}

def _text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(f"{key} {item}" for key, item in value.items())
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return str(value)

def _terms(text: str) -> list[str]:
    return [
        token.lower()
        for token in TOKEN_RE.findall(text)
        if len(token) > 1 and token.lower() not in STOPWORDS
    ]

def _state_to_retrieval_query(state: dict) -> str:
    """Build a deterministic query from only the currently active state."""
    fragments: list[str] = []

    def add(value: object) -> None:
        fragment = str(value).strip()
        if fragment and fragment not in fragments:
            fragments.append(fragment)

    add(state.get("category", ""))
    add(state.get("department", ""))
    positive_markers = {"true", "yes", "affirmative", "required", "included"}
    negative_markers = {"false", "no", "none", "n/a", "null", "other", ""}
    slots = state.get("disclosed_slots", {})
    for attr in sorted(slots, key=lambda value: str(value).casefold()):
        raw_values = slots[attr]
        values = raw_values if isinstance(raw_values, (set, list, tuple)) else [raw_values]
        for value in sorted(values, key=lambda item: str(item).casefold()):
            marker = str(value).strip().lower()
            if marker in positive_markers:
                add(attr)
            elif marker not in negative_markers:
                add(value)
    return " ".join(fragments)


def _json_safe_state(value: Any) -> Any:
    """Copy canonical parser state without exposing vector-bearing objects."""

    if isinstance(value, dict):
        return {str(key): _json_safe_state(item) for key, item in value.items()}
    if isinstance(value, (set, frozenset)):
        return sorted((_json_safe_state(item) for item in value), key=str)
    if isinstance(value, (list, tuple)):
        return [_json_safe_state(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)

class Agent:
    """Canonical short-term dialogue and gated vector-memory shopping agent."""
    def __init__(
        self,
        catalog_path: str | Path = None,
        embedding_backend: EmbeddingBackend | None = None,
        *,
        test_mode: bool = TEST_MODE,
        allow_catalog_embedding: bool = False,
        embedding_cache_dir: str | Path | None = None,
        memory_store: InMemoryUserMemoryStore | None = None,
    ) -> None:
        initialization_started = time.perf_counter()
        self.instrumentation = {
            "initialization": {
                "catalog_loading_seconds": 0.0,
                "product_text_build_seconds": 0.0,
                "embedding_cache_load_seconds": 0.0,
                "catalog_embedding_generation_seconds": 0.0,
                "total_seconds": 0.0,
                "cache_status": "not-attempted",
            },
            "semantic_queries": [],
            "turns": [],
            "agent_errors": [],
            "baseline_fallback_count": 0,
        }
        if catalog_path is None:
            catalog_path = CATALOG_PATH
        self.catalog_path = Path(catalog_path)

        self.test_mode = bool(test_mode)
        self.embedding_backend = embedding_backend or embedding_backend_for_mode(
            self.test_mode
        )
        self.embedding_backend_id = self.embedding_backend.backend_id
        self.embedding_model_id = self.embedding_backend.model_id
        self.embedding_space_id = self.embedding_backend.embedding_space_id
        self.model_path = self.embedding_model_id
        self.model = getattr(self.embedding_backend, "_model", None)
        self.allow_catalog_embedding = bool(allow_catalog_embedding)
        self.embedding_cache_dir = Path(embedding_cache_dir or EMBEDDING_CACHE_DIR)
        self.memory_store = memory_store or InMemoryUserMemoryStore()
        self.vector_memory_config = DEFAULT_VECTOR_MEMORY_CONFIG
        self._active_lifecycle: dict[str, dict[str, Any]] = {}
        self._ended_lifecycle: dict[str, dict[str, Any]] = {}
        self._forensic_capture_sessions: set[str] = set()
        self._forensic_ranking_snapshots: dict[str, list[ForensicRankingSnapshot]] = {}
        self._forensic_update_sessions: set[str] = set()
        self._forensic_update_vectors: dict[str, np.ndarray] = {}

        self._sessions = {}
        self._closed = False

        # Build only metadata and the cached vector index used by the demo path.
        catalog_started = time.perf_counter()
        self._build_category_index()
        self.instrumentation["initialization"]["catalog_loading_seconds"] = (
            time.perf_counter() - catalog_started
        )

        print(
            f"[Hybrid Agent] Embedding backend: {self.embedding_backend_id} "
            f"({self.embedding_model_id})"
        )
        self._build_vector_index()

        self.instrumentation["initialization"]["total_seconds"] = (
            time.perf_counter() - initialization_started
        )


    def _build_category_index(self) -> None:
        print("[System Agent] Loading catalogue metadata and FTS5 index...")
        self.catalogue = Catalogue(self.catalog_path)
        self.catalog_ids = self.catalogue.ids
        self.catalog_ids_arr = self.catalogue.ids_array
        self.catalog_row_by_asin = self.catalogue.row_by_asin
        self.catalog_products = self.catalogue.products
        self.catalog_prices = self.catalogue.prices
        self.catalog_departments = self.catalogue.departments
        self.catalog_categories_set = self.catalogue.categories
        self.catalog_avg_ratings = self.catalogue.avg_ratings
        self.catalog_rating_numbers = self.catalogue.rating_numbers
        self.catalog_brands = self.catalogue.brands
        self.catalog_popularity = self.catalog_rating_numbers.astype(float)
        self.catalog_metadata = self.catalogue.metadata
        self.connection = self.catalogue.connection
        print(f"[System Agent] Loaded {len(self.catalog_ids):,} catalogue rows.")

    def _build_vector_index(self) -> None:
        product_text_started = time.perf_counter()
        self.catalog_texts = []
        for p in self.catalog_products:
            title = p.get("title") or ""
            cats = ", ".join(p.get("categories") or [])
            feats = "; ".join((p.get("features") or [])[:3])
            text = f"Product: {title}. Categories: {cats}. Features: {feats}.".strip()
            self.catalog_texts.append(text)
        self.instrumentation["initialization"]["product_text_build_seconds"] = (
            time.perf_counter() - product_text_started
        )

        self.catalog_fingerprint = fingerprint_file(self.catalog_path)
        self.product_text_fingerprint = fingerprint_texts(self.catalog_texts)
        expectation = CacheExpectation(
            backend_id=self.embedding_backend_id,
            model_id=self.embedding_model_id,
            embedding_space_id=self.embedding_space_id,
            catalog_ids=self.catalog_ids,
            product_text_version=PRODUCT_TEXT_VERSION,
            product_text_fingerprint=self.product_text_fingerprint,
            catalog_fingerprint=self.catalog_fingerprint,
            vector_dimension=getattr(self.embedding_backend, "vector_dimension", None),
            normalized=True,
        )
        cache_path = self.embedding_cache_dir / cache_filename(self.embedding_backend_id)
        self.embedding_cache_path = cache_path

        if cache_path.exists():
            print(f"[Hybrid Agent] Loading pre-computed embeddings: {cache_path.name}")
            cache_started = time.perf_counter()
            try:
                self.catalog_embeddings = load_embedding_cache(cache_path, expectation)
                self.instrumentation["initialization"]["cache_status"] = "hit"
                self.instrumentation["initialization"]["embedding_cache_load_seconds"] = (
                    time.perf_counter() - cache_started
                )
                return
            except CacheValidationError as exc:
                self.instrumentation["initialization"]["cache_status"] = "rejected"
                self.instrumentation["initialization"]["embedding_cache_load_seconds"] = (
                    time.perf_counter() - cache_started
                )
                print(f"[Hybrid Agent] Rejecting incompatible embedding cache: {exc}")

        if not self.allow_catalog_embedding:
            raise CatalogCacheMissError(
                f"No valid {self.embedding_backend_id} catalog cache at {cache_path}. "
                "Catalog generation is disabled; run the explicit benchmark cache-build command."
            )

        count = len(self.catalog_texts)
        batch_size = int(getattr(self.embedding_backend, "batch_size", count or 1))
        expected_batches = (count + batch_size - 1) // batch_size
        print(
            f"[Hybrid Agent] Encoding {count} products with {self.embedding_backend_id} "
            f"in approximately {expected_batches} batch(es); cache={cache_path}"
        )
        generation_started = time.perf_counter()
        self.catalog_embeddings = self.embedding_backend.embed_catalog(self.catalog_texts)
        self.instrumentation["initialization"]["catalog_embedding_generation_seconds"] = (
            time.perf_counter() - generation_started
        )
        self.instrumentation["initialization"]["cache_status"] = "built"
        save_embedding_cache(cache_path, self.catalog_embeddings, expectation)
        print("[Hybrid Agent] Vector index built and cached.")

    def close(self) -> None:
        """Release owned catalogue resources; repeated calls are harmless."""

        if not getattr(self, "_closed", False):
            for session in list(getattr(self, "_active_lifecycle", {})):
                self.discard_session(session)
            catalogue = getattr(self, "catalogue", None)
            if catalogue is not None:
                catalogue.close()
            self._closed = True

    def _call_llm(self, prompt: str, system_prompt: str = "", session_id: str = None, response_json: bool = False) -> str:
        import urllib.request

        if os.environ.get("FAST_EVAL") == "1":
            return "Here are the top matches based on your preferences."

        openai_key = os.environ.get("OPENAI_API_KEY")
        deepseek_key = os.environ.get("DEEPSEEK_API_KEY")
        gemini_key = os.environ.get("GEMINI_API_KEY")

        # Cleanup placeholders
        if openai_key and (openai_key.startswith("your_") or "placeholder" in openai_key.lower()):
            openai_key = None
        if deepseek_key and (deepseek_key.startswith("your_") or "placeholder" in deepseek_key.lower()):
            deepseek_key = None
        if gemini_key and (gemini_key.startswith("your_") or "placeholder" in gemini_key.lower()):
            gemini_key = None

        model_used = "Static Fallback"
        res_text = ""

        # ==========================================
        # 1. Attempt DeepSeek if key is present
        # ==========================================
        if deepseek_key:
            # Method A: OpenAI SDK pointing to DeepSeek
            if HAS_OPENAI:
                try:
                    client = openai.OpenAI(api_key=deepseek_key, base_url="https://api.deepseek.com/v1")
                    res = client.chat.completions.create(
                        model="deepseek-chat",
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.4,
                        max_tokens=150,
                        response_format={"type": "json_object"} if response_json else None
                    )
                    res_text = res.choices[0].message.content.strip()
                    model_used = "DeepSeek-Chat (DeepSeek SDK)"
                except Exception as sdk_err:
                    print(f"[Hybrid Agent] DeepSeek SDK call failed: {sdk_err}. Trying urllib...")

            # Method B: urllib.request (System SSL native)
            if not res_text:
                url = "https://api.deepseek.com/v1/chat/completions"
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {deepseek_key}",
                    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"
                }
                payload = {
                    "model": "deepseek-chat",
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.4,
                    "max_tokens": 150
                }
                if response_json:
                    payload["response_format"] = {"type": "json_object"}
                try:
                    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
                    with urllib.request.urlopen(req, timeout=3) as response:
                        data = json.loads(response.read().decode("utf-8"))
                        res_text = data["choices"][0]["message"]["content"].strip()
                        model_used = "DeepSeek-Chat (DeepSeek urllib)"
                except Exception as urllib_err:
                    print(f"[Hybrid Agent] DeepSeek urllib call failed: {urllib_err}. Trying requests...")

            # Method C: requests
            if not res_text:
                url = "https://api.deepseek.com/v1/chat/completions"
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {deepseek_key}"
                }
                payload = {
                    "model": "deepseek-chat",
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.4,
                    "max_tokens": 150
                }
                if response_json:
                    payload["response_format"] = {"type": "json_object"}
                try:
                    res = requests.post(url, json=payload, timeout=3)
                    res.raise_for_status()
                    data = res.json()
                    res_text = data["choices"][0]["message"]["content"].strip()
                    model_used = "DeepSeek-Chat (DeepSeek requests)"
                except Exception as e:
                    print(f"[Hybrid Agent] DeepSeek requests call failed: {e}")

        # ==========================================
        # 2. Attempt OpenAI if key is present
        # ==========================================
        if not res_text and openai_key:
            # Method A: OpenAI SDK
            if HAS_OPENAI:
                try:
                    client = openai.OpenAI(api_key=openai_key)
                    res = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.4,
                        max_tokens=150,
                        response_format={"type": "json_object"} if response_json else None
                    )
                    res_text = res.choices[0].message.content.strip()
                    model_used = "GPT-4o-Mini (OpenAI SDK)"
                except Exception as sdk_err:
                    print(f"[Hybrid Agent] OpenAI SDK call failed: {sdk_err}. Trying urllib...")

            # Method B: urllib.request (System SSL native)
            if not res_text:
                url = "https://api.openai.com/v1/chat/completions"
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {openai_key}",
                    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"
                }
                payload = {
                    "model": "gpt-4o-mini",
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.4,
                    "max_tokens": 150
                }
                if response_json:
                    payload["response_format"] = {"type": "json_object"}
                try:
                    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
                    with urllib.request.urlopen(req, timeout=3) as response:
                        data = json.loads(response.read().decode("utf-8"))
                        res_text = data["choices"][0]["message"]["content"].strip()
                        model_used = "GPT-4o-Mini (OpenAI urllib)"
                except Exception as urllib_err:
                    print(f"[Hybrid Agent] OpenAI urllib call failed: {urllib_err}. Trying requests...")

            # Method C: requests
            if not res_text:
                url = "https://api.openai.com/v1/chat/completions"
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {openai_key}"
                }
                payload = {
                    "model": "gpt-4o-mini",
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.4,
                    "max_tokens": 150
                }
                if response_json:
                    payload["response_format"] = {"type": "json_object"}
                try:
                    res = requests.post(url, json=payload, timeout=3)
                    res.raise_for_status()
                    data = res.json()
                    res_text = data["choices"][0]["message"]["content"].strip()
                    model_used = "GPT-4o-Mini (OpenAI requests)"
                except Exception as e:
                    print(f"[Hybrid Agent] OpenAI requests call failed: {e}")

        # ==========================================
        # 3. Attempt Gemini if key is present
        # ==========================================
        if not res_text and gemini_key:
            # Method A: Google GenerativeAI SDK
            if HAS_GEMINI:
                try:
                    genai.configure(api_key=gemini_key)
                    model = genai.GenerativeModel("gemini-1.5-flash")
                    # Format system instructions if supported, or prepend
                    res = model.generate_content(
                        f"{system_prompt}\n\nUser request: {prompt}",
                        generation_config={
                            "temperature": 0.4,
                            "max_output_tokens": 150,
                            "response_mime_type": "application/json" if response_json else "text/plain"
                        }
                    )
                    res_text = res.text.strip()
                    model_used = "Gemini-1.5-Flash (Gemini SDK)"
                except Exception as sdk_err:
                    print(f"[Hybrid Agent] Gemini SDK call failed: {sdk_err}. Trying urllib...")

            # Method B: urllib.request (System SSL native)
            if not res_text:
                url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={gemini_key}"
                headers = {
                    "Content-Type": "application/json",
                    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"
                }
                payload = {
                    "contents": [{
                        "role": "user",
                        "parts": [{"text": f"{system_prompt}\n\nUser request: {prompt}"}]
                    }],
                    "generationConfig": {
                        "temperature": 0.4,
                        "maxOutputTokens": 150
                    }
                }
                if response_json:
                    payload["generationConfig"]["responseMimeType"] = "application/json"
                try:
                    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
                    with urllib.request.urlopen(req, timeout=3) as response:
                        data = json.loads(response.read().decode("utf-8"))
                        res_text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
                        model_used = "Gemini-1.5-Flash (Gemini urllib)"
                except Exception as urllib_err:
                    print(f"[Hybrid Agent] Gemini urllib call failed: {urllib_err}. Trying requests...")

            # Method C: requests
            if not res_text:
                url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={gemini_key}"
                headers = {"Content-Type": "application/json"}
                payload = {
                    "contents": [{
                        "role": "user",
                        "parts": [{"text": f"{system_prompt}\n\nUser request: {prompt}"}]
                    }],
                    "generationConfig": {
                        "temperature": 0.4,
                        "maxOutputTokens": 150
                    }
                }
                if response_json:
                    payload["generationConfig"]["responseMimeType"] = "application/json"
                try:
                    res = requests.post(url, json=payload, timeout=3)
                    res.raise_for_status()
                    data = res.json()
                    res_text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
                    model_used = "Gemini-1.5-Flash (Gemini requests)"
                except Exception as e:
                    print(f"[Hybrid Agent] Gemini requests call failed: {e}")

        # ==========================================
        # 4. Attempt local Ollama as final API option
        # ==========================================
        if not res_text:
            url = "http://localhost:11434/api/chat"
            payload = {
                "model": "llama3.1",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                "stream": False,
                "options": {
                    "temperature": 0.4
                }
            }
            if response_json:
                payload["format"] = "json"
            try:
                res = requests.post(url, json=payload, timeout=3)
                res.raise_for_status()
                data = res.json()
                res_text = data["message"]["content"].strip()
                model_used = "Llama 3.1 (Ollama)"
            except Exception as e:
                print(f"[Hybrid Agent] Ollama fallback failed: {e}")
                res_text = "Here are the top matches based on your preferences."
                model_used = "Static Fallback"

        # Save to debug session info if session_id is available
        if session_id and session_id in self._sessions:
            debug_info = self._sessions[session_id].setdefault("debug_info", {})
            debug_info["model"] = model_used
            debug_info["system_prompt"] = system_prompt
            debug_info["user_prompt"] = prompt

        return res_text

    def _classify_constraint_locally(self, val: str) -> str:
        """Categorize a value into one of the evaluator's allowed attributes."""
        val_lower = val.lower()
        if re.search(r"\b(?:ratings|reviews|feedback)\b", val_lower) and re.search(r"\d", val_lower):
            return "reviews"
        if re.search(r"\bstars?\b", val_lower) and re.search(r"\d", val_lower):
            return "rating"
        if re.search(r"\b(women|men|boys|girls|kids|toddler)\b", val_lower):
            return "gender"
        if "brand" in val_lower or "store" in val_lower:
            return "brand"
        if "budget" in val_lower or re.search(r"(?:\$|<=|under)\s*\d", val_lower):
            return "budget"
        if any(m in val_lower for m in MATERIALS):
            return "material"
        if any(c in val_lower for c in COLORS):
            return "color"
        if any(word in val_lower for word in ("size", "sizing", "width", "wide", "narrow")):
            return "size"
        if any(word in val_lower for word in ("sole", "heel", "wedge", "cushion", "rubber", "flat")):
            return "sole"
        if any(word in val_lower for word in ("department", "style", "fit", "sleeve", "neck", "combat", "fashion", "riding", "chelsea", "casual", "dressy", "western", "cowboy", "rain", "snow", "bootie")):
            return "style"
        return "feature"

    @staticmethod
    def _slot_values(value: object) -> set[str]:
        if value is None:
            return set()
        values = value if isinstance(value, (set, list, tuple)) else [value]
        return {str(item).strip() for item in values if str(item).strip()}

    def _record_constraint(
        self, state: dict, attr: str, value: str, turn: int, source_type: str,
        *, promote_existing: bool = False,
    ) -> None:
        normalized = _normalize(value)
        for record in state["constraint_provenance"]:
            if record["attribute"] == attr and _normalize(record["value"]) == normalized and record["status"] == "active":
                if source_type == "explicit_override" or promote_existing:
                    record["source_turn"] = turn
                    record["source_type"] = source_type
                return
        state["constraint_provenance"].append({
            "attribute": attr, "value": value, "source_turn": int(turn),
            "source_type": source_type, "status": "active",
        })

    def _revoke_constraint_record(self, state: dict, record: dict) -> None:
        if record["status"] != "active":
            return
        record["status"] = "revoked"
        for term in _terms(record["value"]):
            if term not in state["stashed_terms"]:
                state["stashed_terms"].append(term)
        attr = record["attribute"]
        active = self._slot_values(state["disclosed_slots"].get(attr))
        remaining = {value for value in active if _normalize(value) != _normalize(record["value"])}
        if remaining:
            state["disclosed_slots"][attr] = remaining
        else:
            state["disclosed_slots"].pop(attr, None)
            if attr == "gender": state["target_department"] = ""
            elif attr == "rating": state["min_avg_rating"] = 0.0
            elif attr == "reviews": state["min_rating_number"] = 0
            elif attr == "budget": state["price_max"] = 9999.0
            elif attr == "brand": state["store"] = ""

    def _set_constraint(
        self, state: dict, attr: str, values: object, turn: int, source_type: str,
        source_message: str = "",
    ) -> None:
        new_values = self._slot_values(values)
        normalized = {_normalize(value) for value in new_values}
        for record in state["constraint_provenance"]:
            if record["attribute"] == attr and record["status"] == "active" and _normalize(record["value"]) not in normalized:
                self._revoke_constraint_record(state, record)
        if new_values:
            state["disclosed_slots"][attr] = new_values
        else:
            state["disclosed_slots"].pop(attr, None)
        for value in new_values:
            self._record_constraint(
                state, attr, value, turn, source_type,
                promote_existing=(source_type == "clarification" and _normalize(value) in _normalize(source_message)),
            )

    def _rebuild_active_terms(self, state: dict) -> None:
        terms: list[str] = []
        for term in _terms(state.get("category", "")):
            if term not in terms:
                terms.append(term)
        for attr, values in state["disclosed_slots"].items():
            for value in self._slot_values(values):
                lowered = value.lower()
                if lowered in {"false", "no", "none", "n/a", "null", "other"}:
                    continue
                source = attr if lowered in {"true", "yes", "affirmative", "required", "included"} else value
                for term in _terms(source):
                    if term not in terms and term not in state.get("negated_terms", set()):
                        terms.append(term)
        state["accumulated_terms"] = terms

    @staticmethod
    def _advance_search_epoch(state: dict) -> None:
        state["search_epoch"] += 1
        current_seen: set[str] = set()
        state["seen_asins_by_epoch"][state["search_epoch"]] = current_seen
        state["seen_asins"] = current_seen

    @staticmethod
    def _extract_override_value(message: str) -> str | None:
        match = re.search(r"what i need is:\s*(.+?)\.?\s*$", message, re.IGNORECASE | re.DOTALL)
        return None if not match else match.group(1).strip().rstrip(".").strip() or None

    def _apply_explicit_override(self, state: dict, new_value: str, turn: int) -> None:
        new_attr = self._classify_constraint_locally(new_value)
        for record in list(state["constraint_provenance"]):
            revoke_initial = record["source_type"] == "initial_preference"
            revoke_conflict = record["attribute"] == new_attr and _normalize(record["value"]) != _normalize(new_value)
            if record["status"] == "active" and (revoke_initial or revoke_conflict):
                self._revoke_constraint_record(state, record)
        values = set() if new_attr in {"brand", "budget", "color", "material", "size", "sole", "style"} else self._slot_values(state["disclosed_slots"].get(new_attr))
        values.add(new_value)
        state["disclosed_slots"][new_attr] = values
        if new_attr == "budget":
            match = re.search(r"\d+(?:\.\d+)?", new_value)
            if match: state["price_max"] = float(match.group())
        elif new_attr == "gender":
            match = re.search(r"\b(women|men|boys|girls|kids|toddler)\b", new_value.lower())
            if match: state["target_department"] = match.group(1)
        elif new_attr == "rating":
            match = re.search(r"\d+(?:\.\d+)?", new_value)
            if match: state["min_avg_rating"] = float(match.group())
        elif new_attr == "reviews":
            match = re.search(r"\d+", new_value)
            if match: state["min_rating_number"] = int(match.group())
        elif new_attr == "brand":
            state["store"] = re.sub(r"\b(?:brand|store|only|like)\b", " ", new_value.lower()).strip()
        if any(word in new_value.lower() for word in [
            "shoe", "boot", "sandal", "slide", "sneaker", "clog", "shirt", "pant",
            "hoodie", "jacket", "dress", "ring", "necklace", "earring", "bracelet",
        ]):
            state["category"] = " ".join(new_value.lower().split()[:4])
        self._record_constraint(state, new_attr, new_value, turn, "explicit_override")
        self._advance_search_epoch(state)
        self._rebuild_active_terms(state)

    def _erase_attribute_memory(self, state: dict, attr: str) -> None:
        """Purge the attribute from active slot memory, stash its keywords, and update terms."""
        direct_values = self._slot_values(state["disclosed_slots"].get(attr))
        for value in direct_values:
            for term in _terms(value):
                if term not in state["stashed_terms"]:
                    state["stashed_terms"].append(term)
        for record in state["constraint_provenance"]:
            if record["attribute"] == attr and record["status"] == "active":
                self._revoke_constraint_record(state, record)
        state["disclosed_slots"].pop(attr, None)
        if attr == "gender": state["target_department"] = ""
        elif attr == "rating": state["min_avg_rating"] = 0.0
        elif attr == "reviews": state["min_rating_number"] = 0
        elif attr == "budget": state["price_max"] = 9999.0
        elif attr == "brand": state["store"] = ""
        self._rebuild_active_terms(state)

    @staticmethod
    def _new_session_state() -> dict:
        initial_seen: set[str] = set()
        return {
            "disclosed_slots": {},
            "constraint_provenance": [],
            "accumulated_terms": [],
            "stashed_terms": [],
            "search_epoch": 0,
            "seen_asins": initial_seen,
            "seen_asins_by_epoch": {0: initial_seen},
            "history": [],
            "negated_terms": set(),
            "asked_attributes": set(),
            "intent_mode": BuyerMode.BROWSING.value,
            "intent_source": "session_default",
            # Slot attributes
            "category": "clothing",
            "department": "",
            "price_max": 9999.0,
            "target_department": "",
            "min_avg_rating": 0.0,
            "min_rating_number": 0,
            "store": "",
            "debug_info": {},
        }

    def reset(
        self,
        session_id: str,
        user_profile: dict,
        *,
        user_id: str | None = None,
        sequence_index: int | None = None,
    ) -> None:
        """Start a canonical Fast Memory session, optionally in shadow mode.

        ``user_profile`` remains accepted for evaluator compatibility but is
        never used to infer identity or longitudinal preference facts.
        """

        del user_profile
        session = str(session_id)
        if not session.strip():
            raise ValueError("session_id must be non-empty")

        resolved_user: str | None = None
        resolved_sequence: int | None = None
        visible_state = None
        if user_id is None:
            if sequence_index is not None:
                raise ValueError("sequence_index requires an explicit user_id")
        else:
            resolved_user = str(user_id)
            if not resolved_user.strip():
                raise ValueError("user_id must be non-empty when provided")
            if (
                isinstance(sequence_index, bool)
                or not isinstance(sequence_index, int)
                or sequence_index < 0
            ):
                raise ValueError(
                    "sequence_index must be a non-negative integer in longitudinal mode"
                )
            resolved_sequence = sequence_index
            if any(
                metadata["user_id"] == resolved_user
                for active_id, metadata in self._active_lifecycle.items()
                if active_id != session and metadata["user_id"] is not None
            ):
                raise ValueError(
                    f"user {resolved_user!r} already has an active session"
                )
            visible_state = self.memory_store.get_state(
                resolved_user, before_sequence_index=resolved_sequence
            )
            if visible_state is not None and visible_state.embedding_space_id != self.embedding_space_id:
                raise ValueError("stored memory belongs to a different embedding space")
            self.memory_store.begin_session(resolved_user, session, resolved_sequence)

        # This remains the exact frozen canonical M0 state construction.
        self._sessions[session] = self._new_session_state()
        self._active_lifecycle[session] = {
            "session_id": session,
            "user_id": resolved_user,
            "sequence_index": resolved_sequence,
            "visible_state": visible_state,
        }
        self._ended_lifecycle.pop(session, None)

    def end_session(
        self,
        session_id: str,
        outcome: Any = None,
        purchased_product: Any = None,
        evidence: Any = None,
    ) -> Any:
        """Finalize and atomically commit positive disclosed-slot evidence.

        Lifecycle outcome values are deliberately ignored by extraction. They
        cannot become preference text, embedding input, or Fast Memory state.
        """

        del outcome, purchased_product, evidence
        session = str(session_id)
        if session not in self._sessions or session not in self._active_lifecycle:
            raise RuntimeError("reset must be called before end_session")
        metadata = self._active_lifecycle[session]
        final_fast_memory = deepcopy(self._sessions[session])
        self._reconcile_negated_state(final_fast_memory)
        preference_text = ""
        commit = None

        if metadata["user_id"] is not None:
            soft_slots = {
                attr: values
                for attr, values in final_fast_memory.get("disclosed_slots", {}).items()
                if str(attr).strip().lower() not in SESSION_ONLY_HARD_ATTRIBUTES
            }
            preference_text = positive_slot_text(soft_slots)
            new_preferences = None
            if preference_text:
                # Embed before mutating the store. Provider failures leave the
                # active session intact so end_session can be retried.
                new_preferences = self.embed_dense_query(preference_text)
                if session in getattr(self, "_forensic_update_sessions", set()):
                    update = np.array(new_preferences, copy=True)
                    update.setflags(write=False)
                    self._forensic_update_vectors[session] = update
            commit = self.memory_store.commit(
                user_id=metadata["user_id"],
                session_id=session,
                sequence_index=metadata["sequence_index"],
                embedding_space_id=self.embedding_space_id,
                new_preferences=new_preferences,
                alpha=self.vector_memory_config.ewma_alpha,
            )

        self._ended_lifecycle[session] = {
            "session_id": session,
            "user_id": metadata["user_id"],
            "sequence_index": metadata["sequence_index"],
            "visible_state": metadata["visible_state"],
            "final_fast_memory": final_fast_memory,
            "preference_text": preference_text,
            "commit": commit,
        }
        while len(self._ended_lifecycle) > 32:
            oldest = next(iter(self._ended_lifecycle))
            self._ended_lifecycle.pop(oldest, None)
        getattr(self, "_forensic_capture_sessions", set()).discard(session)
        getattr(self, "_forensic_update_sessions", set()).discard(session)
        del self._active_lifecycle[session]
        del self._sessions[session]
        return commit

    def discard_session(self, session_id: str) -> None:
        """Abandon active Fast Memory without committing longitudinal state."""

        session = str(session_id)
        metadata = self._active_lifecycle.pop(session, None)
        self._sessions.pop(session, None)
        self._forensic_capture_sessions.discard(session)
        self._forensic_update_sessions.discard(session)
        self._forensic_ranking_snapshots.pop(session, None)
        self._forensic_update_vectors.pop(session, None)
        if metadata is not None and metadata.get("user_id") is not None:
            self.memory_store.cancel_session(session)

    def get_visible_memories(self, session_id: str) -> tuple[Any, ...]:
        """Return the reset-time shadow snapshot without applying it."""

        session = str(session_id)
        metadata = self._active_lifecycle.get(session) or self._ended_lifecycle.get(session)
        if metadata is None:
            raise RuntimeError(f"unknown session {session!r}")
        state = metadata["visible_state"]
        return () if state is None else (state,)

    def get_memory_debug(self, session_id: str, *, consume: bool = False) -> dict[str, Any]:
        """Return vector-free Phase-5 lifecycle and memory observability."""

        session = str(session_id)
        metadata = self._active_lifecycle.get(session)
        ended = False
        if metadata is None:
            metadata = self._ended_lifecycle.get(session)
            ended = metadata is not None
        if metadata is None:
            raise RuntimeError(f"unknown session {session!r}")

        visible = metadata["visible_state"]
        commit = metadata.get("commit")

        final_state = (
            metadata["final_fast_memory"]
            if ended
            else deepcopy(self._sessions[session])
        )
        result = {
            "session_id": session,
            "user_id": metadata["user_id"],
            "sequence_index": metadata["sequence_index"],
            "ended": ended,
            "final_fast_memory": deepcopy(final_state),
            "preference_text": metadata.get("preference_text", ""),
            "vector_changed": None if commit is None else commit.vector_changed,
            "ltm_updated_after_turn": bool(commit is not None and commit.vector_changed),
            "ltm_updated_after_session": bool(commit is not None and commit.vector_changed),
            "memory_update_text": metadata.get("preference_text", ""),
            "memory_version": SNAPSHOT_VERSION,
            "post_update_memory": None if commit is None or commit.state is None else {
                "user_id": commit.state.user_id,
                "last_committed_sequence": commit.state.last_committed_sequence,
                "update_count": commit.state.update_count,
            },
            "visible_prior_memory_count": int(visible is not None),
            "visible_prior_memories": ([] if visible is None else [{
                "user_id": visible.user_id,
                "last_committed_sequence": visible.last_committed_sequence,
                "update_count": visible.update_count,
            }]),
            "embedding_space_id": self.embedding_space_id,
            "historical_memory_applied": bool(final_state.get("debug_info", {}).get("memory_trace", {}).get("gate_passed", False)),
        }
        if consume and ended:
            self._ended_lifecycle.pop(session, None)
        return result

    def enable_forensic_ranking(self, session_id: str) -> None:
        """Opt a live session into vector-bearing evaluator evidence."""

        session = str(session_id)
        if session not in self._active_lifecycle:
            raise RuntimeError("reset must be called before enabling forensic ranking")
        self._forensic_capture_sessions.add(session)

    def enable_forensic_memory_update(self, session_id: str) -> None:
        """Capture the one update embedding used by ``end_session``."""

        session = str(session_id)
        if session not in self._active_lifecycle:
            raise RuntimeError("reset must be called before enabling forensic update")
        self._forensic_update_sessions.add(session)

    def get_forensic_memory_update(self, session_id: str) -> np.ndarray | None:
        return self._forensic_update_vectors.get(str(session_id))

    def get_forensic_ranking_snapshots(
        self, session_id: str
    ) -> tuple[ForensicRankingSnapshot, ...]:
        """Return evaluator-only snapshots; normal response/debug stays vector-free."""

        return tuple(self._forensic_ranking_snapshots.get(str(session_id), ()))

    @staticmethod
    def _explicit_no_preference_attributes(message: str) -> set[str]:
        matches = re.finditer(
            r"\bi (?:do not|don't) have (?:an? |any )?(?:additional )?preference for\s+([a-z_]+)",
            str(message).lower(),
        )
        return {match.group(1).strip() for match in matches}

    @staticmethod
    def _extract_negated_terms(message: str) -> set[str]:
        """Extract bounded negative values without swallowing the product noun."""

        text = str(message).lower()
        known_values = sorted(
            set(MATERIALS) | set(COLORS) | {
                "polyester", "nylon", "silk", "pvc", "resin", "denim", "canvas",
                "suede", "rubber", "flat", "heel", "wedge", "cushion", "combat",
                "fashion", "riding", "chelsea", "casual", "dressy", "western",
                "cowboy", "rain", "snow", "bootie",
            },
            key=len,
            reverse=True,
        )
        result: set[str] = set()
        for match in re.finditer(
            r"\b(?:not|no|except|without|instead of)\b\s*([^,;.!?]+)", text
        ):
            clause = re.split(
                r"\b(?:but|although|however|under|below|max(?:imum)?|budget|with|that|which)\b",
                match.group(1), maxsplit=1,
            )[0]
            recognized = [value for value in known_values if contains_phrase(clause, value)]
            if recognized:
                result.update(recognized)
                continue
            words = [word for word in re.findall(r"[a-z0-9]+", clause) if word not in STOPWORDS]
            if words:
                result.add(words[0])
        return result

    def _reconcile_negated_state(self, state: dict, turn: int = 0) -> None:
        negatives = {_normalize(term) for term in state.get("negated_terms", set()) if _normalize(term)}
        state["negated_terms"] = negatives
        for attr, values in list(state.get("disclosed_slots", {}).items()):
            retained = {
                value for value in self._slot_values(values)
                if not any(contains_phrase(value, term) or contains_phrase(term, value) for term in negatives)
            }
            if retained == self._slot_values(values):
                continue
            for record in state.get("constraint_provenance", []):
                if record.get("attribute") == attr and record.get("status") == "active":
                    if any(contains_phrase(record.get("value", ""), term) or contains_phrase(term, record.get("value", "")) for term in negatives):
                        self._revoke_constraint_record(state, record)
            if retained:
                state["disclosed_slots"][attr] = retained
            else:
                state["disclosed_slots"].pop(attr, None)
        self._rebuild_active_terms(state)


    def _parse_message_locally(self, session_id: str, message: str, turn: int = 0) -> None:
        state = self._sessions[session_id]
        msg_lower = message.lower()

        # 1. Check for Intent Override
        override_value = self._extract_override_value(message)
        if override_value:
            self._apply_explicit_override(state, override_value, turn)
            return

        # 2. Check for Boundary Case
        no_preference_attrs = self._explicit_no_preference_attributes(message)
        if no_preference_attrs:
            for attr in no_preference_attrs:
                if "asked_attributes" not in state:
                    state["asked_attributes"] = set()
                state["asked_attributes"].add(attr)
                self._erase_attribute_memory(state, attr)
            return

        # 3. Extract Negated Terms
        negated_terms = self._extract_negated_terms(message)

        if "negated_terms" not in state:
            state["negated_terms"] = set()
        state["negated_terms"].update(negated_terms)

        # 4. Extract Category, Department, Budget slots
        cat_match = re.search(
            r"i'm looking for\s+(.+?)(?=\s+(?:under|below|max(?:imum)?|with|without|but|that|which|budget|from\s+brands?)\b|[,.;]|$)",
            msg_lower,
        )
        if cat_match:
            cand_cat = cat_match.group(1).strip()
            for negated in sorted(negated_terms, key=len, reverse=True):
                cand_cat = re.sub(rf"\b(?:no|not|without)?\s*{re.escape(negated)}\b", " ", cand_cat)
            cand_cat = re.sub(r"\s+", " ", cand_cat).strip()
            if len(cand_cat.split()) <= 4:
                state["category"] = cand_cat
            else:
                state["category"] = " ".join(cand_cat.split()[:4])

        if any(w in msg_lower for w in ["shoe", "boot", "sandal", "slide", "sneaker", "clog"]):
            state["department"] = "shoes"
        elif any(w in msg_lower for w in ["ring", "necklace", "earring", "bracelet", "jewelry"]):
            state["department"] = "jewelry"
        elif any(w in msg_lower for w in ["shirt", "pant", "hoodie", "jacket", "underwear", "socks", "onesie", "dress", "leotard"]):
            state["department"] = "clothing"

        budget_match = re.search(r"(?:under|below|max|budget of)\s*\$?(\d+(?:\.\d+)?)", msg_lower)
        if budget_match:
            state["price_max"] = float(budget_match.group(1))
            self._set_constraint(state, "budget", f"under {budget_match.group(1)}", turn, "initial_preference" if turn <= 1 else "clarification", message)

        for demographic in ["women", "men", "boys", "girls", "kids", "toddler"]:
            if re.search(rf"\b{demographic}\b", msg_lower):
                state["target_department"] = demographic
                self._set_constraint(state, "gender", demographic, turn, "initial_preference" if turn <= 1 else "clarification", message)
                break
        rating_match = re.search(r"(\d+(?:\.\d+)?)\s*stars?(?:\s+and\s+above|\s+or\s+more)?", msg_lower)
        if rating_match:
            state["min_avg_rating"] = float(rating_match.group(1))
            self._set_constraint(state, "rating", rating_match.group(0), turn, "initial_preference" if turn <= 1 else "clarification", message)
        reviews_match = re.search(r"(?:at\s+least|minimum|more\s+than)\s*(\d+)\s*(?:ratings|reviews|feedback)", msg_lower)
        if reviews_match:
            state["min_rating_number"] = int(reviews_match.group(1))
            self._set_constraint(state, "reviews", reviews_match.group(0), turn, "initial_preference" if turn <= 1 else "clarification", message)

        # 5. Extract standard attributes (matters is, key requirement is)
        if "a key requirement is:" in msg_lower:
            idx = msg_lower.find("a key requirement is:")
            val_part = message[idx + len("a key requirement is:"):].strip()
            if val_part.endswith("."):
                val_part = val_part[:-1]
            attr = self._classify_constraint_locally(val_part)
            self._set_constraint(state, attr, val_part, turn, "initial_preference" if turn <= 1 else "clarification", message)

        if "what matters is:" in msg_lower:
            idx = msg_lower.find("what matters is:")
            val_part = message[idx + len("what matters is:"):].strip()
            if val_part.endswith("."):
                val_part = val_part[:-1]
            values = [v.strip() for v in val_part.split(";")]
            for val in values:
                attr = self._classify_constraint_locally(val)
                current = self._slot_values(state["disclosed_slots"].get(attr))
                current = {val} if attr in {"brand", "budget", "color", "material", "size", "sole", "style"} else current | {val}
                self._set_constraint(state, attr, current, turn, "clarification", message)

        # Extract brand
        brand_match = re.search(
            r"brand(?:s)? like\s+(.+?)(?=\s+(?:under|below|max(?:imum)?|budget|with|without|but|not|no)\b|[,.;]|$)",
            msg_lower,
        )
        if brand_match:
            b_val = brand_match.group(1).strip().lower()
            state["store"] = b_val
            self._set_constraint(state, "brand", b_val, turn, "initial_preference" if turn <= 1 else "clarification", message)

        # Extract material
        materials_found = [m for m in ["leather", "wool", "cotton", "polyester", "nylon", "silk", "pvc", "resin", "denim", "canvas"] if contains_phrase(msg_lower, m) and m not in negated_terms]
        if materials_found:
            self._set_constraint(state, "material", materials_found[0], turn, "initial_preference" if turn <= 1 else "clarification", message)

        # Extract sole
        sole_found = [s for s in ["rubber", "flat", "heel", "wedge", "cushion"] if contains_phrase(msg_lower, s) and s not in negated_terms]
        if sole_found:
            self._set_constraint(state, "sole", sole_found[0], turn, "initial_preference" if turn <= 1 else "clarification", message)

        # Extract style
        style_found = [st for st in ["combat", "fashion", "riding", "chelsea", "casual", "dressy", "western", "cowboy", "rain", "snow", "bootie"] if contains_phrase(msg_lower, st) and st not in negated_terms]
        if style_found:
            self._set_constraint(state, "style", style_found[0], turn, "initial_preference" if turn <= 1 else "clarification", message)

        # Extract color
        colors_found = [c for c in COLORS if contains_phrase(msg_lower, c) and c not in negated_terms]
        if colors_found:
            self._set_constraint(state, "color", colors_found[0], turn, "initial_preference" if turn <= 1 else "clarification", message)

        # Extract color
        color_found = [c for c in ["black", "white", "blue", "red", "pink", "green", "brown", "gray", "grey", "purple", "yellow", "orange", "gold", "silver"] if contains_phrase(msg_lower, c) and c not in negated_terms]
        if color_found:
            self._set_constraint(state, "color", color_found[0], turn, "initial_preference" if turn <= 1 else "clarification", message)

        self._reconcile_negated_state(state, turn)

    @staticmethod
    def _can_use_fast_path(user_message: str, turn: int) -> bool:
        msg_lower = user_message.lower().strip()
        if turn == 1:
            return msg_lower.startswith("i'm looking for ")
        return (
            msg_lower.startswith("for that, what matters is:") or
            msg_lower.startswith("actually, ignore my earlier preference. what i need is:") or
            msg_lower.startswith("i don't have a preference for") or
            msg_lower.startswith("i don't have an additional preference for") or
            "those options are not quite right yet" in msg_lower
        )

    @staticmethod
    def _detect_intent_locally(state: dict[str, Any], user_message: str) -> BuyerMode | None:
        """Deterministic fallback for Yangxu's live Buying/Browsing transition rules."""

        message = " ".join(str(user_message).lower().split())
        if not message:
            return None
        true_resets = ("start over", "actually, show me other styles")
        if any(signal in message for signal in true_resets):
            return BuyerMode.BROWSING

        previous = str(state.get("intent_mode", BuyerMode.BROWSING.value)).lower()
        buying_signals = (
            "i need", "it must", "must have", "i want specifically", "what i need is",
            "a key requirement is", "for that, what matters is", "my budget", "under $",
        )
        concrete_slots = {"budget", "size", "brand", "material", "color"}
        disclosed = set(state.get("disclosed_slots", {}))
        has_hard_condition = (
            float(state.get("price_max", 9999.0)) < 9999.0
            or bool(state.get("target_department"))
            or float(state.get("min_avg_rating", 0.0)) > 0.0
            or int(state.get("min_rating_number", 0)) > 0
            or bool(state.get("store"))
        )
        message_has_concrete = (
            any(signal in message for signal in buying_signals)
            or bool(re.search(r"(?:\$\s*\d|\b(?:under|below|max(?:imum)?|budget)\s+\$?\d)", message))
            or bool(re.search(r"\b(?:specific|waterproof|nike|size\s+\w+)\b", message))
            or "not just looking" in message
        )
        if message_has_concrete or has_hard_condition or disclosed & concrete_slots:
            return BuyerMode.BUYING
        if "just looking" in message:
            return BuyerMode.BROWSING
        if previous == BuyerMode.BUYING.value:
            return BuyerMode.BUYING
        return BuyerMode.BROWSING

    def _resolve_live_intent(
        self,
        state: dict[str, Any],
        user_message: str,
        caller_fallback: BuyerMode | None,
    ) -> BuyerMode:
        detected = self._detect_intent_locally(state, user_message)
        if state.pop("_intent_detection_succeeded", False):
            llm_mode = BuyerMode(str(state.get("intent_mode", BuyerMode.BROWSING.value)))
            normalized = " ".join(str(user_message).lower().split())
            if detected is BuyerMode.BUYING or any(
                reset in normalized for reset in ("start over", "actually, show me other styles")
            ):
                state["intent_mode"] = detected.value
                state["intent_source"] = "deterministic_precedence"
                return detected
            return llm_mode
        if detected is not None:
            state["intent_mode"] = detected.value
            state["intent_source"] = "deterministic"
            return detected
        if caller_fallback is not None:
            state["intent_mode"] = caller_fallback.value
            state["intent_source"] = "caller_fallback"
            return caller_fallback
        state["intent_source"] = "session_default"
        return BuyerMode(str(state.get("intent_mode", BuyerMode.BROWSING.value)))

    def respond(
        self,
        session_id: str,
        user_message: str,
        turn: int,
        top_k: int,
        *,
        buyer_mode: str | BuyerMode | None = None,
        debug: bool = False,
    ) -> dict:
        respond_started = time.perf_counter()
        route_used = "vector-memory"
        dense_before = len(self.instrumentation["semantic_queries"])
        failed = False
        state_snapshot = None
        forensic_count = 0
        try:
            if session_id not in self._active_lifecycle:
                raise RuntimeError("reset must be called before respond")
            state_snapshot = deepcopy(self._sessions[session_id])
            forensic_count = len(self._forensic_ranking_snapshots.get(session_id, ()))
            visible = self._active_lifecycle[session_id]["visible_state"]
            mode = None
            if buyer_mode is not None:
                try:
                    mode = buyer_mode if isinstance(buyer_mode, BuyerMode) else BuyerMode(buyer_mode)
                except (TypeError, ValueError) as exc:
                    raise ValueError("buyer_mode must be exactly 'buying' or 'browsing'") from exc
            result = self._respond_custom(
                session_id, user_message, turn, top_k, mode, emit_debug=False
            )
            if not debug:
                result.pop("debug", None)
            return result
        except Exception as exc:
            failed = True
            if state_snapshot is not None:
                self._sessions[session_id] = state_snapshot
                snapshots = self._forensic_ranking_snapshots.get(session_id)
                if snapshots is not None:
                    del snapshots[forensic_count:]
            self.instrumentation["agent_errors"].append(
                {
                    "session_id": session_id,
                    "turn": int(turn),
                    "type": type(exc).__name__,
                    "message": str(exc),
                }
            )
            raise
        finally:
            self.instrumentation["turns"].append(
                {
                    "session_id": session_id,
                    "turn": int(turn),
                    "route": route_used,
                    "dense_invoked": len(self.instrumentation["semantic_queries"]) > dense_before,
                    "respond_seconds": time.perf_counter() - respond_started,
                    "failed": failed,
                }
            )

    def _validate_dense_query_embedding(
        self,
        query_embedding: np.ndarray,
    ) -> np.ndarray:
        """Return the float32 M0-boundary vector after contract validation.

        Inputs must already be L2-normalized. Normalizing here would conceal an
        upstream geometry error and could make a future q_star differ from the
        vector the evaluator intended to score.
        """

        catalog_embeddings = np.asarray(self.catalog_embeddings)
        if catalog_embeddings.ndim != 2 or catalog_embeddings.shape[1] == 0:
            raise DenseRetrievalError(
                "catalog_embeddings must be a non-empty two-dimensional matrix"
            )

        raw_query = np.asarray(query_embedding)
        if raw_query.ndim != 1:
            raise DenseRetrievalError(
                f"query_embedding must be one-dimensional, got shape {raw_query.shape}"
            )
        expected_dimension = int(catalog_embeddings.shape[1])
        if raw_query.shape[0] != expected_dimension:
            raise DenseRetrievalError(
                "query_embedding dimension does not match the catalogue: "
                f"expected {expected_dimension}, got {raw_query.shape[0]}"
            )
        try:
            query = np.asarray(raw_query, dtype=catalog_embeddings.dtype)
        except (TypeError, ValueError) as exc:
            raise DenseRetrievalError("query_embedding must be numeric") from exc
        if not np.all(np.isfinite(query)):
            raise DenseRetrievalError("query_embedding must contain only finite values")

        norm = float(np.linalg.norm(query))
        if not np.isclose(norm, 1.0, rtol=1e-5, atol=1e-6):
            raise DenseRetrievalError(
                f"query_embedding must already be L2-normalized; got norm {norm:.8g}"
            )
        return query

    def embed_dense_query(self, query_text: str) -> np.ndarray:
        """Embed current M0 query text once and expose the exact normalized q."""

        query = self.embedding_backend.embed_query(str(query_text))
        return self._validate_dense_query_embedding(query)

    def dense_retrieve_vector(
        self,
        query_embedding: np.ndarray,
        top_n: int = 150,
    ) -> DenseRetrievalResult:
        """Score a precomputed normalized q against the existing M0 matrix.

        This method does not embed text, call OpenAI, apply filters, or consult
        longitudinal memory. It intentionally retains M0's existing full-matrix
        dot product and reversed NumPy argsort tie behaviour.
        """

        if isinstance(top_n, bool) or not isinstance(top_n, (int, np.integer)):
            raise DenseRetrievalError("top_n must be a non-negative integer")
        if int(top_n) < 0:
            raise DenseRetrievalError("top_n must be a non-negative integer")

        query = self._validate_dense_query_embedding(query_embedding)
        catalog_embeddings = np.asarray(self.catalog_embeddings)

        # Canonical M0 dense scoring implementation. Keep this expression and
        # ordering equivalent to the pre-refactor text-driven implementation.
        all_scores = np.dot(catalog_embeddings, query)
        row_indices = np.argsort(all_scores)[::-1][: int(top_n)]
        product_ids = tuple(self.catalog_ids[int(row)] for row in row_indices)

        return DenseRetrievalResult(
            query_embedding=query,
            row_indices=row_indices,
            product_ids=product_ids,
            scores=all_scores[row_indices],
            product_embeddings=catalog_embeddings[row_indices],
        )

    def dense_retrieve_text(
        self,
        query_text: str,
        top_n: int = 150,
    ) -> DenseRetrievalResult:
        """Embed text once, then delegate to the canonical vector scorer."""

        total_started = time.perf_counter()
        embed_started = time.perf_counter()
        query_embedding = self.embed_dense_query(query_text)
        query_embedding_seconds = time.perf_counter() - embed_started

        search_started = time.perf_counter()
        result = self.dense_retrieve_vector(query_embedding, top_n=top_n)
        dense_search_seconds = time.perf_counter() - search_started
        self.instrumentation["semantic_queries"].append(
            {
                "query_text": query_text,
                "query_embedding_seconds": query_embedding_seconds,
                "dense_search_seconds": dense_search_seconds,
                "total_dense_retrieval_seconds": time.perf_counter() - total_started,
                "top_n": int(top_n),
                "embedding_space_id": self.embedding_space_id,
            }
        )
        return result

    def _dense_retrieve(self, query_text: str, top_n: int = 150) -> np.ndarray:
        """Compatibility wrapper returning row indices for the current M0 route."""

        return self.dense_retrieve_text(query_text, top_n=top_n).row_indices

    def freeze_dense_query(
        self,
        *,
        example_id: str,
        raw_user_message: str,
        effective_query_text: str,
        target_product_id: str | None = None,
        current_scope: str | None = None,
        current_category: str | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
    ) -> DenseQuerySnapshot:
        """Create a replayable query snapshot with one embedding operation.

        ``target_product_id`` is copied into evaluation metadata only. It never
        contributes to effective query text or q.
        """

        query_embedding = self.embed_dense_query(effective_query_text)
        return DenseQuerySnapshot(
            example_id=str(example_id),
            raw_user_message=str(raw_user_message),
            effective_query_text=str(effective_query_text),
            query_embedding=query_embedding,
            target_product_id=(
                None if target_product_id is None else str(target_product_id)
            ),
            current_scope=None if current_scope is None else str(current_scope),
            current_category=(
                None if current_category is None else str(current_category)
            ),
            user_id=None if user_id is None else str(user_id),
            session_id=None if session_id is None else str(session_id),
            embedding_space_id=self.embedding_space_id,
        )

    def get_instrumentation(self) -> dict:
        snapshot = deepcopy(self.instrumentation)
        snapshot["backend_id"] = self.embedding_backend_id
        snapshot["model_id"] = self.embedding_model_id
        snapshot["embedding_space_id"] = self.embedding_space_id
        snapshot["embedding_api"] = self.embedding_backend.usage_snapshot()
        return snapshot

    def _update_state_via_llm(self, session_id: str, user_message: str, turn: int | None = None) -> None:
        import json
        import urllib.request
        state = self._sessions[session_id]
        turn = int(state.get("_current_turn", 0) if turn is None else turn)

        # Prepare past state representation for the LLM input
        past_state_data = {
            "intent_mode": state.get("intent_mode", BuyerMode.BROWSING.value),
            "category": state.get("category", "clothing"),
            "department": state.get("department", ""),
            "price_max": state.get("price_max", 9999.0),
            "target_department": state.get("target_department", ""),
            "min_avg_rating": state.get("min_avg_rating", 0.0),
            "min_rating_number": state.get("min_rating_number", 0),
            "store": state.get("store", ""),
            "disclosed_slots": {k: list(v) if isinstance(v, set) else v for k, v in state["disclosed_slots"].items()},
            "negated_terms": list(state.get("negated_terms", set())),
            "asked_attributes": list(state.get("asked_attributes", set()))
        }

        sys_prompt = (
            "You are a precise dialogue state tracking assistant for an e-commerce fashion shopping copilot.\n"
            "Your task is to read the customer's message and update the JSON state representing their active shopping filters and constraints.\n\n"
            "Rules for intent_mode:\n"
            "- buying: the user has a specific item in mind, definite requirements, or hard constraints. Signals include 'I need', 'it must be', 'I want specifically', or an explicit budget, size, brand, or material.\n"
            "- browsing: the user is exploring, vague, uncertain, or open to suggestions. Signals include 'just looking', 'show me options', 'still exploring', 'anything', or a vague category-only query.\n"
            "Re-evaluate intent every turn. Upgrade browsing to buying as soon as requirements become concrete. Revert buying to browsing only after an explicit reset such as 'actually, show me other styles'.\n\n"
            "Guideline attributes you can extract for constraints:\n"
            "- color, material, size, brand, use_case, style, budget.\n"
            "Note: You are NOT confined to this list. If the user specifies requirements for other attributes (e.g. \"zipper closure\" -> closure, \"slim fit\" -> fit, \"striped\" -> pattern), extract them as custom keys inside \"disclosed_slots\".\n"
            "Note: The \"department\" field must ONLY be empty or one of \"clothing\", \"shoes\", \"jewelry\", \"watches\". Put every target gender or age demographic (men, women, boys, girls, kids, toddler) exclusively in \"target_department\"; never put demographics in \"department\", \"use_case\", or any \"disclosed_slots\" key.\n\n"
            "Rules:\n"
            "1. Extract any new constraints specified by the user and add/update them in \"disclosed_slots\". Values should be short strings or lists of strings. \"disclosed_slots\" must be the complete current active mapping; omit revoked or deleted keys.\n"
            "2. If the user overrides a constraint (e.g. \"Actually, I need polyester, not cotton\" or \"I changed my mind, make it red instead of black\"), erase the old preference and update it with the new one.\n"
            "3. If the user overrides the product type (e.g., \"ignore slippers, I want sneakers\"), update the \"category\" field and clear all other attributes in \"disclosed_slots\" since they belonged to the old item type.\n"
            "4. Extract negative preferences (e.g. \"no leather\", \"except dresses\") and add them to \"negated_terms\".\n"
            "5. If the user explicitly states they don't have a preference for an attribute and tells the assistant to use its judgment (e.g., \"I don't have a preference for size\"), add that attribute to \"asked_attributes\" (to prevent asking again) and remove it from \"disclosed_slots\" if present.\n"
            "6. Extract hard conditions when explicitly stated: demographic into target_department; minimum stars into min_avg_rating; minimum reviews into min_rating_number; requested brand/store into store. Preserve prior values when not changed.\n"
            "7. Clean up: ensure \"category\" and \"department\" are updated if mentioned.\n"
            "8. Return ONLY a valid JSON object matching this schema:\n"
            "{\n"
            "  \"intent_mode\": \"buying|browsing\",\n"
            "  \"category\": \"string\",\n"
            "  \"department\": \"string\",\n"
            "  \"price_max\": float,\n"
            "  \"target_department\": \"women|men|boys|girls|kids|toddler|empty\",\n"
            "  \"min_avg_rating\": float,\n"
            "  \"min_rating_number\": integer,\n"
            "  \"store\": \"string\",\n"
            "  \"disclosed_slots\": {\n"
            "    \"attribute_name\": [\"value1\", \"value2\"]\n"
            "  },\n"
            "  \"negated_terms\": [\"term1\", \"term2\"],\n"
            "  \"asked_attributes\": [\"attr1\", \"attr2\"]\n"
            "}\n"
            "Do not include any conversational text or markdown codeblock wrappers except raw valid JSON."
        )

        prompt = (
            f"Past State:\n{json.dumps(past_state_data, indent=2)}\n\n"
            f"Customer Message:\n\"{user_message}\"\n\n"
            "Output the updated JSON state strictly matching the schema:"
        )

        res_text = self._call_llm(prompt, sys_prompt, session_id=session_id, response_json=True)

        # Clean up any potential markdown wrappers
        if "```json" in res_text:
            res_text = res_text.split("```json")[1].split("```")[0]
        elif "```" in res_text:
            res_text = res_text.split("```")[1].split("```")[0]
        res_text = res_text.strip()

        state_before_update = deepcopy(state)
        try:
            new_state = json.loads(res_text)

            # Sync back to state dict
            if "intent_mode" in new_state:
                detected = str(new_state["intent_mode"]).strip().lower()
                if detected in {BuyerMode.BUYING.value, BuyerMode.BROWSING.value}:
                    state["intent_mode"] = detected
                    state["intent_source"] = "llm"
                    state["_intent_detection_succeeded"] = True

            if "category" in new_state:
                new_cat = str(new_state["category"]).strip()
                if new_cat != state.get("category"):
                    for record in state["constraint_provenance"]:
                        if record["status"] == "active":
                            self._revoke_constraint_record(state, record)
                    state["category"] = new_cat
                    state["seen_asins"].clear() # Clear recommendations if product category changed
                    state["disclosed_slots"].clear() # Rule 3: Clear old constraints if category changed
                    state["asked_attributes"].clear()

            demographic_values = {"men", "women", "boys", "girls", "kids", "toddler"}
            canonical_departments = {"clothing", "shoes", "jewelry", "watches", ""}
            dept_val = str(new_state.get("department") or "").strip().lower()
            target_value = str(new_state.get("target_department") or "").strip().lower()
            legacy_demographic = dept_val if dept_val in demographic_values else ""
            resolved_demographic = (
                target_value if target_value in demographic_values else legacy_demographic
            )

            if "department" in new_state:
                if legacy_demographic:
                    cat_val = state.get("category", "").lower()
                    if any(w in cat_val for w in ["shoe", "boot", "sandal", "slide", "sneaker", "clog", "cleat"]):
                        state["department"] = "shoes"
                    elif any(w in cat_val for w in ["ring", "necklace", "earring", "bracelet", "jewelry"]):
                        state["department"] = "jewelry"
                    else:
                        state["department"] = "clothing"
                elif dept_val in canonical_departments:
                    state["department"] = dept_val

            if "price_max" in new_state:
                try:
                    state["price_max"] = float(new_state["price_max"])
                    if state["price_max"] < 9999.0:
                        self._set_constraint(state, "budget", f"under {state['price_max']:g}", turn, "initial_preference" if turn <= 1 else "clarification", user_message)
                    elif "budget" in state["disclosed_slots"]:
                        self._erase_attribute_memory(state, "budget")
                except Exception:
                    state["price_max"] = 9999.0

            if "target_department" in new_state or legacy_demographic:
                if resolved_demographic:
                    state["target_department"] = resolved_demographic
                    self._set_constraint(state, "gender", resolved_demographic, turn, "initial_preference" if turn <= 1 else "clarification", user_message)
                elif target_value == "":
                    state["target_department"] = ""
                    if "gender" in state["disclosed_slots"]:
                        self._erase_attribute_memory(state, "gender")
            if "min_avg_rating" in new_state:
                try:
                    state["min_avg_rating"] = max(0.0, float(new_state["min_avg_rating"] or 0.0))
                    if state["min_avg_rating"] > 0: self._set_constraint(state, "rating", f"{state['min_avg_rating']:g} stars", turn, "initial_preference" if turn <= 1 else "clarification", user_message)
                    elif "rating" in state["disclosed_slots"]: self._erase_attribute_memory(state, "rating")
                except (TypeError, ValueError): pass
            if "min_rating_number" in new_state:
                try:
                    state["min_rating_number"] = max(0, int(new_state["min_rating_number"] or 0))
                    if state["min_rating_number"] > 0: self._set_constraint(state, "reviews", f"{state['min_rating_number']} reviews", turn, "initial_preference" if turn <= 1 else "clarification", user_message)
                    elif "reviews" in state["disclosed_slots"]: self._erase_attribute_memory(state, "reviews")
                except (TypeError, ValueError): pass
            if "store" in new_state:
                state["store"] = str(new_state["store"] or "").strip().lower()
                if state["store"]: self._set_constraint(state, "brand", state["store"], turn, "initial_preference" if turn <= 1 else "clarification", user_message)
                elif "brand" in state["disclosed_slots"]: self._erase_attribute_memory(state, "brand")

            incoming_negatives = set(state.get("negated_terms", set()))
            incoming_negatives.update(self._extract_negated_terms(user_message))
            if "negated_terms" in new_state and isinstance(new_state["negated_terms"], list):
                incoming_negatives.update(
                    str(term).strip() for term in new_state["negated_terms"] if str(term).strip()
                )
            state["negated_terms"] = incoming_negatives

            if "disclosed_slots" in new_state and isinstance(new_state["disclosed_slots"], dict):
                normalized_slots = {}
                for k, v in new_state["disclosed_slots"].items():
                    if isinstance(v, list):
                        normalized_slots[str(k).strip()] = set(str(item).strip() for item in v if str(item).strip())
                    else:
                        normalized_slots[str(k).strip()] = {str(v).strip()} if str(v).strip() else set()
                override_language = bool(re.search(
                    r"\b(?:actually|instead of|changed my mind|make it|what i need is|rather than)\b",
                    user_message.lower(),
                ))
                for attr, values in normalized_slots.items():
                    if not values:
                        continue
                    merged = values if override_language else self._slot_values(state["disclosed_slots"].get(attr)) | values
                    self._set_constraint(
                        state, str(attr), merged, turn,
                        "explicit_override" if override_language else ("initial_preference" if turn <= 1 else "clarification"),
                        user_message,
                    )

            if "asked_attributes" in new_state and isinstance(new_state["asked_attributes"], list):
                valid_asked = {
                    str(attr).strip().lower() for attr in new_state["asked_attributes"]
                    if str(attr).strip().lower() in set(ATTRIBUTE_ORDER)
                }
                state["asked_attributes"].update(valid_asked)

            for attr in self._explicit_no_preference_attributes(user_message):
                if attr in set(ATTRIBUTE_ORDER):
                    state["asked_attributes"].add(attr)
                self._erase_attribute_memory(state, attr)
            self._reconcile_negated_state(state, turn)

        except Exception as parse_err:
            print(f"[Hybrid Agent] Failed to parse updated state JSON: {parse_err}. Content: {res_text}")
            # Fallback to local regex-based parsing if LLM Call 1 fails
            self._sessions[session_id] = state_before_update
            state = self._sessions[session_id]
            self._parse_message_locally(session_id, user_message, turn)

        self._reconcile_negated_state(state, turn)

    def _parse_generator_json(self, res_text: str, all_attrs: set[str]) -> tuple[str, str]:
        import re
        import json

        try:
            res_json = json.loads(res_text)
            msg = str(res_json.get("response_message", "")).strip()
            attr = str(res_json.get("asked_attribute", "")).strip().lower()
            if msg:
                return msg, attr
        except Exception:
            pass

        # Fallback regex extraction
        msg = ""
        attr = "other"

        # 1. Extract message
        # Check for response_message or message or _message followed by : or = and quotes
        msg_match = re.search(r'(?:response_message|message|_message)\s*[:=]\s*["\'](.*?)["\']', res_text, re.DOTALL | re.IGNORECASE)
        if msg_match:
            msg = msg_match.group(1).strip()
        else:
            # If no key found, extract any non-json lines
            lines = []
            for line in res_text.splitlines():
                l = line.strip()
                if not l or l in ["{", "}"]:
                    continue
                if "asked_attribute" in l.lower() or "attribute" in l.lower():
                    continue
                # strip potential keys
                cleaned = re.sub(r'^(?:response_message|message|_message)\s*[:=]\s*', '', l, flags=re.IGNORECASE)
                cleaned = re.sub(r'^["\']|["\']$', '', cleaned.strip())
                if cleaned:
                    lines.append(cleaned)
            if lines:
                msg = " ".join(lines)

        # 2. Extract asked_attribute
        attr_match = re.search(r'(?:asked_attribute|attribute)\s*[:=]\s*["\'](.*?)["\']', res_text, re.IGNORECASE)
        if attr_match:
            attr = attr_match.group(1).strip().lower()
        else:
            # Check if any keyword in all_attrs is in the text
            for a in all_attrs:
                if a in res_text.lower():
                    attr = a
                    break

        if not msg:
            msg = res_text

        return msg, attr

    def _extract_asked_attributes(self, agent_message: str, state: dict) -> set[str]:
        all_attrs = set(ATTRIBUTE_ORDER)
        disclosed_keys = set(state["disclosed_slots"].keys())
        question_sentences = re.findall(r"[^.!?]*\?", agent_message.lower())
        detected: set[str] = set()
        for attr in all_attrs:
            if attr in disclosed_keys:
                continue
            for s in question_sentences:
                if contains_phrase(s, attr.replace("_", " ")):
                    detected.add(attr)
                elif attr == "use_case" and any(contains_phrase(s, word) for word in ["occasion", "activity", "sport", "hiking", "running", "work"]):
                    detected.add("use_case")
                elif attr == "budget" and any(token in s for token in ["price", "spend", "cost"]):
                    detected.add("budget")
                elif attr == "style" and any(contains_phrase(s, word) for word in ["look", "design", "aesthetic"]):
                    detected.add("style")
        return detected

    def _extract_asked_attribute(self, agent_message: str, state: dict) -> str:
        detected = self._extract_asked_attributes(agent_message, state)
        return next((attr for attr in ATTRIBUTE_ORDER if attr in detected), "other")

    def _respond_custom(
        self, session_id: str, user_message: str, turn: int, top_k: int,
        buyer_mode: BuyerMode | None = None,
        *,
        emit_debug: bool = True,
    ) -> dict:
        state = self._sessions[session_id]
        state["history"].append({"role": "user", "content": user_message})
        state["debug_info"] = {"vector_fallback": False, "fts5_count": 0}
        state["_intent_detection_succeeded"] = False

        # 1. Preserve the established fast/local versus full parsing boundary.
        if self._can_use_fast_path(user_message, turn):
            self._parse_message_locally(session_id, user_message, turn)
        else:
            state["_current_turn"] = int(turn)
            try:
                self._update_state_via_llm(session_id, user_message)
            finally:
                state.pop("_current_turn", None)

        live_mode = self._resolve_live_intent(state, user_message, buyer_mode)
        if live_mode is BuyerMode.BUYING:
            fts_or_threshold = BUYING_FTS_OR_THRESHOLD
            keyword_route_threshold = BUYING_KEYWORD_ROUTE_THRESHOLD
        else:
            fts_or_threshold = BROWSING_FTS_OR_THRESHOLD
            keyword_route_threshold = BROWSING_KEYWORD_ROUTE_THRESHOLD

        # 2. Always embed the canonical active state and score all catalogue rows.
        query_text = _state_to_retrieval_query(state)
        v1 = self.embed_dense_query(query_text)
        lifecycle = self._active_lifecycle[session_id]
        visible = lifecycle["visible_state"]
        v2 = None if visible is None else visible.vector
        s1, s2, s3, gate_cosine, gate_passed, a, b = score_catalog(
            self.catalog_embeddings, v1, v2, live_mode, self.vector_memory_config
        )

        # 3. Route lexically, then apply the session-only categorical mask.
        if hasattr(self, "catalogue"):
            eligibility = self.catalogue.eligibility(state)
            fts = self.catalogue.fts_route(
                state.get("accumulated_terms", []), or_threshold=fts_or_threshold
            )
        else:
            # Lightweight compatibility for isolated scorer contract tests.
            hard_mask = np.ones(len(self.catalog_ids), dtype=bool)
            if state["price_max"] < 9999.0:
                hard_mask &= self.catalog_prices <= state["price_max"]
            negative_mask = np.asarray([
                not any(
                    contains_phrase(self.catalog_metadata[pid]["searchable_bag"], normalized)
                    for term in state.get("negated_terms", set())
                    if (normalized := _normalize(term))
                    and normalized not in GENERIC_NEGATIVE_TERMS
                )
                for pid in self.catalog_ids
            ], dtype=bool)
            eligibility = type("EligibilityView", (), {
                "mask": hard_mask & negative_mask,
                "hard_mask": hard_mask,
                "negative_mask": negative_mask,
                "hard_eligible_count": int(np.count_nonzero(hard_mask)),
                "negative_filtered_count": int(np.count_nonzero(hard_mask & ~negative_mask)),
            })()
            fts = type("FTSView", (), {"row_indices": (), "and_count": 0, "or_count": 0})()
        eligible = np.flatnonzero(eligibility.mask)
        eligible_fts = [row for row in fts.row_indices if eligibility.mask[row]]
        if not len(eligible):
            retrieval_route = "no_eligible"
            candidate_rows: list[int] = []
        elif len(eligible_fts) >= keyword_route_threshold:
            retrieval_route = "keyword"
            candidate_rows = eligible_fts
        else:
            retrieval_route = "vector_fallback"
            candidate_rows = sorted(
                eligible.tolist(), key=lambda row: (-float(s3[row]), self.catalog_ids[row])
            )[:VECTOR_FALLBACK_LIMIT]
        ranked = sorted(candidate_rows, key=lambda row: (-float(s3[row]), self.catalog_ids[row]))
        m0_ranked = sorted(eligible.tolist(), key=lambda row: (-float(s1[row]), self.catalog_ids[row]))
        full_m3_ranked = sorted(eligible.tolist(), key=lambda row: (-float(s3[row]), self.catalog_ids[row]))
        price_mask = np.ones(len(self.catalog_ids), dtype=bool)
        if state["price_max"] < 9999.0:
            price_mask &= np.isfinite(self.catalog_prices) & (self.catalog_prices <= state["price_max"])
        if session_id in getattr(self, "_forensic_capture_sessions", set()):
            self._forensic_ranking_snapshots.setdefault(session_id, []).append(
                ForensicRankingSnapshot(
                    session_id=session_id,
                    turn=int(turn),
                    canonical_state=_json_safe_state(state),
                    v1=v1,
                    v2=v2,
                    s1=s1,
                    s2=s2,
                    s3=s3,
                    price_mask=price_mask,
                    negative_mask=eligibility.negative_mask,
                    eligibility_mask=eligibility.mask,
                    m0_ranked_rows=np.asarray(m0_ranked, dtype=np.int64),
                    m3_ranked_rows=np.asarray(full_m3_ranked, dtype=np.int64),
                    gate_cosine=gate_cosine,
                    gate_passed=gate_passed,
                    current_weight=a,
                    memory_weight=b,
                )
            )
        chosen_rows = ranked[:max(0, int(top_k))]
        recommendations = [self.catalog_ids[row] for row in chosen_rows]
        state["seen_asins"].update(recommendations)
        active_constraints = [
            deepcopy(record) for record in state.get("constraint_provenance", [])
            if record.get("status") == "active"
        ]
        revoked_constraints = [
            deepcopy(record) for record in state.get("constraint_provenance", [])
            if record.get("status") == "revoked"
        ]
        state["debug_info"]["memory_trace"] = {
            "user_id": lifecycle["user_id"],
            "mode": live_mode.value,
            "buyer_mode": live_mode.value,
            "intent_mode": live_mode.value,
            "intent_source": state.get("intent_source", "session_default"),
            "caller_buyer_mode": None if buyer_mode is None else buyer_mode.value,
            "current_intent": query_text,
            "useful_slots": {
                "category": state["category"],
                "department": state["department"],
                "price_max": state["price_max"],
                "disclosed_slots": _json_safe_state(state["disclosed_slots"]),
                "negated_terms": sorted(state.get("negated_terms", set())),
            },
            "active_constraints": active_constraints,
            "revoked_constraints": revoked_constraints,
            "search_epoch": state.get("search_epoch", 0),
            "hard_conditions": {
                "price_max": state["price_max"],
                "target_department": state.get("target_department", ""),
                "min_avg_rating": state.get("min_avg_rating", 0.0),
                "min_rating_number": state.get("min_rating_number", 0),
                "store": state.get("store", ""),
            },
            "gate_cosine": gate_cosine,
            "gate_threshold": self.vector_memory_config.relevance_threshold,
            "gate_passed": gate_passed,
            "a": a,
            "b": b,
            "v1_available": True,
            "v2_available": v2 is not None,
            "prior_ltm_exists": v2 is not None,
            "memory_version": SNAPSHOT_VERSION,
            "memory_update_count": 0 if visible is None else visible.update_count,
            "embedding_space_id": self.embedding_space_id,
            "v2_embedding_space_id": None if visible is None else visible.embedding_space_id,
            "embedding_space_match": visible is None or visible.embedding_space_id == self.embedding_space_id,
            "catalog_rows_scored": len(self.catalog_ids),
            "price_filtered_count": int(np.count_nonzero(~price_mask)),
            "hard_eligible_count": eligibility.hard_eligible_count,
            "negative_filtered_count": eligibility.negative_filtered_count,
            "hard_eligible_count_after_negatives": len(eligible),
            "eligible_count": len(eligible),
            "fts_and_count": fts.and_count,
            "fts_or_count": fts.or_count,
            "fts_or_threshold": fts_or_threshold,
            "keyword_route_threshold": keyword_route_threshold,
            "fts5_count": len(eligible_fts),
            "retrieval_route": retrieval_route,
            "candidate_count": len(candidate_rows),
            "ltm_updated_after_turn": False,
            "ltm_updated_after_session": False,
            "memory_update_text": None,
            "returned": [
                {"parent_asin": self.catalog_ids[row],
                 "title": self.catalog_metadata[self.catalog_ids[row]]["title"],
                 "s1": float(s1[row]),
                 "s2": None if s2 is None else float(s2[row]), "s3": float(s3[row])}
                for row in chosen_rows
            ],
            "final_asins": recommendations,
        }

        # A zero-result hard mask is terminal for this turn; never leak an ineligible row.
        if retrieval_route == "no_eligible":
            active_hard: list[tuple[str, str]] = []
            if state["price_max"] < 9999.0: active_hard.append(("budget", f"maximum price ${state['price_max']:g}"))
            if state.get("target_department"): active_hard.append(("gender", f"department {state['target_department']}"))
            if state.get("min_avg_rating", 0.0) > 0: active_hard.append(("rating", f"minimum rating {state['min_avg_rating']:g}"))
            if state.get("min_rating_number", 0) > 0: active_hard.append(("reviews", f"minimum {state['min_rating_number']} reviews"))
            if state.get("store"): active_hard.append(("brand", f"brand/store {state['store']}"))
            if state.get("negated_terms"): active_hard.append(("other", "excluded terms " + ", ".join(sorted(state["negated_terms"]))))
            label = active_hard[0][1] if active_hard else "one of the active hard constraints"
            asked_attr = active_hard[0][0] if active_hard else "other"
            agent_message = f"No products satisfy all of those requirements. Would you like to relax {label}?"
            state["history"].append({"role": "assistant", "content": agent_message})
            state["debug_info"]["memory_trace"]["best_entropy_attributes"] = []
            debug_data = {
                "model": "Deterministic no-result clarification",
                "category": state["category"], "department": state["department"],
                "price_max": state["price_max"],
                "disclosed_slots": _json_safe_state(state["disclosed_slots"]),
                "asked_attributes": sorted(state.get("asked_attributes", set())),
                "best_entropy_attrs": [],
                "negated_terms": sorted(state.get("negated_terms", set())),
                "accumulated_terms": list(state.get("accumulated_terms", [])),
                "stashed_terms": list(state.get("stashed_terms", [])),
                "constraint_provenance": deepcopy(state.get("constraint_provenance", [])),
                "search_epoch": state.get("search_epoch", 0),
                "fts5_count": len(eligible_fts), "vector_fallback": False,
                "memory_trace": deepcopy(state["debug_info"]["memory_trace"]),
            }
            return {"message": agent_message, "ask_attribute": asked_attr, "recommendations": [], "debug": debug_data}

        # 4. Generate the conversational response without changing ranking.
        avoid_attrs = set(state["disclosed_slots"].keys()) | state.get("asked_attributes", set())
        if state["category"]:
            avoid_attrs.add("category")
        if state["department"]:
            avoid_attrs.add("department")

        if state.get("target_department"): avoid_attrs.add("gender")
        if state.get("min_avg_rating", 0.0) > 0: avoid_attrs.add("rating")
        if state.get("min_rating_number", 0) > 0: avoid_attrs.add("reviews")
        if state.get("price_max", 9999.0) < 9000.0: avoid_attrs.add("budget")
        if state.get("store"): avoid_attrs.add("brand")

        all_attrs = set(ATTRIBUTE_ORDER)
        remaining_attrs = all_attrs - avoid_attrs
        if hasattr(self, "catalogue"):
            best_attrs = select_best_attributes(
                self.catalogue, recommendations, remaining_attrs,
                top_n=2, intent_mode=live_mode.value,
            )
        else:
            priority = BUYING_ATTRIBUTE_ORDER if live_mode is BuyerMode.BUYING else BROWSING_ATTRIBUTE_ORDER
            best_attrs = [attr for attr in priority if attr in remaining_attrs][:2]
            best_attrs += ["other"] * (2 - len(best_attrs))
        selected_attrs = [attr for attr in best_attrs if attr != "other"]
        best_attrs_str = " and ".join(f"'{attr}'" for attr in selected_attrs) or "'other'"
        state["debug_info"]["memory_trace"]["best_entropy_attributes"] = list(best_attrs)

        # Format top matching products (ASIN + title) for generator context
        recs_meta = [self.catalog_metadata[rid] for rid in recommendations[:10]]
        candidate_products_str = "\n".join(
            f"- {meta['title']} (ASIN: {rid})" for rid, meta in zip(recommendations[:10], recs_meta)
        )

        sys_prompt = (
            "You are a helpful e-commerce shopping copilot. The user is looking for a product.\n"
            "Based on the conversation history and the current state, generate a conversational response.\n\n"
            f"Active search filters: Category: {state['category']}, Department: {state['department']}, Max Price: {state['price_max']}\n"
            f"Attributes already specified by user: { {k: list(v) for k, v in state['disclosed_slots'].items()} }\n"
            f"Negated terms: {list(state.get('negated_terms', set()))}\n\n"
            "CRITICAL RULES:\n"
            "1. Do not ask the user about any attribute they have already specified or you have already asked about.\n"
            f"Avoid: {', '.join(sorted(list(avoid_attrs)))}\n"
            f"2. Ask naturally about every entropy-selected attribute listed here: {best_attrs_str}. If two are listed, the response must ask about both. If only 'other' is listed, ask about another relevant preference.\n"
            "3. Keep your response short, natural, and conversational (1-2 sentences). Do not include any JSON formatting, raw tags, or markers. Just reply normally to the shopper.\n\n"
            "Input Candidate Products:\n"
            f"{candidate_products_str}\n"
        )

        history_str = ""
        for msg in state["history"][-4:]:
            role = "Customer" if msg["role"] == "user" else "Copilot"
            history_str += f"{role}: {msg['content']}\n"

        prompt = f"Dialogue history:\n{history_str}\n\nCopilot Response (direct text reply):"

        res_text = self._call_llm(prompt, sys_prompt, session_id=session_id, response_json=False)

        agent_message = res_text.strip()
        detected_asked = self._extract_asked_attributes(agent_message, state)
        asked_attr = next((attr for attr in ATTRIBUTE_ORDER if attr in detected_asked), "other")

        state["history"].append({"role": "assistant", "content": agent_message})

        if "asked_attributes" not in state:
            state["asked_attributes"] = set()

        for attr in detected_asked:
            if attr in all_attrs:
                state["asked_attributes"].add(attr)

        # Populate debug data
        state["debug_info"]["model"] = "DeepSeek-Chat" if os.environ.get("DEEPSEEK_API_KEY") else "GPT-4o-Mini"
        state["debug_info"]["system_prompt"] = sys_prompt
        state["debug_info"]["user_prompt"] = prompt

        debug_data = {
            "model": state["debug_info"].get("model", "None"),
            "system_prompt": state["debug_info"].get("system_prompt", ""),
            "user_prompt": state["debug_info"].get("user_prompt", ""),
            "category": state["category"],
            "department": state["department"],
            "price_max": state["price_max"],
            "disclosed_slots": {k: list(v) if isinstance(v, set) else v for k, v in state["disclosed_slots"].items()},
            "asked_attributes": list(state.get("asked_attributes", set())),
            "best_entropy_attrs": list(best_attrs),
            "negated_terms": list(state.get("negated_terms", set())),
            "accumulated_terms": list(state.get("accumulated_terms", [])),
            "stashed_terms": list(state.get("stashed_terms", [])),
            "constraint_provenance": deepcopy(state.get("constraint_provenance", [])),
            "search_epoch": state.get("search_epoch", 0),
            "intent_mode": live_mode.value,
            "intent_source": state.get("intent_source", "session_default"),
            "fts5_count": len(eligible_fts),
            "vector_fallback": retrieval_route == "vector_fallback",
            "memory_trace": deepcopy(state["debug_info"]["memory_trace"]),
        }

        # Print Hybrid Telemetry to terminal
        if emit_debug:
            print("\n" + "="*80)
            print(f" [DEMO TRACE] Turn: {turn} | Session: {session_id}")
            print("="*80)
            print(f"User:             {lifecycle['user_id']}")
            print(f"Live intent:      {debug_data['memory_trace']['intent_mode']}")
            print(f"Catalog scored:   {debug_data['memory_trace']['catalog_rows_scored']} products")
            print(f"Eligible rows:    {debug_data['memory_trace']['eligible_count']}")
            print(f"Memory gate:      {debug_data['memory_trace']['gate_passed']}")
            print(f"Weights a/b:      {a:.2f}/{b:.2f}")
            print(f"Current intent:   {query_text}")
            print("-"*80)

        return {
            "message": agent_message,
            "ask_attribute": asked_attr,
            "recommendations": [{"parent_asin": r} for r in recommendations],
            "debug": debug_data
        }
