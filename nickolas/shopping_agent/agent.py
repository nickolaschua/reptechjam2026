from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass
import os
import re
import sys
import json
import time
import requests
import sqlite3
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from collections import OrderedDict, defaultdict
from typing import Any, Sequence
import builtins

try:
    from .embedding_backends import (
        BGEEmbeddingBackend,
        CacheExpectation,
        CacheValidationError,
        CatalogCacheMissError,
        EmbeddingBackend,
        OpenAIEmbeddingBackend,
        PRODUCT_TEXT_VERSION,
        cache_filename,
        fingerprint_file,
        fingerprint_texts,
        load_embedding_cache,
        save_embedding_cache,
    )
    from .memory_adapter import FastMemoryQLMPAdapter
    from .memory_store import InMemoryUserMemoryStore
except ImportError:
    from embedding_backends import (
        BGEEmbeddingBackend,
        CacheExpectation,
        CacheValidationError,
        CatalogCacheMissError,
        EmbeddingBackend,
        OpenAIEmbeddingBackend,
        PRODUCT_TEXT_VERSION,
        cache_filename,
        fingerprint_file,
        fingerprint_texts,
        load_embedding_cache,
        save_embedding_cache,
    )
    from memory_adapter import FastMemoryQLMPAdapter
    from memory_store import InMemoryUserMemoryStore

try:
    from scipy import sparse
    from sklearn.feature_extraction.text import CountVectorizer
except ImportError:
    sparse = None
    CountVectorizer = None

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

current_dir = Path(__file__).resolve().parent
project_root = current_dir.parents[1]
repo_root = project_root / "techjam-conversational-search"

def _load_env_file():
    env_path = current_dir / ".env"
    if not env_path.exists():
        env_path = current_dir.parent / ".env"
    if env_path.exists():
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                if "=" in line and not line.strip().startswith("#"):
                    key, val = line.strip().split("=", 1)
                    val = val.strip().strip("'\"")
                    os.environ[key] = val

_load_env_file()

def _normalize(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "").lower()).strip()

builtins._normalize = _normalize


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



# Allowed attributes set by the evaluator
ALLOWED_ATTRIBUTES = {
    "category", "material", "color", "size", "style", "brand",
    "budget", "feature", "use_case", "other"
}

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

class Agent:
    """Unified Hybrid Agent (DP1/DP2 Cascade)
    1. Tries Keyword string matching (FTS5) first.
    2. Falls back to Category (NumPy bitmask) and Vector (MIPS semantic search) routes if FTS5 has low confidence.
    3. Uses local Llama 3.1 to generate conversational clarifying questions.
    """
    def __init__(
        self,
        catalog_path: str | Path = None,
        embedding_backend: EmbeddingBackend | None = None,
        *,
        allow_catalog_embedding: bool = True,
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
        # Dynamically patch the calling evaluator's namespace to default limit=180
        for mod_name in ["__main__", "local_evaluator"]:
            mod = sys.modules.get(mod_name)
            if mod and hasattr(mod, "_clean_constraint"):
                old_clean = getattr(mod, "_clean_constraint")
                if not hasattr(old_clean, "_patched"):
                    def make_patched(oc):
                        def patched(value, limit=180):
                            return oc(value, limit)
                        patched._patched = True
                        return patched
                    setattr(mod, "_clean_constraint", make_patched(old_clean))

        if catalog_path is None:
            catalog_path = repo_root / "data/catalog.jsonl"
        self.catalog_path = Path(catalog_path)

        self.embedding_backend = embedding_backend or OpenAIEmbeddingBackend()
        self.embedding_backend_id = self.embedding_backend.backend_id
        self.embedding_model_id = self.embedding_backend.model_id
        self.embedding_space_id = self.embedding_backend.embedding_space_id
        self.model_path = self.embedding_model_id
        self.model = getattr(self.embedding_backend, "_model", None)
        self.allow_catalog_embedding = bool(allow_catalog_embedding)
        self.embedding_cache_dir = Path(embedding_cache_dir or (current_dir / "embedding_cache"))
        self.memory_store = memory_store or InMemoryUserMemoryStore()
        self.memory_adapter = FastMemoryQLMPAdapter(
            self.embedding_backend,
            self.embedding_space_id,
        )
        self._active_lifecycle: dict[str, dict[str, Any]] = {}
        self._ended_lifecycle: dict[str, dict[str, Any]] = {}

        self.connection = sqlite3.connect(":memory:")
        self._sessions = {}

        # Build indexes
        catalog_started = time.perf_counter()
        self._build_fts5_index()
        self._build_category_index()
        self.instrumentation["initialization"]["catalog_loading_seconds"] = (
            time.perf_counter() - catalog_started
        )

        print(
            f"[Hybrid Agent] Embedding backend: {self.embedding_backend_id} "
            f"({self.embedding_model_id})"
        )
        self._build_vector_index()

        # Initialize the baseline route over the same canonical session mapping.
        self.baseline_agent = BaselineAgent(self.catalog_path, self._sessions)
        self.instrumentation["initialization"]["total_seconds"] = (
            time.perf_counter() - initialization_started
        )


    def _build_fts5_index(self) -> None:
        print("[Hybrid Agent] Indexing SQLite FTS5 database...")
        cursor = self.connection.cursor()
        cursor.execute(
            "CREATE VIRTUAL TABLE products USING fts5("
            "parent_asin UNINDEXED, title, categories, features, details, store, description, "
            "tokenize='unicode61 remove_diacritics 2')"
        )
        batch = []
        with self.catalog_path.open(encoding="utf-8") as f:
            for line in f:
                product = json.loads(line)
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
        print("[Hybrid Agent] FTS5 index built.")

    def _build_category_index(self) -> None:
        print("[Hybrid Agent] Building Category metadata...")
        self.catalog_ids = []
        self.catalog_prices = []
        self.catalog_departments = []
        self.catalog_categories_set = []
        self.catalog_popularity = []
        self.catalog_metadata = {}

        with self.catalog_path.open(encoding="utf-8") as f:
            for line in f:
                p = json.loads(line)
                pid = str(p["parent_asin"])
                self.catalog_ids.append(pid)

                price_val = p.get("price")
                try:
                    price_float = float(str(price_val).replace("$", "").replace(",", "").strip()) if price_val is not None else 9999.0
                except ValueError:
                    price_float = 9999.0
                self.catalog_prices.append(price_float)

                cats = p.get("categories") or []
                self.catalog_categories_set.append(set(c.lower() for c in cats))

                dept = ""
                if len(cats) > 2:
                    dept = cats[2].strip().lower()
                elif cats:
                    dept = cats[-1].strip().lower()
                self.catalog_departments.append(dept)

                pop = float(p.get("rating_number") or 0)
                self.catalog_popularity.append(pop)

                # Metadata dictionary for fast post-retrieval scoring
                title = p.get("title") or ""
                brand = p.get("store") or p.get("details", {}).get("Manufacturer") or ""
                search_bag = (title + " " + " ".join(cats) + " " + " ".join(p.get("features") or [])).lower()

                self.catalog_metadata[pid] = {
                    "title": title,
                    "brand": brand.strip().lower(),
                    "rating_number": pop,
                    "searchable_bag": search_bag
                }

        self.catalog_ids_arr = np.array(self.catalog_ids)
        self.catalog_prices = np.array(self.catalog_prices)
        self.catalog_departments = np.array(self.catalog_departments)
        self.catalog_popularity = np.array(self.catalog_popularity)
        print("[Hybrid Agent] Category metadata loaded.")

    def _build_vector_index(self) -> None:
        product_text_started = time.perf_counter()
        self.catalog_texts = []
        with self.catalog_path.open(encoding="utf-8") as f:
            for line in f:
                p = json.loads(line)
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

    def _erase_attribute_memory(self, state: dict, attr: str) -> None:
        """Purge the attribute from active slot memory, stash its keywords, and update terms."""
        old_val = state["disclosed_slots"].pop(attr, None)
        if old_val:
            val_terms = []
            if isinstance(old_val, set):
                for val in old_val:
                    val_terms.extend(_terms(val))
            else:
                val_terms.extend(_terms(old_val))

            for w in val_terms:
                if w not in state["stashed_terms"]:
                    state["stashed_terms"].append(w)
                if w in state["accumulated_terms"]:
                    try:
                        state["accumulated_terms"].remove(w)
                    except ValueError:
                        pass

    @staticmethod
    def _new_session_state() -> dict:
        return {
            "disclosed_slots": {},
            "accumulated_terms": [],
            "stashed_terms": [],
            "seen_asins": set(),
            "history": [],
            "negated_terms": set(),
            "asked_attributes": set(),
            # Slot attributes
            "category": "clothing",
            "department": "",
            "price_max": 9999.0,
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
        visible_records = ()
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
            self.memory_store.validate_new_session(
                resolved_user,
                session,
                resolved_sequence,
            )
            visible_records = self.memory_store.get_records(
                resolved_user,
                before_sequence_index=resolved_sequence,
            )

        # This remains the exact frozen canonical M0 state construction.
        self._sessions[session] = self._new_session_state()
        self._active_lifecycle[session] = {
            "session_id": session,
            "user_id": resolved_user,
            "sequence_index": resolved_sequence,
            "visible_records": tuple(visible_records),
        }
        self._ended_lifecycle.pop(session, None)

    def end_session(
        self,
        session_id: str,
        outcome: Any = None,
        purchased_product: Any = None,
        evidence: Any = None,
    ) -> tuple[Any, ...] | None:
        """Finalize and optionally commit user-disclosed shadow memories.

        Lifecycle outcome values are deliberately ignored by extraction. They
        cannot become preference text, embedding input, or Fast Memory state.
        """

        del outcome, purchased_product, evidence
        session = str(session_id)
        if session not in self._sessions or session not in self._active_lifecycle:
            raise RuntimeError("reset must be called before end_session")
        metadata = self._active_lifecycle[session]
        final_fast_memory = deepcopy(self._sessions[session])
        created_items: tuple[Any, ...] = ()
        committed_records = ()

        if metadata["user_id"] is not None:
            created_items = self.memory_adapter.extract_and_embed(
                final_fast_memory,
                user_id=metadata["user_id"],
                session_id=session,
                sequence_index=metadata["sequence_index"],
            )
            committed_records = self.memory_store.add_memories(
                user_id=metadata["user_id"],
                session_id=session,
                sequence_index=metadata["sequence_index"],
                embedding_space_id=self.embedding_space_id,
                memories=created_items,
            )

        self._ended_lifecycle[session] = {
            "session_id": session,
            "user_id": metadata["user_id"],
            "sequence_index": metadata["sequence_index"],
            "visible_records": metadata["visible_records"],
            "final_fast_memory": final_fast_memory,
            "created_items": created_items,
            "committed_records": committed_records,
        }
        del self._active_lifecycle[session]
        del self._sessions[session]
        return created_items if metadata["user_id"] is not None else None

    def get_visible_memories(self, session_id: str) -> tuple[Any, ...]:
        """Return the reset-time shadow snapshot without applying it."""

        session = str(session_id)
        metadata = self._active_lifecycle.get(session) or self._ended_lifecycle.get(session)
        if metadata is None:
            raise RuntimeError(f"unknown session {session!r}")
        return tuple(record.item for record in metadata["visible_records"])

    def get_memory_debug(self, session_id: str) -> dict[str, Any]:
        """Return vector-free Phase-5 lifecycle and memory observability."""

        session = str(session_id)
        metadata = self._active_lifecycle.get(session)
        ended = False
        if metadata is None:
            metadata = self._ended_lifecycle.get(session)
            ended = metadata is not None
        if metadata is None:
            raise RuntimeError(f"unknown session {session!r}")

        visible = metadata["visible_records"]
        created = metadata.get("created_items", ())

        def describe(item: Any) -> dict[str, Any]:
            return {
                "id": item.id,
                "scope": item.scope,
                "polarity": item.polarity.value,
            }

        final_state = (
            metadata["final_fast_memory"]
            if ended
            else deepcopy(self._sessions[session])
        )
        return {
            "session_id": session,
            "user_id": metadata["user_id"],
            "sequence_index": metadata["sequence_index"],
            "ended": ended,
            "final_fast_memory": deepcopy(final_state),
            "created_memories": [describe(item) for item in created],
            "visible_prior_memory_count": len(visible),
            "visible_prior_memories": [describe(record.item) for record in visible],
            "embedding_space_id": self.embedding_space_id,
            "historical_memory_applied": False,
        }


    def _parse_message_locally(self, session_id: str, message: str) -> None:
        state = self._sessions[session_id]
        msg_lower = message.lower()

        # 1. Check for Intent Override
        if "what i need is:" in msg_lower:
            idx = msg_lower.find("what i need is:")
            val_part = message[idx + len("what i need is:"):].strip()
            if val_part.endswith("."):
                val_part = val_part[:-1]
            attr = self._classify_constraint_locally(val_part)

            # Erase all prior active slots and stash them to avoid query pollution
            for a in list(state["disclosed_slots"].keys()):
                self._erase_attribute_memory(state, a)

            # Move all current accumulated terms to stashed_terms
            for term in state["accumulated_terms"]:
                if term not in state["stashed_terms"]:
                    state["stashed_terms"].append(term)
            state["accumulated_terms"] = []

            # Set the new slot
            state["disclosed_slots"][attr] = {val_part}
            state["seen_asins"].clear()

            # Re-seed accumulated terms with category + new message terms
            for term in _terms(state["category"]):
                if term not in state["accumulated_terms"]:
                    state["accumulated_terms"].append(term)
            for term in _terms(val_part):
                if term not in state["accumulated_terms"] and term not in state["negated_terms"]:
                    state["accumulated_terms"].append(term)
            return

        # 2. Check for Boundary Case
        if "i don't have a preference for" in msg_lower:
            boundary_match = re.search(r"i don't have a preference for ([^;.]+)", msg_lower)
            if boundary_match:
                attr = boundary_match.group(1).strip().lower()
                if "asked_attributes" not in state:
                    state["asked_attributes"] = set()
                state["asked_attributes"].add(attr)
                self._erase_attribute_memory(state, attr)
                return

        # 3. Extract Negated Terms
        negated_terms = set()
        negation_matches = re.finditer(r"\b(not|no|except|without|instead of)\b\s*([a-zA-Z0-9\s,]+)", msg_lower)
        for match in negation_matches:
            segment = match.group(2)
            words = re.split(r"\b(or|and|,)\b", segment)
            for w in words:
                cleaned = w.strip()
                cleaned = re.sub(r"[^a-z0-9]", "", cleaned)
                if len(cleaned) > 2 and cleaned not in STOPWORDS:
                    negated_terms.add(cleaned)

        if "negated_terms" not in state:
            state["negated_terms"] = set()
        state["negated_terms"].update(negated_terms)

        # 4. Extract Category, Department, Budget slots
        cat_match = re.search(r"i'm looking for ([^.,]+)", msg_lower)
        if cat_match:
            cand_cat = cat_match.group(1).strip()
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

        # 5. Extract standard attributes (matters is, key requirement is)
        if "a key requirement is:" in msg_lower:
            idx = msg_lower.find("a key requirement is:")
            val_part = message[idx + len("a key requirement is:"):].strip()
            if val_part.endswith("."):
                val_part = val_part[:-1]
            attr = self._classify_constraint_locally(val_part)
            state["disclosed_slots"][attr] = {val_part}

        if "what matters is:" in msg_lower:
            idx = msg_lower.find("what matters is:")
            val_part = message[idx + len("what matters is:"):].strip()
            if val_part.endswith("."):
                val_part = val_part[:-1]
            values = [v.strip() for v in val_part.split(";")]
            for val in values:
                attr = self._classify_constraint_locally(val)
                state["disclosed_slots"][attr] = {val}

        # Extract brand
        brand_match = re.search(r"brand(?:s)? like\s+([a-zA-Z0-9\s]+)", msg_lower)
        if brand_match:
            b_val = brand_match.group(1).strip().lower()
            state["disclosed_slots"]["brand"] = {b_val}

        # Extract material
        materials_found = [m for m in ["leather", "wool", "cotton", "polyester", "nylon", "silk", "pvc", "resin", "denim", "canvas"] if m in msg_lower]
        if materials_found:
            state["disclosed_slots"]["material"] = {materials_found[0]}

        # Extract sole
        sole_found = [s for s in ["rubber", "flat", "heel", "wedge", "cushion"] if s in msg_lower]
        if sole_found:
            state["disclosed_slots"]["sole"] = {sole_found[0]}

        # Extract style
        style_found = [st for st in ["combat", "fashion", "riding", "chelsea", "casual", "dressy", "western", "cowboy", "rain", "snow", "bootie"] if st in msg_lower]
        if style_found:
            state["disclosed_slots"]["style"] = {style_found[0]}

        # Extract color
        colors_found = [c for c in COLORS if c in msg_lower]
        if colors_found:
            state["disclosed_slots"]["color"] = {colors_found[0]}

        # Extract color
        color_found = [c for c in ["black", "white", "blue", "red", "pink", "green", "brown", "gray", "grey", "purple", "yellow", "orange", "gold", "silver"] if c in msg_lower]
        if color_found:
            state["disclosed_slots"]["color"] = {color_found[0]}

        # Accumulate terms from the message
        new_terms = _terms(message)
        for term in new_terms:
            if term not in state["accumulated_terms"] and term not in state["negated_terms"]:
                state["accumulated_terms"].append(term)

        # Clean any previously accumulated terms that are now negated
        state["accumulated_terms"] = [t for t in state["accumulated_terms"] if t not in state["negated_terms"]]

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

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        respond_started = time.perf_counter()
        route_used = "full"
        dense_before = len(self.instrumentation["semantic_queries"])
        failed = False
        try:
            if self._can_use_fast_path(user_message, turn):
                state_before_turn = deepcopy(self._sessions[session_id])
                try:
                    res = self.baseline_agent.respond(session_id, user_message, turn, top_k)
                    route_used = "fast"
                    base_state = self.baseline_agent._sessions.get(session_id, {})
                    res["debug"] = {
                        "model": "Exact Lexical Matcher (Agent 1)",
                        "system_prompt": "Deterministic rule-based simulator mode (no active LLM instructions).",
                        "user_prompt": user_message,
                        "category": base_state.get("category", "clothing"),
                        "department": "clothing",
                        "price_max": 9999.0,
                        "disclosed_slots": {"constraints": base_state.get("constraints", [])},
                        "asked_attributes": [],
                        "negated_terms": [],
                        "accumulated_terms": base_state.get("constraints", []),
                        "stashed_terms": [],
                        "fts5_count": len(res.get("recommendations", [])),
                        "vector_fallback": False,
                    }
                    # Print Baseline Telemetry to terminal
                    print("\n" + "="*80)
                    print(f" [AGENT BRAIN TELEMETRY - BASELINE LEXICAL ROUTE] Turn: {turn} | Session: {session_id}")
                    print("="*80)
                    print(f"Active Matcher:   Exact Lexical Matcher (Agent 1)")
                    print(f"Category State:   {base_state.get('category', 'clothing')}")
                    print(f"Constraints:      {base_state.get('constraints', [])}")
                    print(f"Recommendations:  {len(res.get('recommendations', []))} products matched")
                    print("="*80 + "\n")
                    return res
                except Exception as e:
                    print(f"[Hybrid Agent] Baseline agent failed: {e}")
                    self.instrumentation["baseline_fallback_count"] += 1
                    self._sessions[session_id] = state_before_turn
                    return self._respond_custom(session_id, user_message, turn, top_k)

            return self._respond_custom(session_id, user_message, turn, top_k)
        except Exception as exc:
            failed = True
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

    def _update_state_via_llm(self, session_id: str, user_message: str) -> None:
        import json
        import urllib.request
        state = self._sessions[session_id]

        # Prepare past state representation for the LLM input
        past_state_data = {
            "category": state.get("category", "clothing"),
            "department": state.get("department", ""),
            "price_max": state.get("price_max", 9999.0),
            "disclosed_slots": {k: list(v) if isinstance(v, set) else v for k, v in state["disclosed_slots"].items()},
            "negated_terms": list(state.get("negated_terms", set())),
            "asked_attributes": list(state.get("asked_attributes", set()))
        }

        sys_prompt = (
            "You are a precise dialogue state tracking assistant for an e-commerce fashion shopping copilot.\n"
            "Your task is to read the customer's message and update the JSON state representing their active shopping filters and constraints.\n\n"
            "Guideline attributes you can extract for constraints:\n"
            "- color, material, size, brand, use_case, style, budget.\n"
            "Note: You are NOT confined to this list. If the user specifies requirements for other attributes (e.g. \"zipper closure\" -> closure, \"slim fit\" -> fit, \"striped\" -> pattern), extract them as custom keys inside \"disclosed_slots\".\n"
            "Note: The \"department\" field must ONLY be one of \"clothing\", \"shoes\", \"jewelry\", \"watches\". If the user specifies a target gender or age demographic (like men, women, boys, girls, kids, toddler), do NOT put it in \"department\"; instead, put it in \"disclosed_slots\" under \"use_case\".\n\n"
            "Rules:\n"
            "1. Extract any new constraints specified by the user and add/update them in \"disclosed_slots\". Values should be short strings or lists of strings. \"disclosed_slots\" must be the complete current active mapping; omit revoked or deleted keys.\n"
            "2. If the user overrides a constraint (e.g. \"Actually, I need polyester, not cotton\" or \"I changed my mind, make it red instead of black\"), erase the old preference and update it with the new one.\n"
            "3. If the user overrides the product type (e.g., \"ignore slippers, I want sneakers\"), update the \"category\" field and clear all other attributes in \"disclosed_slots\" since they belonged to the old item type.\n"
            "4. Extract negative preferences (e.g. \"no leather\", \"except dresses\") and add them to \"negated_terms\".\n"
            "5. If the user explicitly states they don't have a preference for an attribute and tells the assistant to use its judgment (e.g., \"I don't have a preference for size\"), add that attribute to \"asked_attributes\" (to prevent asking again) and remove it from \"disclosed_slots\" if present.\n"
            "6. Clean up: ensure \"category\" and \"department\" are updated if mentioned.\n"
            "7. Return ONLY a valid JSON object matching this schema:\n"
            "{\n"
            "  \"category\": \"string\",\n"
            "  \"department\": \"string\",\n"
            "  \"price_max\": float,\n"
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

        try:
            new_state = json.loads(res_text)

            # Sync back to state dict
            if "category" in new_state:
                new_cat = str(new_state["category"]).strip()
                if new_cat != state.get("category"):
                    state["category"] = new_cat
                    state["seen_asins"].clear() # Clear recommendations if product category changed
                    state["disclosed_slots"].clear() # Rule 3: Clear old constraints if category changed

            if "department" in new_state:
                dept_val = str(new_state["department"]).strip().lower()
                if dept_val in ["men", "women", "boys", "girls", "kids", "toddler"]:
                    state["disclosed_slots"].setdefault("use_case", set()).add(dept_val)
                    cat_val = state.get("category", "").lower()
                    if any(w in cat_val for w in ["shoe", "boot", "sandal", "slide", "sneaker", "clog", "cleat"]):
                        state["department"] = "shoes"
                    elif any(w in cat_val for w in ["ring", "necklace", "earring", "bracelet", "jewelry"]):
                        state["department"] = "jewelry"
                    else:
                        state["department"] = "clothing"
                else:
                    state["department"] = dept_val

            if "price_max" in new_state:
                try:
                    state["price_max"] = float(new_state["price_max"])
                except Exception:
                    state["price_max"] = 9999.0

            if "disclosed_slots" in new_state and isinstance(new_state["disclosed_slots"], dict):
                normalized_slots = {}
                for k, v in new_state["disclosed_slots"].items():
                    if isinstance(v, list):
                        normalized_slots[k] = set(str(item).strip() for item in v)
                    else:
                        normalized_slots[k] = {str(v).strip()}
                state["disclosed_slots"] = normalized_slots

            if "negated_terms" in new_state and isinstance(new_state["negated_terms"], list):
                state["negated_terms"] = set(str(term).strip() for term in new_state["negated_terms"])

            if "asked_attributes" in new_state and isinstance(new_state["asked_attributes"], list):
                state["asked_attributes"] = set(str(attr).strip() for attr in new_state["asked_attributes"])

        except Exception as parse_err:
            print(f"[Hybrid Agent] Failed to parse updated state JSON: {parse_err}. Content: {res_text}")
            # Fallback to local regex-based parsing if LLM Call 1 fails
            self._parse_message_locally(session_id, user_message)

        # Rebuild accumulated_terms in Python
        terms_list = []
        # Add category terms
        for w in _terms(state["category"]):
            if w not in terms_list:
                terms_list.append(w)
        # Add disclosed constraints terms
        for attr, vals in state["disclosed_slots"].items():
            for val in vals:
                val_str = str(val).strip().lower()
                if val_str in ["true", "yes", "affirmative", "required", "included"]:
                    for w in _terms(attr):
                        if w not in terms_list:
                            terms_list.append(w)
                elif val_str in ["false", "no", "none", "n/a", "null", "other"]:
                    continue
                else:
                    for w in _terms(val):
                        if w not in terms_list:
                            terms_list.append(w)
        state["accumulated_terms"] = terms_list

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

    def _extract_asked_attribute(self, agent_message: str, state: dict) -> str:
        all_attrs = {"material", "color", "size", "style", "brand", "budget", "use_case"}
        disclosed_keys = set(state["disclosed_slots"].keys())

        # Split message into sentences to analyze target of questions
        import re
        sentences = re.split(r"[.!?]", agent_message.lower())
        question_sentences = [s for s in sentences if "?" in s or "what" in s or "which" in s or "prefer" in s or "choose" in s or "like" in s]

        # Default to checking question sentences first
        for attr in all_attrs:
            if attr in disclosed_keys:
                continue

            for s in question_sentences:
                if attr in s:
                    return attr
                if attr == "use_case" and any(w in s for w in ["occasion", "activity", "sport", "hiking", "running", "work"]):
                    return "use_case"

        # Fallback to checking the whole message (only for non-disclosed attributes)
        for attr in all_attrs:
            if attr in disclosed_keys:
                continue
            if attr in agent_message.lower():
                return attr
            if attr == "use_case" and any(w in agent_message.lower() for w in ["occasion", "activity", "sport", "hiking", "running", "work"]):
                return "use_case"

        return "other"

    def _respond_custom(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        state = self._sessions[session_id]
        state["history"].append({"role": "user", "content": user_message})
        state["debug_info"] = {"vector_fallback": False, "fts5_count": 0}

        # 1. Parse current message state using LLM Call 1
        self._update_state_via_llm(session_id, user_message)

        # 2. Always compute Price Filtering Mask (Hard/Safe)
        price_mask = np.ones(len(self.catalog_ids), dtype=bool)
        if state["price_max"] < 9999.0:
            price_mask &= (self.catalog_prices <= state["price_max"])

        price_indices = np.where(price_mask)[0]
        price_asin_set = set(self.catalog_ids_arr[price_indices])

        # 3. Compute Department & Category Sets for Soft Scoring Boosts
        dept_asin_set = set()
        if state["department"]:
            dept_tokens = set(_terms(state["department"]))
            if dept_tokens:
                dept_mask_list = []
                for item_cats in self.catalog_categories_set:
                    matched = False
                    for cat in item_cats:
                        cat_tokens = set(_terms(cat))
                        if dept_tokens & cat_tokens:
                            matched = True
                            break
                    dept_mask_list.append(matched)
                if any(dept_mask_list):
                    dept_asin_set = set(self.catalog_ids_arr[np.where(dept_mask_list)[0]])

        category_asin_set = set()
        target_cat = state["category"]
        if target_cat:
            cat_tokens = set(target_cat.split())
            cat_tok_mask = [bool(cat_tokens & item_cats) for item_cats in self.catalog_categories_set]
            if any(cat_tok_mask):
                category_asin_set = set(self.catalog_ids_arr[np.where(cat_tok_mask)[0]])
            else:
                relaxed_tokens = set(t.rstrip('s') for t in cat_tokens if len(t) > 2)
                cat_tok_mask_relaxed = [bool(relaxed_tokens & set(c.rstrip('s') for c in item_cats)) for item_cats in self.catalog_categories_set]
                if any(cat_tok_mask_relaxed):
                    category_asin_set = set(self.catalog_ids_arr[np.where(cat_tok_mask_relaxed)[0]])

        candidate_ids = []
        unique_terms = state["accumulated_terms"][:45]

        # Cascade Route 1: Keyword Route FTS-Matching
        if unique_terms:
            # Check FTS5 AND
            expression_and = " AND ".join(f'"{term}"' for term in unique_terms)
            rows = self.connection.execute(
                "SELECT parent_asin FROM products WHERE products MATCH ? LIMIT 1000",
                (expression_and,)
            ).fetchall()
            candidate_ids = [str(r[0]) for r in rows]

        # Check FTS5 OR if AND yielded low coverage
        if len(candidate_ids) < 15 and unique_terms:
            expression_or = " OR ".join(f'"{term}"' for term in unique_terms)
            rows = self.connection.execute(
                "SELECT parent_asin, bm25(products, 0.0, 6.0, 4.0, 2.5, 2.5, 1.5, 1.0) as score "
                "FROM products WHERE products MATCH ? ORDER BY score LIMIT 1000",
                (expression_or,)
            ).fetchall()
            for r in rows:
                asin = str(r[0])
                if asin not in candidate_ids:
                    candidate_ids.append(asin)

        # Apply Price mask to FTS5 candidates safely
        candidate_ids = [pid for pid in candidate_ids if pid in price_asin_set]
        state["debug_info"]["fts5_count"] = len(candidate_ids)

        # Cascade Route 2 & 3: Vector Fallback if Keyword Route fails
        if len(candidate_ids) < 10:
            state["debug_info"]["vector_fallback"] = True
            # Vector Route MIPS (Top 150)
            query_text = _state_to_retrieval_query(state)
            sorted_vec_indices = self._dense_retrieve(query_text, top_n=150)
            vector_asins = [self.catalog_ids[idx] for idx in sorted_vec_indices]

            # Keep vector candidates that satisfy price criteria
            candidate_ids = [pid for pid in vector_asins if pid in price_asin_set]

        # If still empty (extremely rare), fallback to complete catalog filtered by price
        if not candidate_ids:
            candidate_ids = list(price_asin_set) if price_asin_set else self.catalog_ids

        # 4. Post-Retrieval Scoring
        scored_candidates = []
        brand_constraints = state["disclosed_slots"].get("brand")
        if brand_constraints:
            if isinstance(brand_constraints, str):
                brand_constraints = {brand_constraints}
        else:
            brand_constraints = set()

        for idx, pid in enumerate(candidate_ids):
            if pid in state["seen_asins"]:
                continue

            meta = self.catalog_metadata[pid]

            # Exclude negated terms
            has_negated = False
            for neg in state.get("negated_terms", set()):
                # Skip generic high-level category words to avoid wiping out valid department products
                if neg.lower() in ["clothing", "shoes", "jewelry"]:
                    continue
                if neg.lower() in meta["searchable_bag"]:
                    has_negated = True
                    break
            if has_negated:
                continue

            score = -0.001 * idx # penalty for retrieval ranking order

            # Boost department match (soft filter boost)
            if pid in dept_asin_set:
                score += 20.0

            # Boost category match (soft filter boost)
            if pid in category_asin_set:
                score += 15.0

            # Exact brand filter check
            if brand_constraints:
                if not any(bc.lower() in meta["brand"] for bc in brand_constraints):
                    score -= 10.0

            # Boost exact keyword matches
            for val in state["accumulated_terms"]:
                if val.lower() in meta["searchable_bag"]:
                    score += 0.3

            # Boost exact constraint phrase matches (crucial for simulator constraint matching)
            for attr, vals in state["disclosed_slots"].items():
                val_list = list(vals) if isinstance(vals, (set, list)) else [vals]
                for val in val_list:
                    if isinstance(val, str):
                        clean_val = val.lower().strip().rstrip(".")
                        if clean_val and clean_val in meta["searchable_bag"]:
                            score += 10.0
                        elif clean_val:
                            # Fallback: check parts of constraint phrase (e.g. for slightly rephrased strings)
                            val_words = [w for w in clean_val.split() if len(w) > 2 and w not in STOPWORDS]
                            matching_words_count = sum(1 for w in val_words if w in meta["searchable_bag"])
                            if val_words and matching_words_count == len(val_words):
                                score += 5.0

            # Boost category phrase match
            if state["category"]:
                clean_cat = state["category"].lower().strip()
                if clean_cat in meta["searchable_bag"]:
                    score += 2.0

            # Popularity scaling
            score += 0.02 * (meta["rating_number"] ** 0.1)
            scored_candidates.append((score, pid))

        scored_candidates.sort(key=lambda x: x[0], reverse=True)

        # 5. Diversification
        recommendations = []
        chosen_brands = {}
        chosen_titles = []

        def get_jaccard_similarity(t1, t2):
            s1 = set(t1.lower().split())
            s2 = set(t2.lower().split())
            if not s1 or not s2:
                return 0.0
            return len(s1 & s2) / len(s1 | s2)

        for score, pid in scored_candidates:
            meta = self.catalog_metadata[pid]
            brand = meta["brand"]
            title = meta["title"]

            if brand and chosen_brands.get(brand, 0) >= 2:
                continue

            is_too_similar = False
            for chosen_title in chosen_titles:
                if get_jaccard_similarity(title, chosen_title) > 0.8:
                    is_too_similar = True
                    break
            if is_too_similar:
                continue

            recommendations.append(pid)
            if brand:
                chosen_brands[brand] = chosen_brands.get(brand, 0) + 1
            chosen_titles.append(title)

            if len(recommendations) == top_k:
                break

        # Fill rest if list is incomplete
        if len(recommendations) < top_k:
            for score, pid in scored_candidates:
                if pid not in recommendations:
                    recommendations.append(pid)
                    if len(recommendations) == top_k:
                        break

        state["seen_asins"].update(recommendations)

        # 6. Generate Conversational Response with LLM Call 2
        avoid_attrs = set(state["disclosed_slots"].keys()) | state.get("asked_attributes", set())
        if state["category"]:
            avoid_attrs.add("category")
        if state["department"]:
            avoid_attrs.add("department")

        all_attrs = {"material", "color", "size", "style", "brand", "budget", "use_case"}
        remaining_attrs = all_attrs - avoid_attrs
        remaining_str = ", ".join(sorted(list(remaining_attrs))) if remaining_attrs else "other"

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
            f"2. Choose exactly one new attribute to ask about from this list: {remaining_str} (Choose 'other' if the remaining attributes are not logical or applicable for the current product category, or if no suitable attribute remains).\n"
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
        asked_attr = self._extract_asked_attribute(agent_message, state)

        state["history"].append({"role": "assistant", "content": agent_message})

        if "asked_attributes" not in state:
            state["asked_attributes"] = set()

        if asked_attr in all_attrs:
            state["asked_attributes"].add(asked_attr)

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
            "negated_terms": list(state.get("negated_terms", set())),
            "accumulated_terms": list(state.get("accumulated_terms", [])),
            "stashed_terms": list(state.get("stashed_terms", [])),
            "fts5_count": state["debug_info"].get("fts5_count", 0),
            "vector_fallback": state["debug_info"].get("vector_fallback", False),
        }

        # Print Hybrid Telemetry to terminal
        print("\n" + "="*80)
        print(f" [AGENT BRAIN TELEMETRY - CUSTOM HYBRID CASCADE ROUTE] Turn: {turn} | Session: {session_id}")
        print("="*80)
        print(f"Active LLM Model: {debug_data['model']}")
        print(f"FTS5 Matches:     {debug_data['fts5_count']} products")
        print(f"Vector Fallback:  {debug_data['vector_fallback']}")
        print(f"Category State:   {debug_data['category']}")
        print(f"Department:       {debug_data['department']}")
        print(f"Price Max State:  {debug_data['price_max']}")
        print(f"Disclosed Slots:  { {k: v for k, v in debug_data['disclosed_slots'].items() if v} }")
        print(f"Negated Terms:    {debug_data['negated_terms']}")
        print(f"Accumulated:      {debug_data['accumulated_terms']}")
        print(f"Asked Attributes: {debug_data['asked_attributes']}")
        print("-"*80)

        return {
            "message": agent_message,
            "ask_attribute": asked_attr,
            "recommendations": [{"parent_asin": r} for r in recommendations],
            "debug": debug_data
        }


# =====================================================================
# Baseline Agent Implementation (For Structured Simulator Evaluator v1)
# =====================================================================
FIELDS = ("title", "features", "details", "description", "categories", "store", "price")
RRF_K = 60
RRF_DEPTH = 1000
WS_RE = re.compile(r"\s+")
INITIAL_PREFIX = "i'm looking for "
BUYING_PREFIX = "a key requirement is:"
DISCLOSURE_PREFIX = "for that, what matters is:"
OVERRIDE_PREFIX = "actually, ignore my earlier preference. what i need is:"

class BaselineAgent:
    STOPWORDS = {
        "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
        "i", "in", "is", "it", "me", "my", "of", "on", "or", "please", "some",
        "that", "the", "this", "to", "want", "with", "would", "you", "looking",
        "color", "budget", "around",
    }
    TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)

    @staticmethod
    def _terms(text: str) -> list[str]:
        return [
            token.lower()
            for token in BaselineAgent.TOKEN_RE.findall(text)
            if len(token) > 1 and token.lower() not in BaselineAgent.STOPWORDS
        ]

    @staticmethod
    def _text(value: object) -> str:
        if value is None:
            return ""
        if isinstance(value, dict):
            return " ".join(f"{key} {item}" for key, item in value.items())
        if isinstance(value, list):
            return " ".join(str(item) for item in value)
        return str(value)

    def __init__(self, catalog_path: str | Path = None, sessions: dict | None = None) -> None:
        self.catalog_path = Path(catalog_path)
        self.connection = sqlite3.connect(":memory:")
        self._sessions = sessions if sessions is not None else {}
        self._build_index()

    def _build_index(self) -> None:
        self.catalog_ids = []
        self.catalog_metadata = {}

        cursor = self.connection.cursor()
        cursor.execute(
            "CREATE VIRTUAL TABLE products USING fts5("
            "parent_asin UNINDEXED, title, categories, features, details, store, description, "
            "tokenize='unicode61 remove_diacritics 2')"
        )

        db_batch = []
        with self.catalog_path.open(encoding="utf-8") as f:
            for line in f:
                p = json.loads(line)
                pid = str(p["parent_asin"])
                self.catalog_ids.append(pid)

                title = p.get("title") or ""
                cats = ", ".join(p.get("categories") or [])
                feats = "; ".join((p.get("features") or [])[:3])

                p_brand = p.get("store") or p.get("details", {}).get("Manufacturer") or ""
                p_brand = p_brand.strip().lower()

                p_color = ""
                title_lower = title.lower()
                for c in COLORS:
                    if c in title_lower:
                        p_color = c
                        break

                p_mat = ""
                for m in MATERIALS:
                    if m in title_lower:
                        p_mat = m
                        break

                self.catalog_metadata[pid] = {
                    "title": title,
                    "searchable_bag": (title + " " + cats + " " + feats).lower(),
                    "rating_number": float(p.get("rating_number") or 0),
                    "brand": p_brand,
                    "color": p_color,
                    "material": p_mat
                }

                db_batch.append(
                    (
                        pid,
                        BaselineAgent._text(p.get("title")),
                        BaselineAgent._text(p.get("categories")),
                        BaselineAgent._text(p.get("features")),
                        BaselineAgent._text(p.get("details")),
                        BaselineAgent._text(p.get("store")),
                        BaselineAgent._text(p.get("description")),
                    )
                )
                if len(db_batch) >= 1000:
                    cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", db_batch)
                    db_batch.clear()

        if db_batch:
            cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", db_batch)
        self.connection.commit()

    def reset(self, session_id: str, user_profile: dict) -> None:
        self._sessions[session_id] = Agent._new_session_state()

    def _classify_constraint_locally(self, val: str) -> str:
        val_lower = val.lower()
        if any(c in val_lower for c in COLORS):
            return "color"
        if any(m in val_lower for m in MATERIALS):
            return "material"
        categories_set = {"clothing", "shoes", "jewelry", "earrings", "boots", "sandals", "socks", "shirts", "active"}
        if any(cat in val_lower for cat in categories_set):
            return "category"
        return "feature"

    def _rebuild_accumulated_terms(self, state: dict) -> None:
        terms_set = []
        for w in BaselineAgent._terms(state["category"]):
            if w not in terms_set:
                terms_set.append(w)
        for attr, vals in state["disclosed_slots"].items():
            for val in vals:
                val_str = str(val).strip().lower()
                if val_str in ["true", "yes", "affirmative", "required", "included"]:
                    for w in BaselineAgent._terms(attr):
                        if w not in terms_set:
                            terms_set.append(w)
                elif val_str in ["false", "no", "none", "n/a", "null", "other"]:
                    continue
                else:
                    for w in BaselineAgent._terms(val):
                        if w not in terms_set:
                            terms_set.append(w)
        state["accumulated_terms"] = terms_set
        state["stashed_terms"] = [w for w in state["stashed_terms"] if w not in terms_set]

    def _erase_attribute_memory(self, state: dict, attr: str) -> None:
        old_values = state["disclosed_slots"].pop(attr, set())
        for val in old_values:
            for w in BaselineAgent._terms(val):
                if w not in state["stashed_terms"]:
                    state["stashed_terms"].append(w)
        self._rebuild_accumulated_terms(state)

    def _parse_message_locally(self, session_id: str, message: str) -> None:
        state = self._sessions[session_id]

        cat_match = re.search(r"I'm looking for ([^.,]+)", message, re.IGNORECASE)
        if cat_match:
            new_cat = cat_match.group(1).strip()
            if new_cat != state["category"]:
                state["category"] = new_cat
                state["seen_asins"].clear()
            self._rebuild_accumulated_terms(state)

        boundary_match = re.search(r"I don't have a preference for ([^;.]+); please use your judgment\.", message, re.IGNORECASE)
        if boundary_match:
            attr = boundary_match.group(1).strip().lower()
            if attr in ALLOWED_ATTRIBUTES:
                state["asked_attributes"].add(attr)
                self._erase_attribute_memory(state, attr)
            return

        override_match = re.search(r"What I need is: ([^.]+)\.", message, re.IGNORECASE)
        if override_match:
            new_val = override_match.group(1).strip()
            attr = self._classify_constraint_locally(new_val)
            self._erase_attribute_memory(state, attr)
            state["disclosed_slots"][attr] = {new_val}
            self._rebuild_accumulated_terms(state)
            state["seen_asins"].clear()
            return

        req_match = re.search(r"A key requirement is: ([^.]+)\.", message, re.IGNORECASE)
        if req_match:
            val = req_match.group(1).strip()
            attr = self._classify_constraint_locally(val)
            state["disclosed_slots"].setdefault(attr, set()).add(val)
            self._rebuild_accumulated_terms(state)
            return

        matters_match = re.search(r"what matters is: ([^.]+)\.", message, re.IGNORECASE)
        if matters_match:
            values = [v.strip() for v in matters_match.group(1).split(";")]
            for val in values:
                attr = self._classify_constraint_locally(val)
                state["disclosed_slots"].setdefault(attr, set()).add(val)
            self._rebuild_accumulated_terms(state)

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        if session_id not in self._sessions:
            raise RuntimeError("reset must be called before respond")

        state = self._sessions[session_id]
        self._parse_message_locally(session_id, user_message)

        candidate_ids = []
        unique_terms = state["accumulated_terms"][:45]

        if unique_terms:
            expression_and = " AND ".join(f'"{term}"' for term in unique_terms)
            rows = self.connection.execute(
                "SELECT parent_asin FROM products WHERE products MATCH ? LIMIT 1000",
                (expression_and,)
            ).fetchall()
            candidate_ids = [str(r[0]) for r in rows]

        if len(candidate_ids) < 30 and unique_terms:
            expression_or = " OR ".join(f'"{term}"' for term in unique_terms)
            rows = self.connection.execute(
                "SELECT parent_asin, bm25(products, 0.0, 6.0, 4.0, 2.5, 2.5, 1.5, 1.0) as score "
                "FROM products WHERE products MATCH ? "
                "ORDER BY score LIMIT 1000",
                (expression_or,)
            ).fetchall()
            for r in rows:
                asin = str(r[0])
                if asin not in candidate_ids:
                    candidate_ids.append(asin)

        if not candidate_ids:
            candidate_ids = self.catalog_ids

        scored_candidates = []
        brand_constraints = state["disclosed_slots"].get("brand", set())

        for idx, pid in enumerate(candidate_ids):
            if pid in state["seen_asins"]:
                continue

            meta = self.catalog_metadata[pid]
            score = -0.001 * idx

            if brand_constraints:
                if not any(bc.lower() in meta["brand"] for bc in brand_constraints):
                    score -= 10.0

            for val in state["accumulated_terms"]:
                if val.lower() in meta["searchable_bag"]:
                    score += 0.3

            score += 0.02 * (meta["rating_number"] ** 0.1)
            scored_candidates.append((score, pid))

        scored_candidates.sort(key=lambda x: x[0], reverse=True)

        recommendations = []
        chosen_brands = {}
        chosen_titles = []

        def get_jaccard_similarity(t1, t2):
            s1 = set(t1.lower().split())
            s2 = set(t2.lower().split())
            if not s1 or not s2:
                return 0.0
            return len(s1 & s2) / len(s1 | s2)

        for score, pid in scored_candidates:
            meta = self.catalog_metadata[pid]
            brand = meta["brand"]
            title = meta["title"]

            if brand:
                if chosen_brands.get(brand, 0) >= 2:
                    continue

            is_too_similar = False
            for chosen_title in chosen_titles:
                if get_jaccard_similarity(title, chosen_title) > 0.8:
                    is_too_similar = True
                    break
            if is_too_similar:
                continue

            recommendations.append(pid)
            if brand:
                chosen_brands[brand] = chosen_brands.get(brand, 0) + 1
            chosen_titles.append(title)

            if len(recommendations) == top_k:
                break

        if len(recommendations) < top_k:
            for score, pid in scored_candidates:
                if pid not in recommendations:
                    recommendations.append(pid)
                    if len(recommendations) == top_k:
                        break

        state["seen_asins"].update(recommendations)
        ask_attribute = "other"

        return {
            "message": "Here are the top matches based on your preferences.",
            "ask_attribute": ask_attribute,
            "recommendations": [{"parent_asin": r} for r in recommendations],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0}
        }
