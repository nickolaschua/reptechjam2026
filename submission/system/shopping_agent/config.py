"""Single source of truth for the active TechJam demo configuration."""

from __future__ import annotations

import os
import math
from dataclasses import dataclass
from pathlib import Path


SHOPPING_AGENT_DIR = Path(__file__).resolve().parent
SYSTEM_DIR = SHOPPING_AGENT_DIR.parent
PROJECT_ROOT = SYSTEM_DIR.parent

# Bundle layout: PROJECT_ROOT is the submission/ root, not the source repo, so
# these defaults must match starter/agent.py rather than the main-repo tree.
CATALOG_PATH = PROJECT_ROOT / "data" / "catalog.jsonl"
EMBEDDING_CACHE_DIR = PROJECT_ROOT / "artifacts"
MEMORY_STORE_PATH = PROJECT_ROOT / ".demo_state" / "vector_memory.json"

DEFAULT_OPENAI_CHAT_MODEL = "gpt-4o-mini"
DEFAULT_OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_OPENAI_EMBEDDING_DIMENSIONS = 1536


def _load_env_file() -> None:
    """Load the nearest project .env without overriding process environment."""
    env_path = SHOPPING_AGENT_DIR / ".env"
    if not env_path.exists():
        env_path = SYSTEM_DIR / ".env"
    if not env_path.exists():
        env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(
        f"{name} must be one of true/false, 1/0, yes/no, or on/off; got {value!r}"
    )


_load_env_file()

@dataclass(frozen=True)
class RuntimeConfig:
    test_mode: bool
    provider: str
    openai_api_key: str | None
    openai_chat_model: str
    openai_embedding_model: str
    openai_embedding_dimensions: int
    openai_timeout_seconds: float
    ollama_host: str
    ollama_model: str
    ollama_timeout_seconds: float
    allow_catalog_embedding: bool
    confidence_similarity_threshold: float = 0.40


def _positive_float(name: str, default: str) -> float:
    try: value = float(os.environ.get(name, default))
    except ValueError as exc: raise ValueError(f"{name} must be numeric") from exc
    if value <= 0: raise ValueError(f"{name} must be greater than zero")
    return value


def _positive_int(name: str, default: int) -> int:
    raw = os.environ.get(name, str(default))
    try: value = int(raw)
    except ValueError as exc: raise ValueError(f"{name} must be an integer") from exc
    if value <= 0: raise ValueError(f"{name} must be greater than zero")
    return value


def _cosine_threshold(name: str, default: str) -> float:
    try:
        value = float(os.environ.get(name, default))
    except ValueError as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(value) or not -1.0 <= value <= 1.0:
        raise ValueError(f"{name} must be a finite cosine value between -1 and 1")
    return value


def load_runtime_config() -> RuntimeConfig:
    """Read and validate the process-start provider configuration."""
    test_mode = _env_bool("TEST_MODE", False)
    key = os.environ.get("OPENAI_API_KEY", "").strip() or None if test_mode else None
    chat_model = os.environ.get("OPENAI_CHAT_MODEL", DEFAULT_OPENAI_CHAT_MODEL).strip()
    embedding_model = os.environ.get("OPENAI_EMBEDDING_MODEL", DEFAULT_OPENAI_EMBEDDING_MODEL).strip()
    if test_mode:
        if key is None: raise ValueError("OPENAI_API_KEY is required when TEST_MODE=true")
        if not chat_model: raise ValueError("OPENAI_CHAT_MODEL must be non-empty")
        if not embedding_model: raise ValueError("OPENAI_EMBEDDING_MODEL must be non-empty")
        openai_dimensions = _positive_int("OPENAI_EMBEDDING_DIMENSIONS", DEFAULT_OPENAI_EMBEDDING_DIMENSIONS)
        if openai_dimensions != DEFAULT_OPENAI_EMBEDDING_DIMENSIONS:
            raise ValueError("OPENAI_EMBEDDING_DIMENSIONS must be 1536")
        openai_timeout = _positive_float("OPENAI_TIMEOUT_SECONDS", "30")
        ollama_timeout = 30.0
    else:
        openai_dimensions = DEFAULT_OPENAI_EMBEDDING_DIMENSIONS
        openai_timeout = 30.0
        ollama_timeout = _positive_float("OLLAMA_TIMEOUT_SECONDS", "30")
    return RuntimeConfig(
        test_mode=test_mode, provider="openai" if test_mode else "ollama",
        openai_api_key=key, openai_chat_model=chat_model,
        openai_embedding_model=embedding_model,
        openai_embedding_dimensions=openai_dimensions,
        openai_timeout_seconds=openai_timeout,
        ollama_host=os.environ.get("OLLAMA_HOST", "http://localhost:11434").strip(),
        ollama_model=os.environ.get("OLLAMA_MODEL", "llama3.1:8b").strip(),
        ollama_timeout_seconds=ollama_timeout,
        allow_catalog_embedding=_env_bool("ALLOW_CATALOG_EMBEDDING", False),
        confidence_similarity_threshold=_cosine_threshold(
            "CONFIDENCE_SIMILARITY_THRESHOLD", "0.40"
        ),
    )


RUNTIME_CONFIG = load_runtime_config()
TEST_MODE = RUNTIME_CONFIG.test_mode
ALLOW_CATALOG_EMBEDDING = _env_bool("ALLOW_CATALOG_EMBEDDING", False)
CONFIDENCE_SIMILARITY_THRESHOLD = RUNTIME_CONFIG.confidence_similarity_threshold


def memory_store_path(config: RuntimeConfig = RUNTIME_CONFIG) -> Path:
    if not config.test_mode: return MEMORY_STORE_PATH
    safe_model = config.openai_embedding_model.replace("/", "_").replace(":", "_")
    return SHOPPING_AGENT_DIR / ".demo_state" / f"vector_memory_openai-{safe_model}-d{config.openai_embedding_dimensions}.json"


ACTIVE_MEMORY_STORE_PATH = memory_store_path()

RELEVANCE_THRESHOLD = 0.30
BUYING_CURRENT_WEIGHT = 0.80
BUYING_MEMORY_WEIGHT = 0.20
BROWSING_CURRENT_WEIGHT = 0.20
BROWSING_MEMORY_WEIGHT = 0.80
EWMA_ALPHA = 0.30
DEMO_TOP_K = 5
EXPECTED_CATALOG_ROWS = 50_000


__all__ = [
    "ALLOW_CATALOG_EMBEDDING",
    "CONFIDENCE_SIMILARITY_THRESHOLD",
    "ACTIVE_MEMORY_STORE_PATH",
    "BROWSING_CURRENT_WEIGHT",
    "BROWSING_MEMORY_WEIGHT",
    "BUYING_CURRENT_WEIGHT",
    "BUYING_MEMORY_WEIGHT",
    "CATALOG_PATH",
    "DEMO_TOP_K",
    "EMBEDDING_CACHE_DIR",
    "EWMA_ALPHA",
    "EXPECTED_CATALOG_ROWS",
    "MEMORY_STORE_PATH",
    "RUNTIME_CONFIG",
    "RuntimeConfig",
    "TEST_MODE",
    "load_runtime_config",
    "memory_store_path",
    "PROJECT_ROOT",
    "RELEVANCE_THRESHOLD",
    "SHOPPING_AGENT_DIR",
    "SYSTEM_DIR",
]
