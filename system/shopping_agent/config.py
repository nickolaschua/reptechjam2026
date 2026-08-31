"""Single source of truth for the active TechJam demo configuration."""

from __future__ import annotations

import os
from pathlib import Path


SHOPPING_AGENT_DIR = Path(__file__).resolve().parent
SYSTEM_DIR = SHOPPING_AGENT_DIR.parent
PROJECT_ROOT = SYSTEM_DIR.parent

CATALOG_PATH = PROJECT_ROOT / "techjam-conversational-search" / "data" / "catalog.jsonl"
EMBEDDING_CACHE_DIR = SHOPPING_AGENT_DIR / "embedding_cache"
MEMORY_STORE_PATH = SHOPPING_AGENT_DIR / ".demo_state" / "vector_memory.json"


def _load_env_file() -> None:
    """Load the nearest project .env without overriding process environment."""
    env_path = SHOPPING_AGENT_DIR / ".env"
    if not env_path.exists():
        env_path = SYSTEM_DIR / ".env"
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

OPENAI_EMBEDDING_MODEL = "text-embedding-3-large"
OPENAI_EMBEDDING_DIMENSIONS = 3072
OPENAI_SMALL_EMBEDDING_MODEL = "text-embedding-3-small"
OPENAI_SMALL_EMBEDDING_DIMENSIONS = 1536
# Main workflow switch: production uses local BGE; test mode uses OpenAI small.
TEST_MODE = _env_bool("TEST_MODE", False)
ALLOW_CATALOG_EMBEDDING = _env_bool("ALLOW_CATALOG_EMBEDDING", False)
OPENAI_CHAT_MODEL = "gpt-4o-mini"
DEEPSEEK_CHAT_MODEL = "deepseek-chat"

RELEVANCE_THRESHOLD = 0.20
BUYING_CURRENT_WEIGHT = 0.80
BUYING_MEMORY_WEIGHT = 0.20
BROWSING_CURRENT_WEIGHT = 0.20
BROWSING_MEMORY_WEIGHT = 0.80
EWMA_ALPHA = 0.30
DEMO_TOP_K = 5
EXPECTED_CATALOG_ROWS = 50_000


__all__ = [
    "ALLOW_CATALOG_EMBEDDING",
    "BROWSING_CURRENT_WEIGHT",
    "BROWSING_MEMORY_WEIGHT",
    "BUYING_CURRENT_WEIGHT",
    "BUYING_MEMORY_WEIGHT",
    "CATALOG_PATH",
    "DEMO_TOP_K",
    "DEEPSEEK_CHAT_MODEL",
    "EMBEDDING_CACHE_DIR",
    "EWMA_ALPHA",
    "EXPECTED_CATALOG_ROWS",
    "MEMORY_STORE_PATH",
    "OPENAI_CHAT_MODEL",
    "OPENAI_EMBEDDING_DIMENSIONS",
    "OPENAI_EMBEDDING_MODEL",
    "OPENAI_SMALL_EMBEDDING_DIMENSIONS",
    "OPENAI_SMALL_EMBEDDING_MODEL",
    "PROJECT_ROOT",
    "RELEVANCE_THRESHOLD",
    "SHOPPING_AGENT_DIR",
    "SYSTEM_DIR",
    "TEST_MODE",
]
