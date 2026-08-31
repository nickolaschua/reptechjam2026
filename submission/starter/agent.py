"""Frozen BGE/Ollama entry point for the official TechJam evaluator."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


# Provider selection happens while importing the bundled runtime. Force the
# submitted configuration before that import so a developer shell cannot switch
# this release to the OpenAI branch accidentally.
os.environ["TEST_MODE"] = "false"
os.environ["ALLOW_CATALOG_EMBEDDING"] = "false"

from system.shopping_agent.agent import Agent as _CoreAgent  # noqa: E402


BUNDLE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CATALOG_PATH = BUNDLE_ROOT / "data" / "catalog.jsonl"
DEFAULT_CACHE_DIR = BUNDLE_ROOT / "artifacts"


def _path_from_env(name: str, default: Path) -> Path:
    value = os.environ.get(name, "").strip()
    return Path(value).expanduser() if value else default


class Agent(_CoreAgent):
    """Official contract adapter locked to local BGE embeddings and Ollama."""

    def __init__(self, catalog_path: str | Path | None = None, **kwargs: Any) -> None:
        forbidden = {
            "embedding_backend", "llm_client", "ollama_client", "test_mode",
            "allow_catalog_embedding", "explicit_cache_build",
        } & kwargs.keys()
        if forbidden:
            names = ", ".join(sorted(forbidden))
            raise TypeError(f"submission provider configuration is frozen; remove: {names}")

        resolved_catalog = Path(catalog_path) if catalog_path is not None else _path_from_env(
            "TECHJAM_CATALOG_PATH", DEFAULT_CATALOG_PATH
        )
        cache_dir = kwargs.pop(
            "embedding_cache_dir",
            _path_from_env("TECHJAM_BGE_CACHE_DIR", DEFAULT_CACHE_DIR),
        )
        super().__init__(
            catalog_path=resolved_catalog,
            embedding_cache_dir=cache_dir,
            allow_catalog_embedding=False,
            explicit_cache_build=False,
            **kwargs,
        )


__all__ = ["Agent"]
