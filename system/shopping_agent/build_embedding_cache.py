"""Explicitly build or validate the billable OpenAI catalogue embedding cache."""

from __future__ import annotations

from .agent import Agent
from .config import CATALOG_PATH, EXPECTED_CATALOG_ROWS, RUNTIME_CONFIG
from .runtime import get_runtime_providers


def main() -> None:
    if not RUNTIME_CONFIG.test_mode:
        raise SystemExit("This command requires TEST_MODE=true; refusing a non-OpenAI cache build.")
    with CATALOG_PATH.open("rb") as handle:
        rows = sum(1 for line in handle if line.strip())
    if rows != EXPECTED_CATALOG_ROWS:
        raise SystemExit(f"Expected {EXPECTED_CATALOG_ROWS:,} catalogue rows, found {rows:,}.")
    providers = get_runtime_providers()
    print(f"[Cache Builder] OpenAI {providers.embedding_backend.model_id} "
          f"d{providers.embedding_backend.vector_dimension}; catalogue generation is billable.")
    agent = Agent(embedding_backend=providers.embedding_backend,
                  llm_client=providers.llm_client, allow_catalog_embedding=True,
                  explicit_cache_build=True)
    try:
        print(f"[Cache Builder] Valid cache ready: {agent.embedding_cache_path} "
              f"({len(agent.catalog_embeddings):,} rows)")
    finally:
        agent.close()


if __name__ == "__main__": main()
