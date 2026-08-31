# System repository classification

## A. Active demo

`shopping_agent/agent.py`, `catalogue.py`, `clarification.py`, `config.py`, `demo.py`, `demo_scenarios.json`, `embedding_backends.py`, `memory_store.py`, `vector_memory.py`, `visualizer/server.py`, `visualizer/simulator.py`, both dashboard HTML pages and assets, and the validated catalogue embedding caches in `shopping_agent/embedding_cache/`.

The 50,000-row catalogue remains at `techjam-conversational-search/data/catalog.jsonl`; it is read-only competition data, not active system code.

## B. Active tests

All modules under `shopping_agent/tests/`, including agent ranking/state, catalogue and clarification, browser lifecycle/server, dense-vector, demo smoke, stability regression, and vector-memory coverage.

## C. Active documentation

`README.md`, `MEMORY_ARCHITECTURE.md`, `MEMORY_EVALUATION_STATUS.md`, and this classification.

## D. Research / evaluation archive

`docs/archive/research_evaluation/`: longitudinal evaluators and frozen v2 evidence, threshold and blend studies, M0/embedding bakeoff material, Colab bundles, retrieval experiments 1–11, result manifests, logs, hashes, and archived evaluator tests.

## E. Legacy / disconnected archive

`docs/archive/legacy_qlmp/`: QLMP library, projector and portability work, old adapters, fixtures, diagnostics, results, and tests. No active file imports this tree.

## F. Safe to delete

Only generated `__pycache__/`, `.pytest_cache/`, and local `.demo_state/` files are disposable. No research evidence was classified for deletion.

## Minimal active runtime dependency chain

```text
python -m system.shopping_agent.demo
  -> demo.DemoApplication
  -> agent.Agent
     -> short-term state + parser/generator
     -> catalogue.Catalogue + clarification.select_best_attributes
     -> vector_memory.score_catalog
     -> memory_store.JsonFileVectorMemoryStore
     -> embedding_backends.OpenAIEmbeddingBackend
     -> embedding_cache/catalog_cache_openai-text-embedding-3-large.npz
     -> techjam-conversational-search/data/catalog.jsonl
     -> OpenAI API (query embeddings and response generation only)
  -> visualizer.server.BrowserApplication + ThreadingHTTPServer
```

Evaluation, retrieval-experiment, projector, and QLMP modules are absent from this chain.
