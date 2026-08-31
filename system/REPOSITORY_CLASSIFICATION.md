# System repository classification

## A. Active demo

`shopping_agent/agent.py`, `ollama_client.py`, `catalogue.py`, `clarification.py`, `config.py`, `demo.py`, `demo_scenarios.json`, `embedding_backends.py`, `memory_store.py`, `vector_memory.py`, `visualizer/server.py`, `visualizer/simulator.py`, both dashboard HTML pages and assets, and the validated BGE catalogue cache in `shopping_agent/embedding_cache/`. The reproducible artifact workflow is under top-level `colab/`.

The 50,000-row catalogue remains at `techjam-conversational-search/data/catalog.jsonl`; it is read-only competition data, not active system code.

## B. Active tests

All modules under `shopping_agent/tests/`, including agent ranking/state, catalogue and clarification, browser lifecycle/server, dense-vector, demo smoke, stability regression, and vector-memory coverage.

## C. Active documentation

`README.md`, `MEMORY_ARCHITECTURE.md`, `MEMORY_EVALUATION_STATUS.md`, and this classification.

## D. Research / evaluation archive

`archive/research_evaluation/`: longitudinal evaluators and frozen v2 evidence, threshold and blend studies, M0/embedding bakeoff material, Colab bundles, retrieval experiments 1–11, result manifests, logs, hashes, and archived evaluator tests.

## E. Legacy / disconnected archive

`archive/legacy_qlmp/`: QLMP library, projector and portability work, old adapters, fixtures, diagnostics, results, and tests. No active file imports this tree.

`archive/winston/`: original constrained-parser and resolver snapshots, probes,
gold/prediction files, messy-input benchmarks, reports, proof adapters, and experimental
dense/fusion/fine-tuning/LoRA branches. The active runtime imports none of this tree.

## F. Safe to delete

Only generated `__pycache__/`, `.pytest_cache/`, and local `.demo_state/` files are disposable. No research evidence was classified for deletion.

## Minimal active runtime dependency chain

```text
python -m system.shopping_agent.demo
  -> demo.DemoApplication
  -> agent.Agent
     -> short-term state editor + response generator
     -> ollama_client.OllamaClient -> local llama3.1:8b for state editing, assistant, shopper
     -> catalogue.Catalogue + clarification.select_best_attributes
     -> vector_memory.score_catalog
     -> memory_store.JsonFileVectorMemoryStore
     -> embedding_backends.BGEEmbeddingBackend
     -> embedding_cache/catalog_cache_bge-base-en-v1.5.npz
     -> techjam-conversational-search/data/catalog.jsonl
  -> visualizer.server.BrowserApplication + ThreadingHTTPServer
```

Evaluation, retrieval-experiment, projector, and QLMP modules are absent from this chain.
