# TechJam 2026 shopping system

The canonical presentation is Yangxu's browser dashboard backed by the active `system` package:

```powershell
python -m system.shopping_agent.visualizer.server
```

Open <http://localhost:8080> for Yangxu's ASTRA product catalog, or <http://localhost:8080/conversation> for the conversational simulator. In the simulator, the public `sample_id` is the persistent `user_id`: rerunning a sample loads its prior vector, while different samples remain isolated.

## Active architecture

```text
catalog landing page or conversation dashboard -> live Buying/Browsing state
          -> exact evaluator template parser OR constrained local Llama 3.1 free-text parser
          -> soft-slot category resolver -> low-confidence open category clarification
          -> intent-aware FTS5 AND/weighted-OR routing -> hard eligibility
          -> BGE v1/v2 gate and full 50,000-row s1/s2/s3 scoring
          -> keyword pool or 150-row vector fallback -> entropy clarification
          -> local Llama 3.1 response/cards -> end-session EWMA memory commit
```

FTS5 controls candidate routing only. `s3` is the authoritative rank score. Minimum/maximum price, demographic, rating, review-count, brand/store, and negative filters are session-local and are never silently relaxed. The category resolver controls only the pre-retrieval ambiguity question; its candidates never filter, boost, or rerank products. Query, catalogue, and longitudinal-memory vectors all use normalized 768-dimensional `BAAI/bge-base-en-v1.5` embeddings.

Copy `system/shopping_agent/.env.example` to `.env` when overrides are needed:

```dotenv
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=llama3.1:8b
OLLAMA_TIMEOUT_SECONDS=30
ALLOW_CATALOG_EMBEDDING=false
```

The constrained parser, assistant response generator, and streamed shopper simulator share this one Ollama client and model. A failed or invalid response is retried once; exhausted failures are typed and the affected agent turn rolls back.

Production deployment must provision and validate
`system/shopping_agent/embedding_cache/catalog_cache_bge-base-en-v1.5.npz`; the file is
an external asset and is not stored in Git.

Live intent is re-evaluated each turn. Buying uses Yangxu's `15/10` lexical thresholds; Browsing uses `30/15`. The existing `buyer_mode` argument remains a failure fallback, while live intent selects the frozen Buying/Browsing weights and clarification priority.

## Main files to use

Work in `system/shopping_agent/` for the active application:

| File | Use it for |
| --- | --- |
| `visualizer/server.py` | Canonical browser entry point, HTTP routes, and browser-session lifecycle |
| `agent.py` | Main shopping-agent API and turn orchestration |
| `ollama_client.py` | Shared local Llama 3.1 transport, retry policy, typed failures, and telemetry |
| `turn_parser.py` | Constrained Llama 3.1 parse, deterministic validation, and typed free-text turns |
| `category_resolver.py` | Soft-slot category candidates and ambiguity confidence |
| `catalogue.py` | Catalogue loading, FTS5 candidate routing, and hard eligibility filters |
| `clarification.py` | Entropy-based clarification selection |
| `vector_memory.py` | Frozen relevance gate and `s1`/`s2`/`s3` scoring equations |
| `memory_store.py` | Per-user longitudinal memory, commits, snapshots, and JSON persistence |
| `embedding_backends.py` | Query embeddings and validation/loading of the fixed catalogue embedding cache |
| `config.py` | Canonical data, cache, and model paths |
| `demo.py` | CLI entry point and demo application wiring |
| `visualizer/simulator.py` | Scripted and generated shopper behavior used by the browser demo |
| `tests/` | Active unit, integration, lifecycle, and regression tests |
| `colab/` | Reproducible production BGE cache build, optional tuning, and artifact verifier |

The runtime reads `techjam-conversational-search/data/catalog.jsonl` and `public_set.jsonl` as competition data; do not add active application logic there. `techjam-conversational-search/` is the submission-style reference implementation, `techjam-conversational-search-participant-kit/` is the untouched starter kit, and `archive/` contains historical research and legacy implementations. Winston's original probes, benchmarks, parser/pipeline snapshots, proof adapters, and experimental branches are preserved at `archive/winston/`.

The supporting CLI is:

```powershell
python -m system.shopping_agent.demo --user alice
python -m system.shopping_agent.demo --scripted
python -m system.shopping_agent.demo --inspect alice
python -m system.shopping_agent.demo --reset-user alice
python -m system.shopping_agent.demo --reset-all
```

Install Ollama, run `ollama pull llama3.1:8b`, and provision the BGE cache with `colab/bge_pipeline.ipynb`. No hosted-provider API key is read by the active runtime.

Run active tests with:

```powershell
python -m pytest system/shopping_agent/tests -q
```

See [`system/README.md`](system/README.md) for lifecycle and trace details. Historical Yangxu code is preserved intact under [`archive/legacy_hybrid_agent/`](archive/legacy_hybrid_agent/); frozen memory/retrieval evidence remains under `archive/research_evaluation/` and `archive/legacy_qlmp/`.
