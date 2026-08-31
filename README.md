# TechJam 2026 shopping system

> Submission status: the active demo is tested, but the repository is not yet a
> frozen submission bundle. The official evaluator imports `starter.agent.Agent`,
> while the active implementation is `system.shopping_agent.Agent`. Follow
> [`SUBMISSION_GUIDE.md`](SUBMISSION_GUIDE.md) before submitting a commit.

The canonical presentation is the browser dashboard backed by the active `system` package:

```powershell
python -m system.shopping_agent.visualizer.server
```

Open <http://localhost:8080> for the ASTRA product catalog, or <http://localhost:8080/conversation> for the conversational simulator. In the simulator, the public `sample_id` is the persistent `user_id`: rerunning a sample loads its prior vector, while different samples remain isolated.

## Active architecture

```text
catalog landing page or conversation dashboard -> live Buying/Browsing state
          -> broad evaluator-template local parser OR selected-model complete-state editor
          -> intent-aware FTS5 AND/weighted-OR routing -> hard eligibility
          -> selected embedding-space v1/v2 gate and full 50,000-row s1/s2/s3 scoring
          -> keyword pool -> handcrafted state ranking; or 150-row vector fallback -> s3 ranking
          -> confidence gate (top-10 product s1 >= 0.40)
          -> surviving recommendations / entropy-based querying
          -> selected model response/cards -> end-session adaptive-centroid memory commit
```

FTS5 controls candidate routing and supplies the stable retrieval order used by successful keyword ranking. Keyword candidates use the root handcrafted state score; long-term memory does not rerank them. `s3` remains authoritative for vector fallback, with ASIN tie-breaking. Previously returned products are removed before the fixed top-10 confidence pool is formed, and surviving rows receive a rank-preserving brand/title diversity pass. Maximum price, demographic, rating, review-count, brand/store, and negative filters are session-local and are never silently relaxed. Query, catalogue, and longitudinal-memory vectors always come from the same selected, normalized embedding space.

Copy `system/shopping_agent/.env.example` to `.env` when overrides are needed:

```dotenv
TEST_MODE=false

OPENAI_API_KEY=
OPENAI_CHAT_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_EMBEDDING_DIMENSIONS=1536
OPENAI_TIMEOUT_SECONDS=30

OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=llama3.1:8b
OLLAMA_TIMEOUT_SECONDS=30
ALLOW_CATALOG_EMBEDDING=false
CONFIDENCE_SIMILARITY_THRESHOLD=0.40
```

`TEST_MODE=false` (or unset) selects local BGE embeddings and one shared Ollama client. `TEST_MODE=true` requires `OPENAI_API_KEY` and selects `text-embedding-3-small` plus one shared OpenAI Responses API client for parsing, assistant text, and shopper simulation. A failed or invalid response is retried once; exhausted failures are typed and the affected agent turn rolls back. There is no provider fallback, and changing `.env` requires a process restart.

Production deployment must provision and validate
`system/shopping_agent/embedding_cache/catalog_cache_bge-base-en-v1.5.npz`; the file is
an external asset and is not stored in Git. OpenAI uses a separate model-and-dimension cache and separate longitudinal-memory file, so its vectors never overwrite or reuse BGE artifacts. Build or validate the billable 50,000-row OpenAI cache explicitly with:

```powershell
python -m system.shopping_agent.build_embedding_cache
```

The command refuses to run unless `TEST_MODE=true`. Normal startup never builds a missing OpenAI cache.

Live intent is re-evaluated each turn. Buying uses Yangxu's `15/10` lexical thresholds; Browsing uses `30/15`. The existing `buyer_mode` argument remains a failure fallback, while live intent selects the frozen Buying/Browsing weights and clarification priority. After route-specific ranking, the fixed top-10 pool is filtered inclusively at current-query `s1 >= 0.40`; lower ranks never backfill it. The threshold can be changed at process start with `CONFIDENCE_SIMILARITY_THRESHOLD`.

## Main files to use

Work in `system/shopping_agent/` for the active application:

| File | Use it for |
| --- | --- |
| `visualizer/server.py` | Canonical browser entry point, HTTP routes, and browser-session lifecycle |
| `agent.py` | Main shopping-agent API and turn orchestration |
| `model_client.py` / `runtime.py` | Provider-neutral contracts and process-start provider selection |
| `ollama_client.py` / `openai_client.py` | Provider transports, retry policy, typed failures, and telemetry |
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

The runtime reads `techjam-conversational-search/data/catalog.jsonl` and `public_set.jsonl` as competition data; do not add active application logic there. `techjam-conversational-search/` contains the official-style evaluator plus an older reference agent; it is not currently wired to the active `system` agent. `techjam-conversational-search-participant-kit/` is the untouched starter kit, and `archive/` contains historical research and legacy implementations. Winston's original probes, benchmarks, parser/pipeline snapshots, proof adapters, and experimental branches are preserved at `archive/winston/`.

The supporting CLI is:

```powershell
python -m system.shopping_agent.demo --user alice
python -m system.shopping_agent.demo --scripted
python -m system.shopping_agent.demo --inspect alice
python -m system.shopping_agent.demo --reset-user alice
python -m system.shopping_agent.demo --reset-all
```

For local mode, install Ollama, run `ollama pull llama3.1:8b`, and provision the BGE cache with `colab/bge_pipeline.ipynb`. For OpenAI mode, set the API key, restart, run the explicit cache command above, and then use the same CLI and browser commands.

Run active tests with:

```powershell
python -m pytest system/shopping_agent/tests -q
```

See [`system/README.md`](system/README.md) for lifecycle and trace details. Historical Yangxu code is preserved intact under [`archive/legacy_hybrid_agent/`](archive/legacy_hybrid_agent/); frozen memory/retrieval evidence remains under `archive/research_evaluation/` and `archive/legacy_qlmp/`.
