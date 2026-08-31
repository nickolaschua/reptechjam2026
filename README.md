# TechJam 2026 shopping system

The canonical presentation is Yangxu's browser dashboard backed by the active `system` package:

```powershell
python -m system.shopping_agent.visualizer.server
```

Open <http://localhost:8080> for Yangxu's ASTRA product catalog, or <http://localhost:8080/conversation> for the conversational simulator. In the simulator, the public `sample_id` is the persistent `user_id`: rerunning a sample loads its prior vector, while different samples remain isolated.

## Active architecture

```text
catalog landing page or conversation dashboard -> live Buying/Browsing state
          -> intent-aware FTS5 AND/weighted-OR routing -> hard eligibility
          -> frozen OpenAI v1/v2 gate and full 50,000-row s1/s2/s3 scoring
          -> keyword pool or 150-row vector fallback -> entropy clarification
          -> response/cards -> end-session EWMA memory commit
```

FTS5 controls candidate routing only. `s3` is the authoritative rank score. Price, demographic, rating, review-count, brand/store, and negative filters are session-local and are never silently relaxed. The 50,000-product OpenAI matrix is a fixed validated cache; there is no continual fine-tuning in the active path.

Live intent is re-evaluated each turn. Buying uses Yangxu's `15/10` lexical thresholds; Browsing uses `30/15`. The existing `buyer_mode` argument remains a failure fallback, while live intent selects the frozen OpenAI Buying/Browsing weights and clarification priority.

## Main files to use

Work in `system/shopping_agent/` for the active application:

| File | Use it for |
| --- | --- |
| `visualizer/server.py` | Canonical browser entry point, HTTP routes, and browser-session lifecycle |
| `agent.py` | Main shopping-agent API and turn orchestration |
| `catalogue.py` | Catalogue loading, FTS5 candidate routing, and hard eligibility filters |
| `clarification.py` | Entropy-based clarification selection |
| `vector_memory.py` | Frozen relevance gate and `s1`/`s2`/`s3` scoring equations |
| `memory_store.py` | Per-user longitudinal memory, commits, snapshots, and JSON persistence |
| `embedding_backends.py` | Query embeddings and validation/loading of the fixed catalogue embedding cache |
| `config.py` | Canonical data, cache, and model paths |
| `demo.py` | CLI entry point and demo application wiring |
| `visualizer/simulator.py` | Scripted and generated shopper behavior used by the browser demo |
| `tests/` | Active unit, integration, lifecycle, and regression tests |

The runtime reads `techjam-conversational-search/data/catalog.jsonl` and `public_set.jsonl` as competition data; do not add active application logic there. `techjam-conversational-search/` is the submission-style reference implementation, `techjam-conversational-search-participant-kit/` is the untouched starter kit, and `docs/archive/` contains historical research and legacy implementations.

The supporting CLI is:

```powershell
python -m system.shopping_agent.demo --user alice
python -m system.shopping_agent.demo --scripted
python -m system.shopping_agent.demo --inspect alice
python -m system.shopping_agent.demo --reset-user alice
python -m system.shopping_agent.demo --reset-all
```

Set `OPENAI_API_KEY` for query embeddings and OpenAI response/shopper generation. `DEEPSEEK_API_KEY` may be used for chat generation; without either shopper key, streamed mode uses local Ollama. Keys may also be placed in `system/shopping_agent/.env`.

Run active tests with:

```powershell
python -m pytest system/shopping_agent/tests -q
```

See [`system/README.md`](system/README.md) for lifecycle and trace details. Historical Yangxu code is preserved intact under [`docs/archive/legacy_hybrid_agent/`](docs/archive/legacy_hybrid_agent/); frozen memory/retrieval evidence remains under `docs/archive/research_evaluation/` and `docs/archive/legacy_qlmp/`.
