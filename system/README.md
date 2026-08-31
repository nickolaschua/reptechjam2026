# System shopping-agent demo

`system.shopping_agent` is the active TechJam runtime. Its canonical browser demo combines Yangxu's ASTRA catalog and conversation dashboard, Winston's constrained free-text parser/category ambiguity gate, and the frozen gated-vector scorer.

## Run

From the repository root:

```powershell
python -m system.shopping_agent.visualizer.server
```

The catalog is served at `/`, and the conversation dashboard is at `/conversation`. `/catalog_search` searches and paginates all 50,000 products, including Watches. The existing simulation routes and payloads remain available: `/sessions_list`, `/manual_start?sample_id=...`, `/manual_step?sample_id=...&message=...`, and `/stream?sample_id=...`.

The dashboard `sample_id` is also the persistent `user_id`. Starting the same sample again advances its longitudinal sequence and loads prior LTM; another sample gets independent memory. Manual sessions commit on target hit, turn 10, replacement, or clean shutdown. Streamed sessions commit on success or turn 10; provider failures and client disconnects discard active state without advancing chronology.

The CLI remains available:

```powershell
python -m system.shopping_agent.demo --user alice
python -m system.shopping_agent.demo --scripted
python -m system.shopping_agent.demo --inspect alice
python -m system.shopping_agent.demo --reset-user alice
python -m system.shopping_agent.demo --reset-all
```

## Turn parsing, retrieval, and memory

Exact evaluator messages use the frozen local template parser. All other free-text
turns use an Ollama schema-constrained Llama 3.1 parser (`llama3.1:8b` by default),
one retry, and deterministic post-validation. Parser or resolver failure is terminal
for the turn and restores its entry snapshot; there is no free-form state-tracking or
regex fallback on this route.

The completed lexical resolver ranks category buckets from the parsed category phrase
plus soft slots. It is built once from the live `Catalogue` and stores its top three
candidates and relative top-two margin in debug telemetry. A category-establishing
turn below `0.20`, without another trusted hard condition, returns no cards and asks
an open category question before any embedding or response-generation call. A usable
answer to that pending question resumes retrieval even when its margin remains flat.

The runtime loads the catalogue once, builds an in-memory FTS5 table over Yangxu's exact fields/tokenizer, and loads the compatible fixed BGE matrix. Live intent is re-evaluated every turn: Buying widens below 15 AND hits and accepts the keyword route at 10 eligible rows, while Browsing uses 30 and 15. Both keyword and 150-row fallback pools are ordered only by `s3`, with ASIN tie-breaking.

Live intent is authoritative for routing, clarification priorities, and selection of the existing frozen memory weights. The public `buyer_mode` argument remains compatible but is used only when live detection cannot resolve a mode. Buying-to-Browsing transitions require an explicit reset.

Every request embeds `v1` with the BGE search prefix, loads the BGE-space reset-time `v2`, evaluates the frozen relevance gate and Buying/Browsing weights, and computes `s1`, `s2`, and `s3` across all 50,000 rows. Popularity mixing, heuristic boosts, diversification, online embedding regeneration, and fine-tuning are not active.

Hard filters are session-only. Minimum and maximum price, demographic department,
ratings, review counts, and requested brands do not enter the EWMA update text.
Material and size remain positive ranking evidence, not unsafe catalogue exclusions.
Unknown rating and review metadata receives Yangxu's benefit of doubt. If no row is
eligible, the agent returns no recommendations and asks which hard constraint to relax.

The catalogue, agent, CLI, and browser server own explicit idempotent shutdown paths. Failed turns restore their entry state, turn counters advance only after a successful response, JSON persistence rolls back in-memory mutations on failure, and completed debug traces are bounded and consumed by the demo adapters.

## Environment and state

The constrained parser, assistant response generator, and browser shopper simulator
share `OLLAMA_HOST` (default `http://localhost:11434`), `OLLAMA_MODEL` (default
`llama3.1:8b`), and `OLLAMA_TIMEOUT_SECONDS` (default `30`). The client retries once
for transport or invalid-response failures and records actual model, latency, retries,
error type, and rollback status. It never routes to a hosted provider or static answer.
A local `.env` may be stored at `system/shopping_agent/.env`.

Persistent demo state defaults to `system/shopping_agent/.demo_state/vector_memory.json`. Normal responses contain no raw vectors; `debug=True` exposes vector-free route, gate, eligibility, constraint, entropy, score, and commit metadata.

The production BGE end-to-end run additionally requires the deployment asset
`system/shopping_agent/embedding_cache/catalog_cache_bge-base-en-v1.5.npz`.
Build and verify it with `colab/bge_pipeline.ipynb` and
`colab/verify_bge_artifact.py`. Loading validates its recorded backend/model/space IDs, row ordering, catalogue and
product-text fingerprints, vector dimension, and normalization metadata. Its absence
is an external provisioning failure; the runtime does not silently regenerate it.

## Tests and archives

```powershell
python -m pytest system/shopping_agent/tests -q
```

Yangxu's original complete sandbox, evaluator scripts, reports, visualizer, and image map are archived at `archive/legacy_hybrid_agent/`. The handoff record is `archive/MAIN_PULL_CHANGES_2026-08-30.md`. Frozen gated-memory evaluation and old QLMP work remain in their respective archive directories.

Winston's original parser/pipeline source, probes, benchmark evidence, proof adapters,
and rejected dense/fusion/fine-tuning branches are archived at `archive/winston/`.

The architecture slide's “persistent hard criteria” and “continual fine-tuning” blocks are intentionally not implemented: hard criteria are session-local, and the stock BGE catalogue matrix is a fixed validated cache. The optional Colab-tuned model is exported separately and is never activated automatically.
