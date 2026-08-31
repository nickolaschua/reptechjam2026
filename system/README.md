# System shopping-agent demo

`system.shopping_agent` is the active TechJam runtime. Its canonical browser demo combines the ASTRA catalog and conversation dashboard, the pre-Winston state-editor preprocessing flow, and the frozen gated-vector scorer.

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

Evaluator-template messages use the broad local regex parser. Other free-text turns
send the complete prior state and current message to a selected-model state editor.
The returned JSON is merged with deterministic handling for intent, overrides,
category changes, demographics, negatives, and no-preference responses. Malformed
JSON restores the prior state before the local regex fallback is applied.

The runtime loads the catalogue once, builds an in-memory FTS5 table over Yangxu's exact fields/tokenizer, and loads the compatible fixed BGE matrix. Live intent is re-evaluated every turn: Buying widens below 15 AND hits and accepts the keyword route at 10 eligible rows, while Browsing uses 30 and 15. Successful keyword pools use the root agent's stable handcrafted state score, while the 150-row fallback remains ordered by descending `s3` with ASIN tie-breaking. Previously returned products are removed before the post-ranking confidence gate evaluates ranks 1-10 of the unseen list. The gate retains products with inclusive current-query cosine `s1 >= 0.40`, never backfills from lower ranks, and passes survivors through rank-preserving brand/title diversity before applying `top_k`.

```text
Recommendation Ranking
        ↓
Confidence Gate: product s1 ≥ 0.40
        ↓
Surviving Recommendations / Entropy-Based Querying
```

When no product clears this gate, the response model receives no product context and is explicitly required to ask the entropy-selected clarification question; entropy is calculated from the rejected pre-gate top-10 pool. `CONFIDENCE_SIMILARITY_THRESHOLD` is a validated process-start override.

Live intent is authoritative for routing, clarification priorities, and selection of the existing frozen memory weights. The public `buyer_mode` argument remains compatible but is used only when live detection cannot resolve a mode. Buying-to-Browsing transitions require an explicit reset.

Every request embeds `v1` with the BGE search prefix, loads the BGE-space reset-time `v2`, evaluates the frozen relevance gate and Buying/Browsing weights, and computes `s1`, `s2`, and `s3` across all 50,000 rows. Those scores remain authoritative for fallback, forensic traces, and confidence filtering. Keyword ranking alone adds the root formula's department/category, brand, accumulated-term, constraint-phrase, category-phrase, FTS-position, and review-count evidence; online embedding regeneration and fine-tuning are not active.

Hard filters are session-only. Maximum price, demographic department,
ratings, review counts, and requested brands do not enter the adaptive update text.
Material and size remain positive ranking evidence, not unsafe catalogue exclusions.
Unknown rating and review metadata receives Yangxu's benefit of doubt. If no row is
eligible, the agent returns no recommendations and asks which hard constraint to relax.

The catalogue, agent, CLI, and browser server own explicit idempotent shutdown paths. Failed turns restore their entry state, turn counters advance only after a successful response, JSON persistence rolls back in-memory mutations on failure, and completed debug traces are bounded and consumed by the demo adapters.

## Environment and state

The state editor, assistant response generator, and browser shopper simulator
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
