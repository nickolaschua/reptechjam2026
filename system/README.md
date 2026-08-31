# System shopping-agent demo

`system.shopping_agent` is the active TechJam runtime. Its canonical browser demo combines Yangxu's ASTRA catalog and conversation dashboard with the frozen OpenAI gated-vector scorer.

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

## Retrieval and memory

The runtime loads the catalogue once, builds an in-memory FTS5 table over Yangxu's exact fields/tokenizer, and loads the compatible fixed OpenAI matrix. Live intent is re-evaluated every turn: Buying widens below 15 AND hits and accepts the keyword route at 10 eligible rows, while Browsing uses 30 and 15. Both keyword and 150-row fallback pools are ordered only by `s3`, with ASIN tie-breaking.

Live intent is authoritative for routing, clarification priorities, and selection of the existing frozen memory weights. The public `buyer_mode` argument remains compatible but is used only when live detection cannot resolve a mode. Buying-to-Browsing transitions require an explicit reset.

Every request still embeds `v1`, loads the reset-time `v2`, evaluates the frozen relevance gate and Buying/Browsing weights, and computes `s1`, `s2`, and `s3` across all 50,000 rows. No BGE model, popularity mixing, heuristic boost, diversification, embedding regeneration, or fine-tuning is active.

Hard filters are session-only. They do not enter the EWMA update text. Unknown rating and review metadata receives Yangxu's benefit of doubt. If no row is eligible, the agent returns no recommendations and asks which hard constraint to relax.

The catalogue, agent, CLI, and browser server own explicit idempotent shutdown paths. Failed turns restore their entry state, turn counters advance only after a successful response, JSON persistence rolls back in-memory mutations on failure, and completed debug traces are bounded and consumed by the demo adapters.

## Environment and state

`OPENAI_API_KEY` is required for query embeddings and may generate assistant/shopper text. `DEEPSEEK_API_KEY` is supported for chat; streamed shopper simulation falls back to local Ollama when neither chat provider key is configured. A local `.env` may be stored at `system/shopping_agent/.env`.

Persistent demo state defaults to `system/shopping_agent/.demo_state/vector_memory.json`. Normal responses contain no raw vectors; `debug=True` exposes vector-free route, gate, eligibility, constraint, entropy, score, and commit metadata.

## Tests and archives

```powershell
python -m pytest system/shopping_agent/tests -q
```

Yangxu's original complete sandbox, evaluator scripts, reports, visualizer, and image map are archived at `docs/archive/legacy_hybrid_agent/`. The handoff record is `docs/archive/MAIN_PULL_CHANGES_2026-08-30.md`. Frozen gated-memory evaluation and old QLMP work remain in their respective archive directories.

The architecture slide's “persistent hard criteria” and “continual fine-tuning” blocks are intentionally not implemented: hard criteria are session-local, and the OpenAI catalogue matrix is a fixed validated cache.
