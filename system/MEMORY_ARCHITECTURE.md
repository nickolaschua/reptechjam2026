# Gated vector-memory architecture

Updated: 2026-08-31

The live `system/shopping_agent` path keeps the existing per-session short-term state and stores at most one normalized long-term vector per explicit user.

## Lifecycle

`Agent.reset(session_id, profile)` remains valid for anonymous sessions. Longitudinal sessions additionally pass `user_id` and a monotonic `sequence_index`. Reset freezes the user vector visible at that instant, rejects overlapping sessions for one user, and validates embedding-space identity.

Every turn calls `respond(session_id, user_message, turn, top_k, buyer_mode="buying")`. `buyer_mode` accepts only `buying` or `browsing`, but remains a caller fallback. Live message intent and active hard conditions are authoritative for routing and clarification; concrete buying evidence wins over exploratory wording, while genuine resets such as `start over` return to Browsing.

Evaluator-template forms update the same canonical state through the broad local parser.
Other turns send the complete prior state and current message to the selected shared
model client and merge the returned complete JSON state. Malformed JSON restores the
prior state and uses local parsing. Provider exceptions propagate and restore history,
state, search epoch, and
forensic snapshots to the turn-entry snapshot.

`end_session` runs only after scoring. It reconciles negatives again, then serializes sorted positive `disclosed_slots` as `attribute: value` fragments. Category, department, budget, negatives, transcript, outcomes, purchases, target data, asked attributes, and retrieval bookkeeping never enter the update text. Embedding or store failure leaves the session active for retry. Failed response turns restore Fast Memory to its entry snapshot. Failed streams call `discard_session`; successful debug traces are bounded to 32 and may be consumed immediately.

## Ranking

The canonical active query contains current category, department, and positive disclosed-slot values. `BAAI/bge-base-en-v1.5` embeds it with the BGE retrieval prefix into a normalized 768-dimensional `v1`; the reset-time BGE-space user vector is `v2`.

```text
gate = cosine(v1, v2)
threshold = 0.30

cold start or gate failure: s3 = s1
buying:                    s3 = 0.8*s1 + 0.2*s2
browsing:                  s3 = 0.2*s1 + 0.8*s2
```

Here `s1 = catalog @ v1` and `s2 = catalog @ v2`. The normalized full catalog matrix is scored on every turn. FTS5/BM25 routes the candidate pool using the frozen Buying thresholds 15/10 and Browsing thresholds 30/15. A successful keyword pool is ordered by the root agent's stable handcrafted state score: `-0.001` per FTS position, department/category boosts of `20/15`, a `-10` requested-brand mismatch penalty, `0.3` per accumulated-term match, `10` per exact constraint phrase (or `5` when all meaningful words match), a `2` exact category-phrase boost, and `0.02 * rating_number^0.1`. Long-term memory never reranks this route. The 150-row vector fallback remains ordered by descending `s3`, then ascending ASIN.

Hard masks apply maximum price, demographic department, minimum rating, minimum review count, requested brand/store, and whole-token negative exclusions before route ranking. Material and size are positive evidence rather than hard exclusions. Unknown rating/review metadata receives benefit of doubt. Previously seen products are removed after ranking; the fixed top 10 unseen rows are filtered inclusively at `s1 >= 0.40` with no backfill, then receive rank-preserving brand/title diversity before `top_k`. Constraints are never relaxed to fill `top_k`.

## Update and storage

Cold start stores the normalized new-preference vector directly. Later positive evidence uses the default adaptive centroid policy: `c = clip(v2·new_preferences, 0, 1)`, `alpha = 0.30(1-c)`, and `v3 = normalize((1-alpha)*v2 + alpha*new_preferences)`. Repetitive evidence therefore moves the centroid less, while orthogonal or negatively aligned evidence receives the `0.30` cap. A fixed `alpha=0.30` policy remains available as an experimental control. An effectively zero mixture preserves `v2` and reports a numerical fallback. An empty positive-slot set advances chronology without changing the vector or update count; exactly redundant positive evidence increments the evidence count while leaving the vector geometrically unchanged.

This remains a single centroid per user: adaptive updating slows directional drift from repetitive sessions, but it does not retain distinct interests the way a prototype bank would. `MemoryUpdatePolicy` is the update-policy boundary for that future schema change; no prototype bank or snapshot migration is introduced here.

`InMemoryVectorMemoryStore` defines the user-partitioned chronology and immutable state containing user ID, vector, embedding-space ID, last committed sequence, and update count. Active sessions may be cancelled without a commit. The demo wraps those exact semantics with `JsonFileVectorMemoryStore`, which transactionally persists unchanged snapshot schema version 2 after commits and resets and restores all in-memory maps if persistence fails. Former QLMP version-1 payloads are explicitly rejected and are not migrated.

## Trace

Each response exposes a vector-free `debug.memory_trace`: mode, long-term-memory gate cosine/pass, threshold, applied `a`/`b`, vector availability, embedding-space checks, full row count, filter counts, active `ranking_method`, returned `s1`/`s2`/`s3`, keyword-state scores for returned keyword candidates, final ASINs, and the post-ranking `confidence_gate` evaluation of the fixed top-10 pool. The end-session trace additionally reports update mode, raw and bounded update similarity, effective alpha, and a numerical fallback reason. LLM instrumentation records the actual selected model, latency, retry count, error type, and rollback status. Raw vectors are never logged.

The disconnected QLMP implementation and experiments are retained only in `archive/legacy_qlmp/`. Current evaluator and calibration evidence is retained in `archive/research_evaluation/` and is not part of the runtime import chain.
