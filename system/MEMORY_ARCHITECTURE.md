# Gated vector-memory architecture

Updated: 2026-08-31

The live `system/shopping_agent` path keeps the existing per-session short-term state and stores at most one normalized long-term vector per explicit user.

## Lifecycle

`Agent.reset(session_id, profile)` remains valid for anonymous sessions. Longitudinal sessions additionally pass `user_id` and a monotonic `sequence_index`. Reset freezes the user vector visible at that instant, rejects overlapping sessions for one user, and validates embedding-space identity.

Every turn calls `respond(session_id, user_message, turn, top_k, buyer_mode="buying")`. `buyer_mode` accepts only `buying` or `browsing`, but remains a caller fallback. Live message intent and active hard conditions are authoritative for routing and clarification; concrete buying evidence wins over exploratory wording, while genuine resets such as `start over` return to Browsing.

`end_session` runs only after scoring. It reconciles negatives again, then serializes sorted positive `disclosed_slots` as `attribute: value` fragments. Category, department, budget, negatives, transcript, outcomes, purchases, target data, asked attributes, and retrieval bookkeeping never enter the update text. Embedding or store failure leaves the session active for retry. Failed response turns restore Fast Memory to its entry snapshot. Failed streams call `discard_session`; successful debug traces are bounded to 32 and may be consumed immediately.

## Ranking

The canonical active query contains current category, department, and positive disclosed-slot values. Its normalized embedding is `v1`; the reset-time user vector is `v2`.

```text
gate = cosine(v1, v2)
threshold = 0.20

cold start or gate failure: s3 = s1
buying:                    s3 = 0.8*s1 + 0.2*s2
browsing:                  s3 = 0.2*s1 + 0.8*s2
```

Here `s1 = catalog @ v1` and `s2 = catalog @ v2`. The normalized full catalog matrix is scored on every turn. FTS5/BM25 routes the candidate pool using the frozen Buying thresholds 15/10 and Browsing thresholds 30/15; both lexical and 150-row vector-fallback pools are still ordered only by `s3` and ASIN. There are no popularity/category boosts, seen-product exclusion, or diversity reshuffling.

After scoring, hard masks apply price, demographic department, minimum rating, minimum review count, requested brand/store, and whole-token negative exclusions. Unknown rating/review metadata receives benefit of doubt. Eligible rows sort by descending `s3`, then ascending ASIN. Constraints are never relaxed to fill `top_k`.

## Update and storage

Cold start stores the normalized new-preference vector directly. Later positive evidence applies `v3 = normalize(0.70*v2 + 0.30*new_preferences)`. An effectively zero mixture falls back to `new_preferences`. An empty positive-slot set advances chronology without changing the vector or update count.

`InMemoryVectorMemoryStore` defines the user-partitioned chronology and immutable state containing user ID, vector, embedding-space ID, last committed sequence, and update count. Active sessions may be cancelled without a commit. The demo wraps those exact semantics with `JsonFileVectorMemoryStore`, which transactionally persists unchanged snapshot schema version 2 after commits and resets and restores all in-memory maps if persistence fails. Former QLMP version-1 payloads are explicitly rejected and are not migrated.

## Trace

Each response exposes a vector-free `debug.memory_trace`: mode, gate cosine/pass, threshold, applied `a`/`b`, vector availability, embedding-space checks, full row count, filter counts, returned `s1`/`s2`/`s3`, and final ASINs. Raw vectors are never logged.

The disconnected QLMP implementation and experiments are retained only in `docs/archive/legacy_qlmp/`. Current evaluator and calibration evidence is retained in `docs/archive/research_evaluation/` and is not part of the runtime import chain.
