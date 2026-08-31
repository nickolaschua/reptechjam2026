# Fast/Slow memory architecture

> **Reference implementation scope:** This document describes `techjam-conversational-search/starter` and `techjam-conversational-search/memory`, not the Nickolas QLMP/longitudinal evaluation stack.

## Contract

Longitudinal memory exists only for an explicit stable user identity. It is
local to one `Agent` instance and has no evaluator, database, network, target,
purchase, or outcome dependency.

```text
message -> FastMemoryState -> legacy candidate pool -> optional Slow rerank

successful end_session -> deterministic summary -> one embedding -> one episode
```

`MemoryConfig` has five fields: `memory_enabled`, `tau`, `lambda_memory`,
`candidate_depth`, and `embedding_model_id`. M0 sets only `memory_enabled=False`;
M1 sets it to `True`. Both modes otherwise execute the same lifecycle.

## Fast Memory

`FastMemoryState` is mutable only while a session is active. It stores:

- category and category source turn;
- buying/browsing intent and intent source turn;
- positive hard constraints and soft preferences;
- negative constraints;
- typed budget, material, color, size, style, brand, use-case, and feature facts;
- current turn, raw messages, intent epoch, and explicit topic-override state.

The deterministic parser retains the three organizer update forms: initial
category/requirement, semicolon disclosures, and same-topic replacement. It
also recognizes typed free-form facts and topic shifts. A semantic parser may
instead return a typed authoritative update for a message. Fast Memory has no
embedding or retrieval index. Only positive facts enter the legacy lexical
query; negatives are preserved for the final summary.

## Visibility and storage

`InMemoryMemoryStore` partitions immutable episode tuples by `user_id`. At
`begin_session`, it captures only episodes whose sequence indexes are lower
than the new session index. This tuple is the session's immutable visibility
snapshot. Later commits cannot change it. Duplicate session IDs, overlapping
sessions for one user, negative/out-of-order sequence indexes, and duplicate
completion are rejected.

A successful completion commits exactly:

```text
SlowMemoryEpisode(
  user_id,
  session_id,
  sequence_index,
  summary_text,
  embedding,
  embedding_space_id,
)
```

Lifecycle outcome arguments remain accepted so orchestrators need no branching,
but their contents never affect summary text or embeddings.

## Aggregation and ranking

M1 filters the begin-time snapshot to the current user, lower sequence indexes,
and the exact product-index embedding space. With current sequence `s`:

```text
age_i = s - i
w_i = exp(-age_i / tau)
slow = normalize(sum(w_i * episode_i))
```

No query embedding, relevance gate, contradiction score, route planner, dense
query blend, or structured retrieval route exists. The exact/BM25 starter policy
produces the sole candidate pool. For one-based baseline rank `r`:

```text
score(product) = 1/r + lambda_memory * cosine(product_embedding, slow)
```

Ordering is descending score, then original rank, then ASIN. Any disabled,
empty, missing-vector, unknown-product, dimensional, or embedding-space failure
returns the input order exactly.

The route decision and output pool widths are separate. Requested `top_k`
continues to decide whether the exact tier invokes BM25, while
`candidate_depth` controls how many already-ranked products may enter the
rerank. This is necessary for legacy M0 parity.

## Embedding safety

The known 50,000-row MiniLM cache is memory-mapped only after validating both
file hashes, catalog row count, 384 dimensions, float32 dtype, finite values,
and unit normalization. Episodes and products carry embedding-space IDs. If
MiniLM is not locally available, summary embeddings fall back to deterministic
SHA-256 lexical hashing. Lexical episodes are retained but cannot be compared
with MiniLM products.

## Known limitation

One whole-session embedding is deliberately unsophisticated. It may entangle
separate topics or represent a negated phrase close to the phrase it rejects.
This baseline keeps those facts auditable in summary text but does not add a
second structured filter or retrieval path to compensate.

The `system/README.md` and `system/MEMORY_ARCHITECTURE.md` files
describe prior research and are intentionally left untouched as historical
documents.
