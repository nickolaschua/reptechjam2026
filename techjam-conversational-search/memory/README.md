# Fast/Slow longitudinal memory

This package is a deliberately small, process-lifetime memory baseline for the
live `starter.Agent`. It is activated only when `Agent.reset` receives a stable,
non-empty `user_id`. Anonymous evaluator sessions keep using the original
template parser and exact/BM25 ranker without constructing `MemorySystem`.

## Modes

```python
from starter.agent import Agent

m0 = Agent("data/catalog.jsonl", memory_mode="M0")
m1 = Agent("data/catalog.jsonl", memory_mode="M1")  # explicit-user default
```

- M0 updates Fast Memory and commits completed episodes, but never reads Slow
  Memory while ranking.
- M1 performs the same writes and may apply one Slow Memory rerank.

For a custom experiment, pass `memory_config=MemoryConfig(...)` instead of
`memory_mode`. Passing both is rejected. Only M0 and M1 are valid; the retired
A0-A7 names are not aliases.

## Lifecycle

```python
agent.reset("session-4", {}, user_id="user-17", sequence_index=3)
response = agent.respond("session-4", "I'm looking for hiking shoes.", 1, 10)

# Score the response before making the session visible to later sessions.
episode = agent.end_session(
    "session-4",
    {"status": "completed"},        # accepted, intentionally ignored
    purchased_product="B000...",    # accepted, intentionally ignored
)
```

`begin_session` freezes a visibility snapshot containing only already committed
episodes for the same user with lower sequence indexes. One user cannot have
overlapping active sessions. Sequence indexes are non-negative and strictly
monotonic, with gaps allowed. A session becomes visible only after a successful
`end_session`; duplicate completion is rejected.

The lifecycle-focused `MemorySystem` surface is:

- `begin_session`
- `update_session`
- `override_intent`
- `rerank_candidates`
- `end_session`
- `get_debug_trace`

## Fast Memory

`fast_memory.py` retains the deterministic official-template parser and adds a
typed free-form fallback. `FastMemoryState` records category and intent source
turns; hard, soft, and negative facts; the budget, material, color, size, style,
brand, use-case, and feature kinds; topic overrides; and intent epochs.

An optional `SemanticParser` can return an authoritative `FastMemoryUpdate`.
Returning `None` delegates that message to the deterministic parser. Fast
Memory is never embedded. Its active positive constraints feed the existing
exact/BM25 candidate policy; negatives remain structured facts for distillation
and do not create another filter or retrieval route.

## Slow Memory

At successful completion the final Fast Memory is rendered in this fixed order:

```text
category=...; intent=...; hard facts: kind=value; soft facts: kind=value; negatives: kind=value
```

That text is embedded once and stored as one immutable `SlowMemoryEpisode`.
Targets, purchases, outcomes, profiles, and external evidence are never included.
The active Fast Memory state is removed only after the episode commit succeeds.

For current sequence `s`, compatible visible history is aggregated as:

```text
age_i = s - episode_i.sequence_index
weight_i = exp(-age_i / tau)
slow_vector = normalize(sum(weight_i * episode_i.embedding))
```

The existing ranker supplies a candidate pool without changing its `top_k`
exact-vs-BM25 route decision. M1 then applies exactly one score:

```text
final_score = 1 / baseline_rank + lambda_memory * cosine(product, slow_vector)
```

Ties use original baseline rank, then ASIN. M0, cold start, incompatible spaces,
and missing product vectors return the supplied baseline order unchanged.

## Embeddings and limitations

The frozen MiniLM cache is accepted only after catalog/cache hashes, row count,
dimension, dtype, finiteness, and normalization validate. Embedding-space IDs
must match exactly; equal dimensions are insufficient. If the local encoder is
unavailable, episode summaries use deterministic lexical feature hashing. That
fallback cannot rerank MiniLM products because the spaces are incompatible.

Whole-session summary embeddings are intentionally a simple baseline. A single
vector can blur negation and unrelated preferences. Negatives remain visible in
the summary but are not independently enforced during candidate retrieval or
reranking.
