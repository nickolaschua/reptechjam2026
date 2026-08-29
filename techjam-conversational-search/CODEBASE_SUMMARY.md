# Live codebase summary

Status: 2026-08-29

The live project is a deterministic conversational product-search agent over a
frozen 50,000-product catalog. `evaluator/local_evaluator.py` runs 200 public
anonymous sessions. `starter/agent.py` owns conversation parsing and the
exact/BM25 recommendation policy. `memory/` adds optional longitudinal behavior
only for callers that supply stable user identities.

## Runtime paths

Anonymous calls use the competition path unchanged:

```text
reset(session_id, profile)
  -> starter template state
  -> exact phrase ranking
  -> conditional exact + BM25 RRF fallback
  -> top-k recommendations
```

Explicit-user calls use the Fast/Slow extension:

```text
reset(..., user_id, sequence_index)
  -> begin Fast Memory with a frozen history snapshot
respond(message)
  -> update typed Fast Memory
  -> run the same exact/BM25 candidate policy
  -> optionally rerank once from Slow Memory
end_session(...) after scoring
  -> distill final Fast Memory
  -> embed once and commit one episode
  -> remove active Fast Memory
```

The default explicit-user mode is M1. M0 performs all state updates and writes
but does not read history, making it the ranking control for M1.

## Main files

| Path | Responsibility |
|---|---|
| `starter/agent.py` | Legacy anonymous ranker and explicit-user lifecycle wiring |
| `memory/config.py` | M0/M1 policy parameters |
| `memory/types.py` | Typed constraints, Fast state, Slow episode, debug trace |
| `memory/fast_memory.py` | Official-template/free-form parsing and semantic hook |
| `memory/slow_memory.py` | Summary, exponential aggregation, single rerank equation |
| `memory/embeddings.py` | MiniLM provider/cache validation and lexical fallback |
| `memory/store.py` | Per-user isolation, chronology, snapshots, commit guards |
| `memory/integration.py` | Lifecycle orchestration |
| `memory/metrics.py` | Rank and memory-harm metric helpers |
| `memory/tests/` | Focused Fast, Slow, embedding, and integration regressions |

The catalog, evaluator, participant kit, research experiments, and public data
are not modified by memory. Documents with the same names under `nickolas/`
describe the earlier research/A0-A7 prototype and are historical research
artifacts, not the live architecture.

## Ranking compatibility boundary

The exact-vs-BM25 cascade historically uses requested `top_k` to decide whether
an exact tier is too wide. Explicit-user ranking retains that decision width but
may return up to `candidate_depth` candidates for the one rerank. Consequently,
the first `top_k` M0 results match the legacy ranker while M0 and M1 receive the
same baseline pool.

The official evaluator does not pass `user_id` or call `end_session`, so it does
not construct or exercise longitudinal memory.
