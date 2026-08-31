# Current codebase summary

Updated: 2026-08-30

This document maps the implementations that still matter. For detailed memory mechanics, see [`MEMORY_ARCHITECTURE.md`](MEMORY_ARCHITECTURE.md). For commands and fixture contracts, see [`shopping_agent/longitudinal_eval/README.md`](shopping_agent/longitudinal_eval/README.md).

## Sources of truth

| Concern | Authoritative source |
|---|---|
| Active longitudinal agent | `nickolas/shopping_agent/agent.py` |
| Memory scoring and evidence serialization | `nickolas/shopping_agent/vector_memory.py` |
| Memory persistence and snapshots | `nickolas/shopping_agent/memory_store.py` |
| Longitudinal lifecycle runner | `nickolas/shopping_agent/run_longitudinal_eval.py` |
| Fixture schema and workflow | `nickolas/shopping_agent/longitudinal_eval/README.md` |
| Competition evaluator | `techjam-conversational-search/evaluator/local_evaluator.py` |
| Competition/reference agent | `techjam-conversational-search/starter/agent.py` |
| Organizer baseline | `techjam-conversational-search-participant-kit/` |

Executable source and tests take precedence over Markdown.

## Active longitudinal system

`nickolas/shopping_agent.Agent` owns current-session shopping state and accepts an explicit longitudinal identity through its extended `reset` contract. It uses an `InMemoryVectorMemoryStore` for one normalized vector per user. Every turn scores the complete normalized catalog matrix with the current query and the reset-time user vector, applies the fixed relevance gate and caller-supplied buyer mode, then enforces only budget and negative-term masks.

The runner creates or receives one store for an evaluation arm, processes sessions in sequence, scores the session, and only then commits completion through `end_session`. History snapshots can be replayed into fresh stores for paired controls.

```text
users_40.json
  -> source rows from public_set.jsonl
  -> constant fixture profile replaces source-row profile
  -> Agent.reset(session, profile, user_id, sequence)
  -> turn evaluation
  -> post-score Agent.end_session(...)
  -> user-partitioned memory snapshot
```

### Profile source

The longitudinal benchmark does not use unrelated profiles from the selected public rows. Each synthetic user has one `constant_profile` in `longitudinal_eval/users_40.json`; `_runtime_sample` installs that profile into every source session for the user.

### Evaluation boundaries

- Stable identity and chronology come only from the fixture; they are never inferred from aggregate profiles or session IDs.
- Targets, evaluator annotations, and private projector labels remain evaluator-only.
- Session completion is delivered after recommendation scoring.
- Cross-user memories and memories from the current or future sequence are rejected.
- Buyer mode is supplied by the caller and is required whenever reset found history.
- Only sorted positive disclosed slots are embedded at post-scoring commit.
- Result directories are evidence artifacts, not runtime inputs or current architecture documentation.

## Other implementations

### `techjam-conversational-search/`

The competition/reference tree contains the deterministic evaluator, catalog, submission-style starter agent, and a separate reference memory package. Its standard public evaluator uses anonymous sessions and therefore does not exercise longitudinal identity or completion lifecycle. Documentation inside this directory describes this implementation only.

### `experiment_1/`

This is an older independent hybrid-agent sandbox. It is useful for comparison but is not authoritative for the Nickolas longitudinal runner or vector-memory store. Its local README and agent document are scoped to that sandbox.

### `nickolas/experiments/` and `nickolas/results/`

The experiment harness and eleven retrieval experiments are reproducible research. Result summaries, freeze manifests, and benchmark reports describe the exact run that produced them. They must not be rewritten to match later code and must not be cited as current runtime behavior unless their recorded source hashes match.

## Documentation policy

- Update this file when implementation ownership or entrypoints change.
- Update `MEMORY_ARCHITECTURE.md` when memory behavior changes.
- Update the longitudinal README when fixtures, commands, validation, or result schemas change.
- Put obsolete proposals and milestone reports in `archive/` with explicit historical scope.
- Do not create new root-level “current architecture” audits.
