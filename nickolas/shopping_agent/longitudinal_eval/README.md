# Longitudinal evaluation fixtures

`fixture_small.json` remains the Phase 5 lifecycle regression fixture (2 users
by 3 sessions). `users_40.json` is the Phase 6 research benchmark: four
synthetic users by ten chronological public buying sessions, with real
catalogue targets and evaluator-private shopper directives.

Offline checks (no model or embedding calls):

```powershell
python nickolas/shopping_agent/longitudinal_eval/validate_fixture.py
python nickolas/shopping_agent/longitudinal_eval/candidate_discovery.py
python nickolas/shopping_agent/longitudinal_eval/microbenchmark.py
```

## Core schema and privacy

Each user has an explicit `user_id`, a broad official-compatible
`constant_profile`, evaluator-control `shopper_private_persona`, and ordered
sessions. Runtime IDs are `{user_id}_s{sequence_index}`.

Phase 6 sessions add `session_role`, `longitudinal_directive`,
`expected_memory`, `target_attribute_audit`, and replay diagnostics. These are
evaluator-only. Directives enter only the shopper LLM's private prompt. The
shopping Agent receives the generated customer message, never the directive,
catalogue audit, latent facts, or expected-memory labels.

Memory facts still come only from final active Nickolas Fast Memory: meaningful
category, non-default price, active disclosed slots, and negated terms. History,
asked attributes, seen ASINs, debug data, stashed terms, outcomes, targets, and
evaluator annotations are excluded. Each fact becomes an atomic QLMP
`MemoryItem` and is embedded in the exact M0 OpenAI embedding space.

## Phase 6 replay

The same S10 fixture object is fingerprinted and replayed with a fresh store for
`NO_HISTORY` and a filtered snapshot for `FULL_HISTORY`. U3 additionally uses
chronological prefixes H0/H1/H3/H5/H9. Store snapshots retain commits,
chronology, MemoryItems, vectors, and embedding-space IDs, but normal result
logs serialize vector-free memory descriptions.

The CLI pins an explicit shopper provider (`ollama` by default) and logs the
actual provider/model. It does not run counterfactual probes unless
`--replay-probes` is supplied. Historical memory remains shadow-only in every
Phase 6 mode: it is not read by query construction, candidates, FTS, dense
retrieval, reranking, diversity, recommendations, or response prompting.
