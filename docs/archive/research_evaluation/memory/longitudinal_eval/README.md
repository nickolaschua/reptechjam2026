# Longitudinal evaluation

`users_40.json` remains available for the legacy four-user evaluation. The frozen
`users_40_v2.json` fixture is the trustworthy-evaluation candidate: 40 isolated
timelines, 10 per diagnostic class, with two setup sessions and one probe each.
It is built only from catalogue evidence by `build_fixture_v2.py`; candidate
selection never observes M3 outcomes.

The runner supplies explicit user identity and chronology, chooses `buying` or `browsing` outside the Agent, passes that mode on every turn, scores recommendations, and only then calls `end_session`. Targets, private directives, outcomes, purchases, and evaluator annotations never contribute to long-term update text.

The current fixture's source rows are buying scenarios. Snapshot replay uses versioned `gated-vector-memory` payloads containing one normalized state per user plus chronology commits. Normal logs contain vector-free descriptions and per-turn memory traces.

Legacy QLMP, projector, portability, masked-memory, relevance experiments, documentation, fixtures, results, and tests are archived under `docs/archive/legacy_qlmp/` and excluded from active imports and test discovery.

Offline fixture validation:

```powershell
python nickolas/shopping_agent/longitudinal_eval/validate_fixture.py
```

Run the active evaluator with its normal provider/cache arguments through:

```powershell
python -m nickolas.shopping_agent.run_longitudinal_eval
```

Run v2 (requires the production OpenAI query-embedding credential; catalogue
embeddings are loaded from the frozen cache):

```powershell
python -m nickolas.shopping_agent.run_longitudinal_eval_v2 --deterministic-rerun --complete-tests-passed
```

The v2 command writes `manifest.json`, `sessions.jsonl`, `vectors.npz`,
`metrics.json`, and `report.md`. `verify_bundle()` reconstructs ranks and metrics
without embedding or network calls and rejects artifact or vector tampering.
