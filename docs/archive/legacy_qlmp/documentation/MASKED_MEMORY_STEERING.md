# Interaction-Masked Memory Steering

## Problem

Blindly applying an entire long-term profile can cause negative transfer. A shopper may historically like black clothes, Nike, wide shoes, and mechanical keyboards, then ask: “I need running shoes.” The current purchase intent should dominate; unrelated keyboard and clothing history should not move retrieval merely because it is in memory.

This is an isolated longitudinal variant of the frozen `M0_OPENAI` baseline, not a submission agent or a new memory architecture.

## Core hypothesis

> Current intent can act as a sparse gate over long-term memory.

The claim is deliberately modest: coordinate-wise interaction may be a useful deterministic heuristic for suppressing historical information weakly expressed in the current intent.

## Mathematical definition

Let `q in R^d` be the normalized current M0 intent embedding and `m in R^d` the normalized equal-weight aggregate of eligible memories from sessions before the current session. Define:

- `z = |q ⊙ m|`, the absolute coordinate-wise interaction;
- `mask_k(z)`, a binary mask retaining the largest `ceil(keep_ratio * d)` values of `z` (ties prefer lower coordinate indices);
- `m_clean = mask_k(z) ⊙ m`, the masked memory, intentionally not renormalized;
- `lambda` (`lambda_memory`), the soft steering strength;
- `q*`, the normalized vector passed to the unchanged M0 scorer.

The main method is:

```text
q* = normalize(q + λ(mask_k(|q ⊙ m|) ⊙ m))
```

Not renormalizing `m_clean` makes aggressive masks naturally contribute less energy. Renormalization would erase that consequence and make every retained fraction exert roughly full memory strength.

## Intuition

For:

```text
q = [0.80, 0.02, 0.60, 0.10]
m = [0.70, 0.90, 0.05, 0.80]
z = [0.56, 0.018, 0.03, 0.08]
```

At `keep_ratio = 0.50`, coordinates 0 and 3 survive, so `m_clean = [0.70, 0, 0, 0.80]`. The other memory coordinates do not steer this query.

## Why interaction masking rather than memory magnitude

A large dense-embedding coordinate is not inherently an important or human-readable feature. Ranking by `|m_j|` alone ignores the current request. Ranking by `|q_j * m_j|` asks for simultaneous local expression in current intent and history. It does not imply that a coordinate means “brand,” “colour,” or any other independent concept.

## Buyer-specific rationale

Buyer intent is explicit and strong, so the current query gates memory. Buyer-only activation occurs in the experimental evaluator/harness through its existing `scenario_type == "buying"` field. That evaluator-private label is converted to the `is_buyer` control argument; it is never passed to `Agent.respond`, prompt construction, or the official response. Non-Buyer cases use the unchanged M0 vector.

## Architecture

```text
Current Buyer Query
        |
        v
 Intent Embedding q
        |
        +------------------+
                           |
Long-Term Memories         |
        |                  |
        v                  |
 Memory Embedding m        |
        |                  |
        +----> |q ⊙ m| <---+
                  |
             Top-k Mask
                  |
                  v
          Cleaned Memory
                  |
                  v
       q* = normalize(q + λm')
                  |
                  v
        Existing M0 Retriever
```

## Baselines

- M0: `q_final = q` (no longitudinal memory).
- M1: `q_final = normalize(q + lambda * m)` (raw memory steering).
- M2: retain memory coordinates selected by `|q|`, then steer (query-only mask ablation).
- M3: retain memory coordinates selected by `|q ⊙ m|`, then steer (main method).

All four call the exact existing `dense_retrieve_vector` scorer, whose catalog operation is `np.dot(catalog_embeddings, query)`.

## Parameters

- `keep_ratio`: fraction of coordinates retained by M2/M3; default `0.20`. A small preregistered future grid is `{0.10, 0.20, 0.30, 0.50}`.
- `lambda_memory`: memory steering coefficient; default `0.20`. A small future grid is `{0.10, 0.20, 0.30}`.

Parameters must be selected on development data, not retrospectively optimized on final evaluation users.

## Implementation

- `masked_memory_steering.py`
  - `interaction_mask_memory`: standalone M3 mask.
  - `aggregate_user_memory`: equal-weight eligible-memory aggregate.
  - `prior_memory_items`: strict user/space/`sequence_index < t` selector for replay snapshots.
  - `steer_query`: M0/M1/M2/M3 vector operation.
  - `steer_query_with_diagnostics`: research diagnostics.
  - `score_snapshot_variants`: identical-condition replay through the M0 scorer.
  - `summarize_variant_sessions`: aggregate metrics while preserving session rows.
- `../tests/test_masked_memory_steering.py`: focused mechanics, chronology, and parity tests.

## Memory construction and temporal isolation

The experiment reuses existing `MemoryItem.embedding` vectors; it makes no embedding calls. It takes records from `InMemoryUserMemoryStore.get_records(user_id, before_sequence_index=t)`, whose strict predicate is `record.sequence_index < t`. The evaluator reads visible memories before a session and commits current-session memory only after scoring.

Eligible positive/neutral vectors receive equal weight, are averaged, and the aggregate is normalized once. Existing QLMP baselines exclude negative-polarity items, so this experiment does too: negative memory text already encodes its semantics, and subtracting it would introduce a second incompatible polarity convention. A zero aggregate is treated as no memory.

## Evaluation

Freeze each existing effective M0 query once, obtain only prior visible records, and call `score_snapshot_variants(..., is_buyer=scenario_type == "buying")`. Persist every session’s product IDs, scores, target rank, hit, reciprocal rank, and diagnostics. Report the evaluator’s recommendation metrics per method plus per-session deltas, including raw-memory gains/losses and cases where M3 rescues or causes negative transfer. M0 parity must be exact.

The checked-in test suite uses synthetic vectors and does not claim an empirical recommendation improvement. A real evaluation requires the existing frozen OpenAI/catalog embeddings and longitudinal replay artifacts.

## Limitations

- Dense embedding coordinates are distributed latent coordinates, not independently interpretable semantic dimensions.
- Coordinate masking is a heuristic, not learned attention or feature attribution.
- Top-k and lambda are hyperparameters.
- This first experiment addresses only Buyer scenarios.
- Historical memory can still be wrong, stale, or misleading.
- Empirical held-out evaluation determines whether the idea works.

## What would falsify the idea

Evidence against the hypothesis includes M3 performing no better than M0; M3 consistently underperforming raw-memory M1; useful memory being destroyed by masking; high instability across preregistered keep ratios; or gains appearing only on tuning fixtures and disappearing on held-out users/sessions. Such a result should be preserved rather than prompting post-hoc method changes.
