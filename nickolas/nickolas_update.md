# Nickolas development update

## Executive summary

This push establishes the controlled baseline and most of the infrastructure
needed to evaluate longitudinal memory in the shopping agent. It includes a
stable OpenAI-embedding M0 baseline, embedding-provider comparison tooling, a
standalone QLMP implementation, safe cross-session memory plumbing, and a
longitudinal evaluation framework.

The main distinction teammates should keep in mind is:

- **implemented and tested** does not always mean **experimentally selected**;
- M0/B0 remains the no-longitudinal-memory control;
- B1 and B2 have integration primitives but still need full benchmark runs;
- B3 query steering is intentionally not enabled yet.

## Status at a glance

| Workstream | Status | What that means |
| --- | --- | --- |
| Fast Memory hardening | Ready | The existing within-session state model has stricter normalization, safer route fallback, and lifecycle capture. |
| Embedding backend abstraction | Ready | The agent can use isolated OpenAI or BGE backends through a shared validated interface. |
| Phase 3 embedding bake-off tooling | Ready | Controlled fixture generation, smoke testing, retrieval comparison, and end-to-end comparison commands are implemented. |
| M0_OPENAI baseline interface | Ready | The no-longitudinal-memory OpenAI baseline and dense-vector scoring seam are implemented and tested. |
| Standalone QLMP mathematics | Ready | Geometry, selection baselines, projection, aggregation, bounded steering, diagnostics, and synthetic fixtures are implemented. |
| Longitudinal memory lifecycle | Ready | Preference extraction, embedding, storage, user isolation, chronology, and end-session commit behavior are implemented. |
| Longitudinal benchmark framework | Ready | The 40-session fixture, replay, snapshots, privacy checks, parity checks, and metric collection are implemented. |
| B0 shadow validation | Ready | A full baseline run exists and historical memory is observed without affecting ranking. |
| M0-to-QLMP contract | Ready with minor follow-up | Ownership, dtype, candidate-universe, fallback, and evaluation boundaries are documented. The M0 freeze manifest should be refreshed before new scientific runs. |
| B1/B2 memory-active evaluation | In progress | Component integration exists, but full controlled benchmark results have not yet selected or validated a final treatment. |
| Projector isolation | In progress | Isolation tooling exists to test projection independently before steering is allowed. |
| B3 QLMP steering | Deferred | Deliberately blocked until projector isolation and selection-policy validation pass. |
| Final production memory policy | In progress | No memory strategy or tuned configuration has been declared the winner. |

## Ready to use

### 1. Hardened Fast Memory

The Nickolas agent retains the `experiment_1` within-session Fast Memory
concept while tightening its behavior:

- disclosed slot values are normalized consistently;
- the LLM-produced slot map represents the complete active state, so revoked
  constraints can be removed;
- state is restored if an attempted route falls back, preventing abandoned
  routing work from mutating memory; and
- the final Fast Memory state can be captured at session end.

This is locally implemented in `nickolas/shopping_agent/agent.py`; M0 does not
import Yangxu's agent at runtime. See `nickolas/fast_memory_developments.md` for
the detailed comparison.

### 2. Embedding backends and bake-off

OpenAI and BGE embeddings now share an explicit backend contract. Cache reuse
is guarded by model, dimension, normalization, catalogue fingerprint, product
text version, and exact row ordering.

The Phase 3 tooling supports:

- deterministic controlled retrieval fixtures;
- an OpenAI smoke test that does not embed the catalogue;
- paired retrieval comparison;
- end-to-end evaluator runs; and
- combined result summaries.

The controlled Phase 3 evidence favored vanilla BGE, but this does not silently
change M0_OPENAI. OpenAI remains the pinned development baseline so experiments
remain attributable and reproducible.

### 3. M0_OPENAI baseline

M0_OPENAI preserves the existing hybrid recommendation path:

```text
current-session Fast Memory
  -> lexical/FTS retrieval
  -> conditional dense OpenAI fallback
  -> deterministic reranking and exclusions
  -> diversification
  -> recommendations
```

It exposes a validated dense-vector interface so experimental query vectors can
later be scored without duplicating or changing catalogue scoring. Historical
memory does not affect M0 recommendations.

### 4. QLMP package

The standalone NumPy QLMP package includes:

- normalized query and memory geometry;
- query-local tangent residuals and local product subspaces;
- memory scope and polarity handling;
- naive and cosine-weighted memory baselines;
- projected residual aggregation;
- bounded angular steering; and
- serializable selection, projection, and steering diagnostics.

The package does not depend on the shopping agent. This keeps the mathematical
mechanism testable independently from retrieval and evaluation orchestration.

### 5. Longitudinal memory and evaluation plumbing

The integration now supports:

- explicit user IDs and chronological session indexes;
- extraction of atomic preferences from final Fast Memory;
- embeddings in the same space as the active M0 backend;
- in-memory user stores with cross-user isolation;
- visibility of prior memories without future-session leakage;
- negative-memory polarity;
- snapshots and clean counterfactual replay; and
- a 40-session research fixture across multiple longitudinal behaviors.

Ground-truth targets, evaluator annotations, purchase outcomes, and future
preferences are kept out of memory and ranking inputs.

### 6. B0 validation

B0 is the M0_OPENAI control with Fast Memory enabled and longitudinal history
shadow-only. A full validation artifact exists. The framework verifies that
history can be loaded and audited while remaining behaviorally inert for
ranking. B0 results characterize benchmark difficulty; they are not evidence
of a memory benefit.

## In progress

### B1 and B2 treatments

The integration layer can run exact M0, B1, and B2 component retrieval through
the unchanged M0 scorer. These modes still require full, controlled execution
on the same longitudinal fixture before the team should interpret their effect
or compare them as candidate final systems.

Pending work includes:

- run B1 and B2 against the frozen B0 fixture and transcripts;
- keep the candidate-universe choice explicit and ablated;
- compare retrieval quality, negative-preference violations, overrides,
  stability, and efficiency; and
- select configurations only from measured results.

### Projector isolation and B3

Projector-isolation code is present, but projection must first demonstrate that
it preserves useful local directions and rejects unsupported memory components.
B3 steering intentionally raises a deferred/error path rather than pretending
that an unvalidated projection rule is production-ready.

Before B3 is enabled:

1. complete projector-isolation experiments;
2. choose and freeze the projection/selection policy;
3. verify bounded steering and fallback behavior end to end; and
4. run B3 on the identical benchmark used for B0/B1/B2.

### Scientific freeze and final selection

The formal M0-to-QLMP contract is usable, but the recorded M0 freeze hashes
predate the latest dense interface. Refresh the freeze manifest before further
scientific runs. No final embedding backend, memory mode, steering strength, or
candidate universe has been selected for production.

## Compatibility and collaboration notes

- Existing callers can continue to call `Agent.reset(session_id,
  user_profile)` without enabling longitudinal memory.
- Longitudinal mode requires explicit user identity and sequence metadata.
- `end_session()` is the commit boundary for extracting durable preferences.
- Do not feed private evaluator annotations or target ASINs into agent, shopper,
  query, or memory inputs.
- Do not describe shadow-visible history as memory-active ranking.
- Do not change the M0 scorer when evaluating QLMP; treatments should produce a
  query vector and hand it back to the same scorer.
- Keep OpenAI and BGE caches separate and do not introduce silent backend
  fallback.
- Large generated result folders, graph outputs, logs, caches, and archive
  bundles should be reviewed separately from source code before committing.

## Verification state

The combined QLMP and shopping-agent test suites currently pass:

```text
155 passed
110 subtests passed
2 dependency deprecation warnings
```

The warnings come from imported SWIG types and are not test failures.

## Recommended next sequence

1. Review and commit the source, tests, fixtures, configs, and documentation as
   coherent changes separate from generated outputs.
2. Refresh and record the M0 scientific freeze.
3. Run B1 and B2 using the frozen B0 fixture and replay inputs.
4. Complete projector isolation and decide whether B3 is allowed to proceed.
5. Run the approved B3 treatment, if any.
6. Compare all conditions and select the final memory policy from evidence.
