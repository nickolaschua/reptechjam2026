# M0_OPENAI ↔ QLMP Formal Integration Contract

> **Archived pre-integration contract:** Future-work statements in this document may already be implemented or superseded.

## A. Verdict

`CONTRACT READY WITH MINOR ISSUES`

The existing M0 vector interface and QLMP APIs are structurally compatible. No redesign of either system is required.

Mandatory implementation details remain:

- Introduce an explicit float32 → normalized float64 working-copy boundary. Real M0 float32 catalogue vectors frequently miss QLMP's default `1e-8` unit tolerance after promotion.
- Refresh the M0 freeze manifest before scientific runs; its recorded hashes predate the current Checkpoint A interface.
- Keep candidate-universe choice explicit and ablated.
- Do projector-isolation before enabling B3 steering.

This contract was produced read-only: no implementation was performed, no evaluator was run, and Graphify was not used.

## B. Current systems

### M0_OPENAI

The canonical implementation is `nickolas/shopping_agent/agent.py`.

Current interfaces:

```python
q = agent.embed_dense_query(query_text)

result = agent.dense_retrieve_vector(
    query_embedding=q,
    top_n=...,
)
```

`DenseRetrievalResult` contains aligned:

```text
query_embedding
row_indices
product_ids
scores
product_embeddings
```

Actual behavior:

- `embed_dense_query()` uses the configured embedding backend and validates the returned vector.
- Under canonical M0, the backend is OpenAI `text-embedding-3-large`, 3072 dimensions, float32, L2-normalized.
- `dense_retrieve_vector()` performs the unchanged full-matrix expression:

```python
all_scores = np.dot(catalog_embeddings, query)
row_indices = np.argsort(all_scores)[::-1][:top_n]
```

- It does not embed text, call OpenAI, consult memory, filter candidates, rerank, or diversify.
- It rejects non-finite, dimensionally incompatible, or non-normalized queries; it does not silently normalize.
- `top_n=0` is accepted and returns an empty result.
- `top_n > catalogue_size` returns the entire catalogue.
- `dense_retrieve_text()` embeds once and delegates to `dense_retrieve_vector()`.
- `_dense_retrieve()` is only the legacy row-index wrapper.
- `DenseQuerySnapshot` captures the current-message/query fields and evaluation-only target metadata.

Important detail: the frozen dataclasses do not make their NumPy arrays intrinsically immutable. Current code and tests establish non-mutation operationally, but the integration layer should own a read-only frozen copy.

The hybrid path remains:

```text
fast lexical route where applicable
or
full route:
    current-session state update
    FTS retrieval
    price filtering
    conditional dense fallback
    M0 heuristic reranking
    negated-term exclusion
    diversification
```

Current constraint semantics must be preserved:

- Price is an exclusion filter, applied after dense Top-N.
- Current negated terms are excluded during post-retrieval scoring.
- Department/category are soft boosts.
- Brand is currently a soft `-10` penalty, not a strict hard filter.

Therefore, the integration must not claim that "Nike only" is currently enforced as a hard filter without separately changing and refreezing M0.

### QLMP

QLMP is a standalone NumPy package and does not import the shopping agent.

Existing public APIs include:

```python
build_naive_memory_baseline(...)
build_cosine_memory_baseline(...)
build_local_subspace(...)
project_memory_residual(...)
aggregate_raw_residuals(...)
aggregate_projected_residuals(...)
bound_query_shift(...)
```

Relevant definitions are in:

- `nickolas/memory/qlmp/models.py`
- `nickolas/memory/qlmp/baselines.py`
- `nickolas/memory/qlmp/geometry.py`
- `nickolas/memory/qlmp/projection.py`
- `nickolas/memory/qlmp/aggregation.py`
- `nickolas/memory/qlmp/steering.py`

Corrections from actual code:

- `build_local_subspace()` currently computes a full thin NumPy SVD and truncates the resulting basis. It is not using an iterative truncated-SVD solver.
- B1 includes positive and neutral memories; negative memories are excluded.
- B2 thresholding is inclusive, Top-K ties preserve input order, and weights are `max(cosine, 0)` followed by normalization.
- Confidence and timestamp are retained but not used in weighting.
- QLMP converts inputs to owned float64 copies and does not mutate M0 arrays.

## C. Ownership matrix

| Responsibility | M0 | QLMP | Longitudinal evaluator / memory subsystem |
|---|---:|---:|---:|
| Raw current user turn | Owner | No | Supplies/replays fixture turn |
| Current-session state | Owner | No | Freezes it for paired runs |
| State-to-query serialization | Owner | No | Captures output |
| OpenAI current-query embedding | Owner | No | Ensures it happens once |
| Canonical frozen `q` | Produces | Consumes a working copy | Owns experimental replay |
| Product catalogue and row ordering | Owner | No | Freezes fingerprints |
| Product text representation | Owner | No | Freezes fingerprint/version |
| Product embedding cache/matrix | Owner | No | Verifies manifest identity |
| FTS and hybrid routing | Owner | No | Replays identical scenario |
| Existing current constraints | Owner | Must not reinterpret | Freezes identical state |
| Dense dot-product scorer | Owner | No | Calls same scorer for all modes |
| Heuristic reranking/diversity | Owner | No | Compares identical pipeline |
| Final recommendations | Owner | No | Evaluates |
| Persistent memory database | No, except current shadow implementation location | No | Owner |
| User identity and chronology | No ranking ownership | No | Owner |
| Candidate-memory order and fixture | No | Consumes | Owner |
| Structured historical eligibility | No | Owner | Supplies scope |
| B1/B2 selection and weighting | No | Owner | Configures, never labels inputs |
| Local tangent geometry | No | Owner | Configures `local_k`/rank |
| B3 projection and later selection | No | Owner | Evaluates |
| Residual aggregation/steering | No | Owner | Uses same config across variants |
| Memory/projection diagnostics | No | Owner | Logs |
| Targets, labels and metrics | No | Must never receive | Owner |

Although the current store resides under `shopping_agent`, `nickolas/shopping_agent/memory_store.py` already models identity and chronology outside `MemoryItem`. That is the correct conceptual boundary.

## D. Query/vector contract

### Canonical vectors

| Vector | Shape | Dtype | Normalization | Owner |
|---|---:|---|---|---|
| `q_m0` | `[3072]` | float32 | M0 unit tolerance | M0 produces; evaluator freezes |
| `q_work` | `[3072]` | float64 | Re-normalized working copy | Integration/QLMP |
| local product working matrix | `[K, 3072]` | float64 | Row-normalized local copy | Integration/QLMP |
| `q_star64` | `[3072]` | float64 | QLMP guarantees unit norm | QLMP |
| `q_star32` | `[3072]` | float32 | Must pass M0 validation | Integration passes to M0 |
| global catalogue | `[N, 3072]` | float32 | Existing M0 representation | M0 only |

All vectors use:

```text
openai-text-embedding-3-large:
text-embedding-3-large:
dimensions=3072:
normalization=l2:
query=symmetric-v1
```

### Exact reuse

For one evaluated state:

1. M0 serializes the frozen current state once.
2. M0 embeds that text once, producing canonical `q_m0`.
3. The evaluator stores one owned, read-only copy.
4. The integration boundary creates `q_work` once by float64 promotion and numerical unit normalization.
5. M0, B1, B2, and B3 receive the same canonical `q_m0`; all QLMP mechanisms receive the same derived `q_work`.
6. B3 first-pass scoring uses `q_m0`, not `q_star`.
7. No mode reruns the shopper, state distillation, serialization, or embedding.

Normalization of `q_work` is numerical conversion only. It is not a new embedding or query representation. No-memory scoring and every fallback use the original bitwise float32 `q_m0`.

### Minor numeric incompatibility

Real cached M0 product vectors are float32 and normalized to M0 tolerances. When promoted without re-normalization, many exceed QLMP's `1e-8` unit tolerance; this occurs in the current 50,000-row cache.

Therefore:

- Do not globally convert the catalogue to float64.
- Normalize only the copied `q_work`, local product rows, and memory working vectors.
- Do not loosen `ProjectionConfig.epsilon` merely to accept float32 norm drift, because that epsilon also participates in projection-fraction computation.
- Ensure future `MemoryItem` creation stores a properly normalized float64 vector.

Expected tests:

```text
||q_work||: within 1e-12
||q_star64||: within 1e-12
q_star32 accepted by M0: current rtol=1e-5, atol=1e-6
actual shift: <= theta_max plus small float32 audit tolerance
none/fallback dense scores: exact array parity
```

QLMP receives copies and may never mutate `q_m0`. M0's scorer also must not mutate it.

## E. Memory contract

Actual `MemoryItem` fields are:

```python
MemoryItem(
    id,
    text,
    embedding,
    source,
    polarity,
    scope=None,
    timestamp=None,
    confidence=1.0,
)
```

Properties:

- Frozen dataclass.
- `embedding` is copied into owned read-only float64 storage.
- `source` and `polarity` are enums.
- `confidence ∈ [0,1]`.
- Scope matching is exact and case-sensitive.
- `MemoryItem` itself does not carry `embedding_space_id`.
- `MemoryItem` validates shape/finiteness but not 3072 dimensions or unit norm.

Consequently, the candidate-memory batch must carry an envelope-level `embedding_space_id`, preferably through existing `StoredMemory` records or a fixture manifest. Bare `MemoryItem` values are insufficient to prove embedding-space identity.

Required memory invariant:

```text
memory model = query model = product model
shape = [3072]
finite = true
normalized float64 working vector = true
embedding_space_id = canonical M0 OpenAI space
```

The evaluator/memory subsystem owns persistence, user lookup, chronology, and candidate order. QLMP receives a frozen ordered tuple and must not:

- Fetch storage.
- Infer identity.
- Change chronology.
- Persist `q` or `q_star`.
- See target IDs or relevance labels.

`q_star` may appear in isolated experiment logs but must never be inserted into `MemoryItem`, `InMemoryUserMemoryStore`, session state, or future memory-writing input.

Negative historical memories remain excluded from soft steering. No embedding subtraction is allowed.

## F. B1 contract

Inputs:

```text
q_work
ordered candidate MemoryItems
canonical query_scope
one BaselineConfig
one shared SteeringConfig
```

Existing flow:

```python
baseline = build_naive_memory_baseline(
    q_work,
    memories,
    query_scope=query_scope,
    config=baseline_config,
)

steered = bound_query_shift(
    q_work,
    baseline.aggregate_delta,
    config=steering_config,
)
```

Semantics:

- Exact scope eligibility.
- Positive and neutral memories eligible.
- Negative memories excluded.
- Every eligible memory selected.
- Raw cosine is diagnostic only.
- Raw tangent residuals.
- Uniform normalized weights.
- No local product retrieval or projection.
- Confidence and recency do not affect weighting.

Output ownership:

```text
MemoryBaselineResult        # QLMP
QuerySteeringResult         # QLMP
q_final                     # integration boundary
DenseRetrievalResult        # M0
```

If no memory is usable or steering reports `delta_zero`, the integration returns canonical `q_m0`, not a round-tripped copy.

Final scoring:

```python
agent.dense_retrieve_vector(q_final, top_n=final_top_n)
```

## G. B2 contract

Inputs are identical to B1.

Existing flow:

```python
baseline = build_cosine_memory_baseline(
    q_work,
    memories,
    query_scope=query_scope,
    config=baseline_config,
)

steered = bound_query_shift(
    q_work,
    baseline.aggregate_delta,
    config=steering_config,
)
```

Semantics:

1. Same structured eligibility and polarity exclusion as B1.
2. Compute raw cosine.
3. Apply inclusive threshold, if configured.
4. Sort descending cosine with original candidate order as tie-break.
5. Select `memory_top_k`.
6. Weight with `max(cosine, 0)`.
7. Normalize non-negative weights.
8. Aggregate raw tangent residuals.
9. Use the shared steering function.

B2 never calls the projector and never consumes local product geometry.

All-zero selected weights produce zero aggregate and therefore canonical `q_m0`.

## H. B3 contract

B3 must remain compositional to preserve the 2×2 ablation.

### First pass

```python
initial = agent.dense_retrieve_vector(
    query_embedding=q_m0,
    top_n=local_k,
)
```

This must always use the original canonical q.

The integration passes these to QLMP:

```text
q_work
normalized float64 copy of initial.product_embeddings
initial.product_ids for diagnostics
initial.scores for diagnostics
```

### Geometry and projection

```python
subspace = build_local_subspace(
    q_work,
    local_product_working_matrix,
    rank=projection_config.rank,
    epsilon=projection_config.epsilon,
)

projections = [
    project_memory_residual(
        q_work,
        memory.embedding,
        subspace.basis,
        epsilon=projection_config.epsilon,
    )
    for memory in structurally_eligible_memories
]
```

Each `MemoryProjection` already exposes:

```text
residual
coefficients
projected_residual
raw_query_memory_cosine
tangent_norm
projected_norm
projection_fraction
```

For the isolation work:

```text
u = residual
delta = projected_residual
rho = projection_fraction
```

### Future selection and aggregation

Projection scoring is deliberately not defined as one opaque B3 function yet.

After the stop/go gate, QLMP will:

1. Apply structured eligibility and polarity exclusion.
2. Compute projections.
3. Select using projection relevance under a preregistered rule.
4. Initially avoid confidence/recency weighting.
5. Choose raw or projected residuals explicitly.
6. Aggregate with `aggregate_raw_residuals()` or `aggregate_projected_residuals()`.
7. Use `bound_query_shift()`.

This preserves:

```text
A cosine selection     + raw residual
B cosine selection     + projected residual
C projection selection + raw residual
D projection selection + projected residual
```

### Second pass

```python
final = agent.dense_retrieve_vector(
    query_embedding=q_star32,
    top_n=final_top_n,
)
```

No OpenAI call, new text embedding, new matrix, or QLMP-owned scorer is allowed.

`local_k` belongs to the QLMP experiment config:

- Initial expected value: `500`, explicitly untuned.
- `local_k <= 0`: integration config error.
- `local_k > catalogue_size`: M0 naturally returns all rows.
- Under a smaller explicit candidate universe: return `min(local_k, eligible_count)`.
- Empty or rank-zero neighbourhood: canonical q fallback.

## I. Candidate-universe issue

There is a real discrepancy.

### Current M0 semantics

```text
full-catalogue dense scoring
→ full-catalogue Top-N
→ price exclusion
→ current-negation exclusion/reranking
→ diversification
```

Price filtering occurs after dense Top-N. There is no consolidated pre-dense hard-filter mask in the scorer.

### Theoretical QLMP assumption

```text
current hard filters
→ eligible product universe
→ dense local Top-K
→ local tangent geometry
```

### Option A — exact M0 universe

Requires only:

```text
q_m0
local_k
dense_retrieve_vector()
```

It maximizes scorer and baseline parity.

### Option B — eligible-product local geometry

Requires:

- A precisely defined eligibility predicate.
- A catalogue-row-aligned mask.
- Explicit decisions about price and negated-term exclusion.
- Confirmation that current soft brand/category/department behavior is not silently promoted to hard filtering.
- Aligned eligible IDs and product embeddings.
- A scorer/subset adapter that preserves M0 tie ordering.

### Required treatment

The experiment config and every result must log:

```text
candidate_universe = "m0_full_catalogue" | "post_current_hard_filter"
```

No implicit default may appear in a scientific result.

The first post-integration ablation should compare A and B on the same frozen q and histories, measuring:

- Effective rank and singular spectrum.
- Projector-isolation AUROC/AUPRC.
- Useful/irrelevant `rho` distributions.
- Empirical-null percentiles.
- Later, only after stop/go, downstream retrieval metrics.

The final scorer remains full canonical M0 in both arms. This choice is not a contract blocker.

## J. Longitudinal evaluator contract

One evaluated dense state should contain:

```text
fixture_id
user_id
session_id
sequence_index
turn_index
Buying/Browsing label
raw current user request
frozen current-session state
frozen effective dense query text
canonical q_m0
embedding_space_id
catalogue/product-text manifest ID
target product ID                   # evaluator-only
ordered candidate MemoryItems
candidate-memory embedding-space ID
per-memory relevance labels         # evaluator-only
distractor condition/count          # evaluator-only
query_scope/current_category
```

The target, labels, distractor annotations, and mode label must not be passed to QLMP. The adapter should receive only:

```text
q
query_scope
candidate memories
mode/config
retrieval owner
```

### Simulator profile versus memory

- `constant_profile` and `shopper_private_persona` are evaluator-private behavior controls.
- Agent memory consists only of materialized historical `MemoryItem`s.
- The hidden profile must not be passed to the integration adapter or QLMP.
- Existing `Agent.reset()` already discards the profile for ranking.

### Labels

Fixtures should support per-memory labels:

```text
useful_additional_steering
relevant_but_redundant
irrelevant
same-category-hard-negative
cross-domain-distractor
```

They are evaluation metadata only.

### Distractors

For each fixed useful-memory set, construct candidate histories with exactly:

```text
0, 2, 5, 10, 20, 50
```

irrelevant items. Every mode receives the identical ordered candidate tuple for a condition.

### Buying/Browsing

The label is for stratification only. It must not affect selection, `beta`, or `max_shift_deg` in the first study.

## K. Failure/fallback contract

The canonical fallback is:

```text
q_final = q_m0
```

Fallback conditions:

- No candidate memories.
- No B1/B2 structurally eligible memories.
- B2 selection empty.
- B2 selected weights sum to zero.
- B3 local neighbourhood empty.
- B3 effective local rank zero.
- B3 no memory passes the eventual projection rule.
- Aggregate correction is zero.
- Steering reports `delta_zero`.
- `beta=0` or `max_shift_deg=0`.
- Recoverable B3 numerical failure where normal M0 scoring remains safe.

M0 then proceeds normally.

Every fallback must log a machine-readable reason. A fallback caused by an unexpected exception must be excluded from evidence for B3 success.

The following are invalid-run contract failures, not ordinary no-memory cases:

- Wrong embedding-space ID.
- Wrong dimension.
- Non-finite vectors.
- Target/label leakage.
- Different frozen q or state across modes.
- `q_star32` failing M0 validation.

Retrieval may safely fall back to q for user-facing continuity, but the evaluator must mark that experimental run invalid.

## L. Metrics/logging contract

### M0 owns

- Product IDs.
- Row indices.
- Dense scores.
- Selected product embeddings.
- Final recommendations and existing route/reranking diagnostics.

### QLMP owns

- Selected memory IDs.
- Per-memory cosine/eligibility/weights.
- Local subspace and rank.
- Per-memory projection results.
- Aggregate correction.
- Requested/applied steering.
- Actual shift angle and clipping.

### Integration layer owns

A conceptual result:

```text
memory_mode
frozen_query_reference
q_m0
q_final
candidate_universe
initial_dense_result       # B3 only
final_dense_result
baseline_result            # B1/B2
local_subspace             # B3
memory_projections         # B3
memory_diagnostics
steering_result
fallback_reason
config/manifest IDs
```

### Evaluator owns

- Exact target rank.
- Recall@10.
- Recall@50.
- MRR.
- Later NDCG.
- Useful/distractor counts.
- Buying/Browsing strata.
- Mode comparison and confidence intervals.

For exact full-catalogue rank with the current API, `final_top_n` must equal catalogue size. If only Top-50 is returned, the metric must be named censored rank/MRR@50 rather than exact rank.

### Raw score decomposition

Current objects are sufficient without changing ranking:

```text
raw_query_score = p · q
applied tangent = applied_beta × tangent(aggregate_delta)
memory numerator adjustment = p · applied_tangent
final_score = p · q_star
```

Because `q_star` is normalized:

```text
p · q_star
=
(p · q_work + p · applied_tangent)
/
||q_work + applied_tangent||
```

This belongs in integration/evaluator diagnostics. It can be computed for final-result products using their returned embeddings. M0 should not learn about projection metrics.

## M. Phase 3 projector-isolation contract

Before any B3 steering:

1. Freeze real M0 current state and q.
2. Retrieve the local neighbourhood with original q.
3. Build the local subspace from real M0 product embeddings.
4. Project labelled candidate memories.
5. Do not construct or score `q_star`.

Log per query-memory pair:

```text
query/fixture ID
memory ID/text
private label
raw cosine(q,m)
||u||              = MemoryProjection.tangent_norm
rho                 = MemoryProjection.projection_fraction
||delta||           = MemoryProjection.projected_norm
scope
candidate universe
local_k
requested/effective rank
```

Compare:

```text
raw cosine
rho
projected norm
```

Metrics:

- AUROC.
- AUPRC.
- Useful/irrelevant means and medians.
- Score distributions by label and distractor type.
- Confidence intervals across queries/users, not only pooled pairs.

Empirical nulls must include:

- Other categories.
- Other users.
- Same-user wrong-domain memories.
- Difficult same-category irrelevant memories.

Record mean, median, p95, and p99. `r/(d-1)` is only a sanity check.

Thresholds must come from a development/calibration split or empirical null, never target-product evaluation outcomes.

Stop/go:

> If projection relevance does not show held-out separation beyond raw cosine, stop B3 integration.

Investigate product text, candidate universe, K, rank, neighbourhood multimodality, or the local-geometry hypothesis. Mathematical elegance is not a go criterion.

## N. Integration test plan

### Interface and numerics

- Realistic 3072-dimensional float32 q enters the boundary.
- Float32 norm drift is accepted at the M0 boundary and normalized only in the QLMP working copy.
- Local products are copied, promoted, and row-normalized without modifying the catalogue.
- Memory vectors are normalized float64 before `MemoryItem` use.
- `q_star64` is normalized; `q_star32` passes the unchanged M0 validator.
- M0, QLMP, memory, and snapshot inputs remain byte-for-byte unchanged.

### Retrieval alignment

- First-pass IDs, rows, scores, and product vectors align.
- `local_k=1`, `local_k=N`, and `local_k>N`.
- Integration rejects `local_k<=0`.
- B3 first pass receives q, never q-star.
- Both final q and q-star call the same bound `dense_retrieve_vector` method.

### Mode behavior

- None returns the original q object/value.
- Every mode with no usable memory produces exact M0 dense parity.
- B1 known memories produce the expected uniform raw aggregate and q-star.
- B2 known memories produce expected thresholding, order, weights, aggregate, and q-star.
- B3 synthetic local geometry produces expected projected aggregate and q-star.
- B1/B2/B3 all call the same `bound_query_shift`.
- Angular bound remains valid after float32 cast.

### M0 parity

At dense component level, direct M0 versus `mode="none"` must have:

```text
same q
same row indices
same product IDs
bitwise-equal dense scores
same product embeddings
same ordering
```

At end-to-end level:

- Same fixed transcript and current state.
- Same fast/full route.
- Same dense invocation decision.
- Same FTS candidates.
- Same reranking/diversification.
- Same ordered recommendation IDs.
- Same memory-disabled debug behavior.
- No additional embedding call.

### Leakage

- Changing only the target ID cannot alter effective query text, q, candidate memories, or results.
- Target title/ID never enters embedding inputs.
- Evaluator labels and Buying/Browsing never reach QLMP.
- Hidden simulator profile never reaches QLMP.
- Outcome/purchase/target never becomes memory.
- `q_star` never enters session state or memory persistence.

## O. Recommended orchestration location

Use one small module:

```text
nickolas/shopping_agent/qlmp_integration.py
```

This is the smallest architecture consistent with the repository.

Reasons:

- `shopping_agent` already owns the M0 retrieval interface.
- `memory_adapter.py` already imports QLMP, so dependency direction is established.
- QLMP remains independent and NumPy-only.
- The existing public and shadow evaluators remain untouched.
- The new scientific evaluator can import M0 and the adapter separately.
- The adapter can depend on a small scorer protocol instead of importing `Agent`, avoiding cycles.

The mode belongs in one `QLMPIntegrationConfig` at this boundary:

```text
none
naive
cosine
projection
```

One switch chooses the mechanism. Mode conditionals should not be distributed through M0.

Component-level evaluation should use this adapter first. End-to-end insertion into `Agent._respond_custom()` should wait for projector stop/go.

## P. Minimum future code changes

| File/function | Future change | Reason |
|---|---|---|
| New `nickolas/shopping_agent/qlmp_integration.py` | Define mode/config/result and one orchestration function | Centralize mode selection, conversion, fallback, diagnostics |
| `nickolas/shopping_agent/memory_adapter.py::embed_drafts()` | Normalize the float64 vector before constructing `MemoryItem` | Current float32-normalized vectors may fail QLMP's strict unit check |
| New `nickolas/shopping_agent/longitudinal_eval/qlmp_component_eval.py` | Freeze safe fixture inputs, replay one q across modes, compute component metrics | Avoid modifying the existing public/shadow evaluator |
| New `nickolas/shopping_agent/tests/test_m0_qlmp_contract.py` | Implement the contract tests above | Establish boundary and M0 parity |
| Existing QLMP diagnostics or a small future selection module | Add B3 selection diagnostics only after projector go | QLMP should own projection selection |
| `nickolas/shopping_agent/agent.py::_respond_custom()` | Later, one injected dense-query transformation call when existing routing invokes dense retrieval | End-to-end integration without changing routing |
| `Agent.__init__()` | Later accept an optional integration callable/config, default disabled | Preserve exact M0 behavior |
| Experiment config artifact | Store `memory_mode`, `local_k`, rank, steering config, universe and depths | One auditable configuration source |

Do not modify `dense_retrieve_vector()`, product embeddings, FTS, reranking, or the old public evaluator.

## Q. Freeze manifest

### M0

Freeze:

- `agent.py`.
- `agent_openai.py`.
- `embedding_backends.py`.
- `run_m0.py`.
- `configs/m0_openai.json`.
- Dense vector-interface tests.
- State-routing/parity tests.
- `_state_to_retrieval_query()` golden behavior.
- Product-text version and fingerprint.
- Catalogue content fingerprint.
- Exact catalogue ID row-order fingerprint.
- Embedding cache metadata:
  - schema version;
  - backend/model/space;
  - dimensions;
  - normalized flag;
  - product-text fingerprint;
  - catalogue fingerprint.
- Hybrid routing thresholds.
- Dense Top-N.
- Reranking/diversification behavior.
- Evaluator version for end-to-end results.

Do not hash the generated embedding cache file itself.

The existing `nickolas/shopping_agent/baseline_results/m0_openai/FREEZE.md` is stale relative to current Checkpoint A code. For example, its recorded hashes for `agent.py`, `run_m0.py`, and `m0_openai.json` differ from the current sources. Refresh it before actual runs.

### QLMP

Freeze independently:

- `__init__.py`.
- `models.py`.
- `config.py`.
- `diagnostics.py`.
- `geometry.py`.
- `projection.py`.
- `aggregation.py`.
- `baselines.py`.
- `steering.py`.
- `synthetic.py`.
- All Phase 1/2 tests.
- README formulas and semantics.
- Default config values, clearly marked untuned.

After projector isolation, freeze the selected B3 scoring rule separately rather than retroactively changing the Phase 2 freeze.

## R. Existing longitudinal infrastructure

### Reusable

Existing infrastructure already provides:

- Stable explicit `user_id`.
- Ordered `sequence_index`.
- Per-user memory isolation.
- Chronology guards.
- Snapshot export/import preserving real embeddings and embedding-space IDs.
- Filtering snapshots by user/session prefix.
- Shadow memory visibility.
- Fixed-transcript shadow/no-history parity.
- Private shopper directives.
- Removal of evaluator-only fields from agent input.
- Target-leakage checks.
- Outcome/target exclusion from memory creation.
- Real catalogue targets.
- Four users × ten sessions.
- U3 history-prefix replay.
- Existing QLMP-compatible `MemoryItem` persistence.

Useful sources include:

- `nickolas/shopping_agent/run_longitudinal_eval.py`
- `nickolas/shopping_agent/memory_store.py`
- `nickolas/shopping_agent/longitudinal_eval/directives.py`
- `nickolas/shopping_agent/longitudinal_eval/users_40.json`

### Still missing

The current infrastructure does not provide:

- A frozen `DenseQuerySnapshot` per evaluated state.
- Exact q reuse across M0/B1/B2/B3.
- Frozen full current-state replay at the dense boundary.
- A memory-mode runner.
- Per-`MemoryItem` projector labels in the required taxonomy.
- Projector-isolation AUROC/AUPRC.
- Dense Recall@10/50 and exact component rank.
- Browsing cases; the current 40-session design uses buying rows.
- Exact `0/2/5/10/20/50` irrelevant-memory sweeps.
- B3 local-neighbourhood capture.
- B1/B2/B3 retrieval comparisons.

Fixed transcripts eliminate shopper-message drift, but full-path state distillation and current-query embedding can still be rerun. Thus fixed transcript alone does not satisfy exact-q reuse.

### Shadow transition

M0 remains:

```text
history may be visible/observed
historical_memory_applied = False
```

B1/B2/B3 must receive the candidate tuple only through an explicit integration call. User/session IDs alone must never activate memory ranking.

## S. Hard invariants

1. Canonical q is computed exactly once per evaluated state.
2. Target ID/title never contributes to current state, query text, q, memory, selection, or ranking.
3. Historical relevance labels never reach M0, QLMP, or the integration adapter.
4. Query, products, and memories share the exact embedding-space identity.
5. `q_star` is ephemeral and never becomes historical memory.
6. QLMP never bypasses or reinterprets M0 current-state constraint enforcement.
7. B1, B2, and B3 use the same aggregation primitives and bounded-steering implementation.
8. Every final vector is scored by unchanged `M0.dense_retrieve_vector()`.
9. `memory_mode="none"` and every no-memory fallback reproduce canonical M0 dense behavior exactly.
10. User, session, current state, candidate history/order, catalogue, scorer, target, evaluator logic, and downstream pipeline are identical across modes.

Additionally, the simulator profile and Buying/Browsing label remain evaluator-private.

## T. Genuine blockers

There are no architectural blockers to implementing the adapter and projector-isolation evaluation.

Mandatory minor issues before scientific runs:

- Normalize float64 QLMP working copies instead of passing raw promoted float32 vectors to the strict `1e-8` checks.
- Ensure future memory creation stores normalized float64 embeddings.
- Refresh the stale M0 freeze manifest.
- Explicitly record the candidate-universe choice.
- Do not treat current brand handling as a hard constraint.
- Do not enable B3 steering before the projector stop/go result.

`local_k`, rank, angular limit, memory Top-K, projection threshold, and candidate-universe choice remain experimental parameters, not blockers.

## U. Recommendation

Yes.

The contract is sufficiently clear and scientifically fair for the next task to implement the M0 ↔ QLMP orchestration boundary and projector-isolation evaluation without redesigning either system.

The next task should not immediately activate full B3 retrieval steering. It should:

1. Implement the numeric/ownership adapter and exact-q replay.
2. Establish `none`/B1/B2 parity tests.
3. Run projector isolation on real M0 vectors.
4. Apply the stop/go rule.
5. Only then add B3 selection and, later, the single end-to-end dense-route hook.
