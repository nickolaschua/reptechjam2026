# QLMP Phases 1-2: isolated geometry and bounded memory steering

This package is a standalone, NumPy-only implementation of the geometric core
of **query-local memory projection (QLMP)** and the pre-Checkpoint-A memory
selection, aggregation, and bounded-steering mechanics. Phase 1 asks a narrow
diagnostic question: how much of a memory's direction away from the current
query is supported by the tangent directions seen in a local product
neighbourhood? Phase 2 adds correct competing B1/B2 mechanics without using a
real shopping retriever.

## Geometry and terminology

Let the normalized current-query embedding be \(q\), and let normalized local
product embeddings be \(p_1, \ldots, p_K\).  QLMP removes each product's
query-parallel component directly:

\[
r_i = p_i - (q^\top p_i)q.
\]

The residual rows form \(R \in \mathbb{R}^{K \times d}\).  With the thin SVD

\[
R = U\Sigma V^\top,
\]

the retained columns of \(V\) form a tangent basis
\(B \in \mathbb{R}^{d \times r_{\mathrm{eff}}}\).  The requested rank is capped
by the conventional float64 numerical rank.  In exact terms, with
\(\sigma_{\max} = \sigma_1\) and \(\epsilon_{64}\) denoting float64 machine
epsilon,

\[
\tau_{\mathrm{rank}}
  = \sigma_{\max}\max(K,d)\epsilon_{64}, \qquad
r_{\mathrm{num}}
  = \#\{j : \sigma_j > \tau_{\mathrm{rank}}\}, \qquad
r_{\mathrm{eff}} = \min(r_{\mathrm{requested}}, r_{\mathrm{num}}).
\]

The comparison is strict.  This rank calculation is independent of the
application-level `epsilon`; that value remains limited to vector/unit
validation and the projection diagnostic.  A fully degenerate neighbourhood
therefore has a valid zero-column basis, and QLMP never invents unsupported
directions.

For a normalized memory embedding \(m\), the query-tangent residual and its
local projection are

\[
t_m = m - (q^\top m)q, \qquad \hat t_m = BB^\top t_m.
\]

The Phase 1 support diagnostic is

\[
f = \frac{\|\hat t_m\|_2^2}{\|t_m\|_2^2 + \epsilon}.
\]

The result also exposes the raw query-memory cosine, tangent norm, projected
norm, basis coefficients, and both residual vectors so the calculation remains
auditable.

The denominator intentionally makes the diagnostic conservative for very small
tangent residuals.  Even when such a residual lies entirely in the local
subspace, its fraction can be far below one because \(\epsilon\) is retained in
the denominator.  This avoids treating a nearly query-parallel memory as strong
directional support.  The formula itself is unchanged.

## Data and basis contracts

`MemoryItem` is a frozen record with these fields:

- `id`: required non-empty string;
- `text`: required non-empty string;
- `embedding`: required finite one-dimensional vector, copied into owned,
  read-only float64 storage;
- `source`: required `MemorySource` value (`USER`, `ASSISTANT`, `SYSTEM`,
  `EXPLICIT_PREFERENCE`, `PURCHASE_EPISODE`, `BEHAVIORAL_INFERENCE`, `CLICK`,
  or `RECOMMENDATION_SHOWN`);
- `polarity`: required `POSITIVE`, `NEGATIVE`, or `NEUTRAL` value;
- `scope`: optional non-empty string, defaulting to `None`;
- `timestamp`: optional `datetime`, defaulting to `None` and preserved without
  recency weighting or other temporal logic; and
- `confidence`: finite number in `[0, 1]`, defaulting to `1.0`; booleans are
  rejected.

Every non-empty basis follows one shared strict float64 contract.  For embedding
dimension \(d\), basis rank \(r\), and

\[
\tau_B = 32\max(1,d,r)\epsilon_{64},
\]

QLMP requires both

\[
\lVert B^\top B-I\rVert_2 \leq \tau_B
\quad\text{and}\quad
\lVert q^\top B\rVert_2 \leq \tau_B.
\]

`LocalSubspace` enforces the orthonormal part when constructed, and projection
enforces both orthonormality and query tangency.  A projection fraction above
one is clamped only when its overshoot is no larger than
\(2\tau_B + \tau_B^2 + 32\epsilon_{64}\); a larger overshoot is an invariant
failure rather than accepted silently.

"Query-centred" is deliberate terminology.  `build_local_subspace` does **not**
subtract the ordinary neighbourhood mean.  Mean-centering is reserved as a
future controlled ablation, not an interchangeable preprocessing step.

## Phase 1 boundary

This package establishes implementation correctness on deterministic synthetic
geometry only.  The default rank and epsilon are engineering starting points
and are explicitly **not tuned**.  The fixture does not establish real-catalogue
validity, retrieval quality, or causal benefit.

Phase 1 does not establish that query-local projection outperforms ordinary query-memory cosine similarity; that comparison is deferred until post-Checkpoint-A scientific evaluation.

## Phase 2 baselines and common steering

Phase 2 represents every historical memory as an additional tangent direction
relative to the normalized current query:

\[
u_i = m_i - (q^\top m_i)q.
\]

It does not replace current intent with a user-profile vector.

### B1: naive memory fusion

B1 includes every structurally eligible positive or neutral memory and takes a
uniform mean of its raw tangent residuals. Raw cosine is diagnostic only: it
does not affect B1 selection or weighting. If no memory is eligible, the
aggregate is zero and steering leaves the query unchanged.

### B2: cosine-gated memory

B2 first applies the same structural eligibility policy, then ranks memories
by descending raw cosine similarity to the current query. Ties preserve
original input order. An optional threshold is inclusive (`similarity >=
threshold`), and the first `memory_top_k` survivors are selected. Selected
scores use the transparent non-negative policy

\[
w_i = \max(\operatorname{cosine}(q,m_i), 0),
\]

followed by sum-to-one normalization. If every selected score is non-positive,
all weights are zero. B2 aggregates raw tangent residuals and never uses the
Phase 1 projector.

### Eligibility and structured overrides

Two present scopes must match exactly and case-sensitively. An absent query
scope imposes no scope restriction. An absent memory scope is eligible by
default and can be disabled by configuration. Negative-polarity historical
memories are excluded rather than subtracted as dense vectors.

Explicit current-turn constraints and negatives such as "No black this time",
"Under $150", and "Nike only" remain outside QLMP vector arithmetic. They
belong to the future B0/current-state hard-filtering layer and cannot be
overridden by historical steering.

### Common aggregation and angular bound

B1 and B2 use the same non-negative weight normalization, residual aggregation,
and bounded query construction. Future B3 will pass already-projected residuals
through those same aggregation and steering functions, allowing selection and
raw/projected residual choices to be ablated independently.

For aggregate steering \(\Delta\), the steering implementation first removes
any query-parallel component, then considers

\[
q_{\mathrm{candidate}} = q + \beta\Delta.
\]

For a unit query and tangent \(\Delta\), the requested angle is
\(\arctan(\beta\lVert\Delta\rVert)\). The applied tangent magnitude is capped at
\(\tan(\theta_{\max})\), `q_candidate` is normalized, and the actual final angle
is independently calculated from the clipped query/query-star dot product.
Clipping and tangency correction are explicit diagnostics. The default
`max_shift_deg=10` is an untuned engineering starting value, not a validated
optimal value.

## Scientific and integration boundary

Phase 2 establishes implementation correctness only. It does not show B1, B2, or QLMP improve real product retrieval.

Checkpoint A remains deferred. This package contains no B0 integration, API
calls, catalogue retrieval, persistence, automatic memory writes,
real-catalogue experiments, empirical nulls, AUROC/AUPRC evaluation, retrieval
distractor sweeps, Buying/Browsing adaptation, or temporal-memory behaviour.

The controlled language fixtures remain text-only. They do not provide
fabricated embeddings or imply a retrieval result.

Run the isolated offline checks from the repository root with:

```powershell
$env:PYTHONDONTWRITEBYTECODE = '1'
python -m unittest discover -s nickolas/memory/qlmp/tests -t . -v
```
