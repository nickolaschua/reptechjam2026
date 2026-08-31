# Relevant-Memory Retrieval Steering

## Motivation

On the frozen 18-session Buyer fixture, averaging every eligible historical memory produced negative transfer: large M1 MRR was below large M0 MRR, and small M1 MRR was below small M0 MRR. M3 coordinate masking reduced some of that harm but did not outperform M0 in either embedding space. These statements apply only to this curated, previously inspected fixture.

## New hypothesis

> Before combining historical memory with current intent, first retrieve only the historical memories semantically relevant to the present Buyer query.

## Algorithm

For normalized current-intent embedding `q` and normalized, strictly prior eligible memory embeddings `e_i`:

```text
s_i = q^T e_i

I = TopK(s_1, ..., s_n)

m_top = normalize(mean_{i in I}(e_i))

q* = normalize(q + lambda * m_top)
```

The primary experiment is locked at:

```text
K = 3
lambda = 0.20
```

Selection uses descending cosine similarity. Ties are resolved deterministically by earlier origin sequence, stable memory ID, and original candidate order. If fewer than three memories are eligible, all are used. With zero eligible memories, `q* = q`. There is no threshold and selected memories receive equal weight.

## Architecture

```text
Historical Memories
 e1  e2  e3 ... en
  \   |   |      /
   \  |   |     /
    cosine with q
          |
          v
       Top K=3
          |
          v
        Mean
          |
          v
   Relevant Memory m
          |
          | lambda = 0.20
          v
Current q -------> q*
                    |
                    v
             Existing Retriever
```

## Difference from M1

M1:

```text
aggregate everything -> steer
```

M4:

```text
retrieve relevant memories -> aggregate -> steer
```

## Difference from M3

M3:

```text
aggregate history -> coordinate mask
```

M4:

```text
whole-memory semantic retrieval -> aggregate
```

M4 does not compute `abs(q * m)` and does not apply a coordinate mask.

## Evaluation controls

M4 is evaluator-private and Buyer-only. It reuses the exact frozen M0 queries, candidate memory IDs and texts, temporal eligibility, targets, catalogue text and ordering, per-space memory vectors, and canonical catalogue dot-product scorer. Large and small are evaluated independently; embeddings are never mixed. The same logical candidates must be present in each space, although their selected top-three sets may differ.

## Limitations

- Only 18 Buyer sessions.
- The projector fixture is curated and was previously inspected.
- Only positive/neutral memories are eligible under the frozen policy.
- Selected memories use an equal-weight mean.
- `K` is fixed at 3.
- There is no similarity threshold.
- There is no learned memory router.
- Selected memories are not automatically proven causally useful.

## Falsification

The hypothesis is not supported if:

- M4 does not outperform M0.
- M4 merely reproduces M1's negative transfer.
- M4 consistently underperforms M3.
- Semantic top-K memories look plausible but do not improve ranking.
- Performance differs arbitrarily across embedding spaces.

If M4 does not beat M0, the result is frozen without tuning `K`, weighting, thresholds, recency, gates, projections, or other vector operations. The next scientific question is:

> Are the stored MemoryItems themselves predictive enough to improve recommendation ranking?
