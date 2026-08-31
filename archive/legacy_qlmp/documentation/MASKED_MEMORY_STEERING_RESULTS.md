# Masked Memory Steering Results

# Experiment

Locked evaluator-private Buyer experiment on `nickolas\shopping_agent\longitudinal_eval\projector_fixture_v1.json` with `keep_ratio = 0.20`, `lambda_memory = 0.20`, equal-weight normalized eligible positive/neutral memory, the unchanged full-catalogue dot-product scorer, and exact target ranks over 50000 products. All 18 sessions are scored under M0/M1/M2/M3 without tuning.

The fixture uses the pre-existing curated projector memory set (109 session-memory pairs, 44 unique memory texts), not every memory committed in the 40-session source replay. Its fixture, baseline rankings, and projector results were previously inspected, so this is reproducible diagnostic evidence rather than clean held-out evidence.

# Embedding spaces

`text-embedding-3-large` and `text-embedding-3-small` are validated as independent spaces. Query, memory, and catalogue operands must share the exact embedding-space identifier and provider-returned dimension. Cross-model operations fail before scoring. Product text and row ordering are identical.

# Primary results

MRR:

| Embedding | M0 | M1 | M2 | M3 |
| --- | ---: | ---: | ---: | ---: |
| text-embedding-3-large | 0.239863 | 0.229643 | 0.238440 | 0.236963 |
| text-embedding-3-small | 0.233505 | 0.228987 | 0.233432 | 0.233082 |

HR@10:

| Embedding | M0 | M1 | M2 | M3 |
| --- | ---: | ---: | ---: | ---: |
| text-embedding-3-large | 0.388889 | 0.333333 | 0.388889 | 0.333333 |
| text-embedding-3-small | 0.333333 | 0.333333 | 0.333333 | 0.333333 |

Mean target rank (lower is better):

| Embedding | M0 | M1 | M2 | M3 |
| --- | ---: | ---: | ---: | ---: |
| text-embedding-3-large | 104.167 | 136.722 | 110.722 | 121.556 |
| text-embedding-3-small | 333.778 | 472.278 | 384.333 | 405.333 |

Median target rank (lower is better):

| Embedding | M0 | M1 | M2 | M3 |
| --- | ---: | ---: | ---: | ---: |
| text-embedding-3-large | 34.000 | 49.500 | 36.000 | 39.500 |
| text-embedding-3-small | 39.000 | 58.500 | 40.500 | 45.500 |

# Pairwise deltas

Positive mean rank change means the compared method moved the target upward relative to the reference.

## text-embedding-3-large

| Pair | Improved | Unchanged | Regressed | Mean rank change* | MRR change |
| --- | ---: | ---: | ---: | ---: | ---: |
| M1-M0 | 2 | 4 | 12 | -32.556 | -0.010219 |
| M2-M0 | 2 | 6 | 10 | -6.556 | -0.001423 |
| M3-M0 | 0 | 7 | 11 | -17.389 | -0.002900 |
| M3-M1 | 12 | 4 | 2 | 15.167 | 0.007320 |
| M3-M2 | 0 | 6 | 12 | -10.833 | -0.001477 |

## text-embedding-3-small

| Pair | Improved | Unchanged | Regressed | Mean rank change* | MRR change |
| --- | ---: | ---: | ---: | ---: | ---: |
| M1-M0 | 3 | 5 | 10 | -138.500 | -0.004518 |
| M2-M0 | 3 | 6 | 9 | -50.556 | -0.000074 |
| M3-M0 | 2 | 6 | 10 | -71.556 | -0.000424 |
| M3-M1 | 9 | 6 | 3 | 66.944 | 0.004095 |
| M3-M2 | 3 | 6 | 9 | -21.000 | -0.000350 |

# Session-level behaviour

- Large masking rescues: 2; destroyed-useful-memory cases: 2.
- Small masking rescues: 1; destroyed-useful-memory cases: 3.
- Session-level rows, exact ranks, diagnostics, and top-10 scores are in `session_results.jsonl`.

# Large vs small

Absolute M0 baseline MRR is 0.239863 for large and 0.233505 for small. This is retrieval-backend quality, not memory uplift.

M3 minus M0 MRR uplift is -0.002900 for large and -0.000424 for small. This within-space comparison is the relevant cross-model test of the memory mechanism.

# Sensitivity

Not run. With 18 previously inspected sessions, a keep-ratio sweep would be weak sensitivity evidence and is intentionally deferred; 0.20 remains the locked primary value.

# Interpretation

M3 did not improve MRR over M0 in either frozen embedding space; this run does not support a retrieval benefit. The paired bootstrap intervals are descriptive only; no significance claim is made for this small, previously inspected fixture.

# Limitations

- Dense coordinates are distributed and are not human-interpretable features.
- Coordinate-wise masking is a heuristic.
- Buyer-only evaluation with four users and a limited session sample.
- The frozen projector fixture contains a curated historical-memory subset rather than all committed prior memories.
- Equal-weight aggregate memory and no negative-memory steering.
- Fixed lambda and primary keep ratio.
- Results may depend on embedding model.
- This fixture and related projector/M0 artifacts were inspected during earlier method work; future tuning and evaluation partitions must remain separated.

# Conclusion

M3 did not improve MRR over M0 in either frozen embedding space; this run does not support a retrieval benefit.
