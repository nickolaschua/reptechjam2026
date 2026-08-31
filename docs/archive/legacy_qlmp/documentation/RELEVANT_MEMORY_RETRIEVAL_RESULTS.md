# Relevant-Memory Retrieval Results

## Frozen experiment

This cache-only run reused the exact 18-session Buyer fixture and the unchanged frozen M0/M1/M2/M3 artifacts. M4 used `K = 3`, `lambda_memory = 0.20`, equal-weight selected-memory aggregation, no threshold, and the canonical full-catalogue scorer. No embeddings were generated.

Positive rank Δ means M4 moved the target upward (reference rank minus M4 rank).

## Primary metrics

MRR:

| Embedding | M0 | M1 | M3 | M4 |
| --- | ---: | ---: | ---: | ---: |
| text-embedding-3-large | 0.239863 | 0.229643 | 0.236963 | 0.227878 |
| text-embedding-3-small | 0.233505 | 0.228987 | 0.233082 | 0.228847 |

HR@10:

| Embedding | M0 | M1 | M3 | M4 |
| --- | ---: | ---: | ---: | ---: |
| text-embedding-3-large | 0.388889 | 0.333333 | 0.333333 | 0.333333 |
| text-embedding-3-small | 0.333333 | 0.333333 | 0.333333 | 0.333333 |

Mean target rank (lower is better):

| Embedding | M0 | M1 | M3 | M4 |
| --- | ---: | ---: | ---: | ---: |
| text-embedding-3-large | 104.167 | 136.722 | 121.556 | 139.111 |
| text-embedding-3-small | 333.778 | 472.278 | 405.333 | 533.056 |

Median target rank (lower is better):

| Embedding | M0 | M1 | M3 | M4 |
| --- | ---: | ---: | ---: | ---: |
| text-embedding-3-large | 34.000 | 49.500 | 39.500 | 50.000 |
| text-embedding-3-small | 39.000 | 58.500 | 45.500 | 60.000 |

The prior M0/M1/M2/M3 values above are copied unchanged from the hash-locked result artifact. The complete M0–M4 table remains in `summary.json`.

### Complete M0–M4 table

| Embedding | Method | MRR | HR@10 | Mean Rank | Median Rank |
| --- | --- | ---: | ---: | ---: | ---: |
| text-embedding-3-large | M0 | 0.239863 | 0.388889 | 104.167 | 34.000 |
| text-embedding-3-large | M1 | 0.229643 | 0.333333 | 136.722 | 49.500 |
| text-embedding-3-large | M2 | 0.238440 | 0.388889 | 110.722 | 36.000 |
| text-embedding-3-large | M3 | 0.236963 | 0.333333 | 121.556 | 39.500 |
| text-embedding-3-large | M4 | 0.227878 | 0.333333 | 139.111 | 50.000 |
| text-embedding-3-small | M0 | 0.233505 | 0.333333 | 333.778 | 39.000 |
| text-embedding-3-small | M1 | 0.228987 | 0.333333 | 472.278 | 58.500 |
| text-embedding-3-small | M2 | 0.233432 | 0.333333 | 384.333 | 40.500 |
| text-embedding-3-small | M3 | 0.233082 | 0.333333 | 405.333 | 45.500 |
| text-embedding-3-small | M4 | 0.228847 | 0.333333 | 533.056 | 60.000 |

## Pairwise results

| Pair | Improved | Same | Regressed | Mean Rank Δ | Median Rank Δ | ΔMRR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Large M4-M0 | 1 | 4 | 13 | -34.944 | -6.500 | -0.011985 |
| Large M4-M1 | 3 | 8 | 7 | -2.389 | 0.000 | -0.001766 |
| Large M4-M3 | 1 | 4 | 13 | -17.556 | -3.000 | -0.009085 |
| Small M4-M0 | 3 | 5 | 10 | -199.278 | -1.500 | -0.004658 |
| Small M4-M1 | 1 | 9 | 8 | -60.778 | 0.000 | -0.000140 |
| Small M4-M3 | 2 | 6 | 10 | -127.722 | -6.500 | -0.004234 |

## Primary questions

- **Q1 — M4 vs M0:** No. M4 MRR is lower in both spaces (-0.011985 large; -0.004658 small), with worse mean and median target rank.
- **Q2 — M4 vs M1:** No at aggregate level. M4 MRR is lower than M1 by -0.001766 large and -0.000140 small; top-three selection did not avoid the negative transfer.
- **Q3 — M4 vs M3:** No. M4 MRR is lower than M3 by -0.009085 large and -0.004234 small.
- **Q4 — cross-space behavior:** The beneficial hypothesis did not generalize. M4 minus M0 is negative in both spaces, while the selected top-three IDs differ in 13 of 18 sessions.

## Negative-transfer diagnostics

- Large: relevant-memory rescues 0; partial rescues 3; M4 improvements over M0 1; destroyed useful-memory cases 1; M4 beats M3 in 1 sessions (mean rank Δ -17.556).
- Small: relevant-memory rescues 0; partial rescues 1; M4 improvements over M0 3; destroyed useful-memory cases 2; M4 beats M3 in 2 sessions (mean rank Δ -127.722).
- Large M4-over-M0 sessions: u4_negative_s8.
- Small M4-over-M0 sessions: u2_override_s7, u3_distractor_s5, u4_negative_s8.
- Large partial rescues: u2_override_s2, u2_override_s4, u3_distractor_s5; small partial rescues: u2_override_s4.

## Representative selections

These examples are qualitative diagnostics only. Plausible text is not evidence of correctness; target rank is the objective result.

| Current Buyer Query | Top Selected Memories (large) | M0 Rank | M4 Rank |
| --- | --- | ---: | ---: |
| shoes low-top synthetic rubber trail running | style: yoga sling 2<br>fit: breathable<br>fit: natural | 124 | 183 |
| clothing cinch zipper grey durable cording gym | closure: double zippers<br>pattern: mesh top<br>material: leather | 22 | 30 |
| clothing short sleeve rayon spandex pull-on women | style: dressy<br>style: flannel<br>style: hooded | 1 | 1 |
| slippers clothing 80% cotton | material: cotton<br>material: 95% cotton, 5% spandex<br>use_case: nightgown | 6 | 7 |
| clothing black distressed red hot chili peppers medium men | color: black<br>material: soft leather<br>style: minimalistic | 1 | 1 |

Across the complete session artifact, selections often reflect surface attribute overlap (for example cotton with cotton memories, or zipper with double zippers), but several omit central intent attributes or retrieve only broadly related style/material memories. This is a qualitative observation, not a correctness or causality label, and the aggregate ranking result remains negative.

The large and small models selected different top-three memory ID sets in 13 of 18 sessions: u1_stable_s5, u1_stable_s6, u1_stable_s9, u2_override_s2, u2_override_s4, u2_override_s7, u2_override_s9, u3_distractor_s3, u3_distractor_s5, u3_distractor_s6, u3_distractor_s7, u3_distractor_s9, u4_negative_s9.

## Paired bootstrap: M4 minus M0 MRR

- Large: ΔMRR -0.011985; descriptive 95% percentile CI [-0.022416857146040147, -0.00429104135477873].
- Small: ΔMRR -0.004658; descriptive 95% percentile CI [-0.01563754787968618, 0.0030988389338308274].
- Seed `20260830`, samples `10000`. These intervals are descriptive only; the fixture is small and previously inspected, so no significance claim is made.

## Artifacts

- `results/relevant_memory_retrieval/session_results.jsonl`: all 36 model-session diagnostics, including every candidate score and selected text.
- `results/relevant_memory_retrieval/fixture_audit.json`: per-session eligibility, logical parity, and temporal-isolation audit.
- `results/relevant_memory_retrieval/summary.json`: frozen M0–M3 metrics plus M4 metrics, pairwise comparisons, rescue categories, and bootstrap intervals.
- `results/relevant_memory_retrieval/run_manifest.json`: hashes and locked run parameters.

## Interpretation

M4 did not beat M0 in either frozen space. The hypothesis is not supported on this fixture, and no K or algorithm tuning was performed. The next scientific question is whether the stored MemoryItems themselves are predictive enough to improve recommendation ranking.
