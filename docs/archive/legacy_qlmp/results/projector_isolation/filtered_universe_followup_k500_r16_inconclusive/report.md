# A. Current-state audit

Current HEAD: `956f0599accc2c40038632e94392770ff0466f03`. Authoritative parent: `preregistered_full_k500_r16_definitive` (manifest SHA-256 `1504db2b0ee5b5d98ca1bda2a92e7ad80cc5210cb4da77ebc7f29d6fb274a151`). Its verdict remains `PROJECTOR STOP`.

The frozen Phase 3A fixture, vector snapshot, catalogue, product-text fingerprint, exact q keys, ordered memories, and B0 final-state snapshot were reused. No shopper/state extraction or embedding was rerun.

# B. Files changed

Diagnostic-only preflight module, deterministic tests, and this run-specific result directory. No M0, QLMP geometry, evaluator, or `experiment_1` implementation was changed.

# C. Included filtered fixtures

Eligible queries: 8; users: 3; memory pairs: 48; primary positives: 4; primary negatives: 37; same-category-labelled hard negatives: 28.

| Fixture | User | price_max | Eligible catalogue | Qualification |
|---|---|---:|---:|---|
| u1_stable_s5_final | u1_stable | 120.00 | 9794 | deterministic current M0 price cap changes universe |
| u1_stable_s6_final | u1_stable | 120.00 | 9794 | deterministic current M0 price cap changes universe |
| u2_override_s4_final | u2_override | 120.00 | 9794 | deterministic current M0 price cap changes universe |
| u2_override_s7_final | u2_override | 120.00 | 9794 | deterministic current M0 price cap changes universe |
| u2_override_s9_final | u2_override | 120.00 | 9794 | deterministic current M0 price cap changes universe |
| u3_distractor_s3_final | u3_distractor | 20.00 | 4792 | deterministic current M0 price cap changes universe |
| u3_distractor_s5_final | u3_distractor | 120.00 | 9794 | deterministic current M0 price cap changes universe |
| u3_distractor_s6_final | u3_distractor | 120.00 | 9794 | deterministic current M0 price cap changes universe |

Identity fixtures recorded separately: 10. They have no current M0 hard condition and provide no evidence for this hypothesis: u1_stable_s8_final, u1_stable_s9_final, u2_override_s2_final, u3_distractor_s7_final, u3_distractor_s8_final, u3_distractor_s9_final, u4_negative_s2_final, u4_negative_s5_final, u4_negative_s8_final, u4_negative_s9_final.

The preflight failed before scientific execution: scientific execution stopped before projection: query_count=8 < 12, positive_count=4 < 8, same_category_hard_negative_type_count=3 < 8.

# D. M0 hard-filter semantics

| Constraint | M0 current treatment | Included in diagnostic hard mask? |
|---|---|---|
| price_max | hard pre-retrieval mask when < 9999; missing/unparseable price is 9999 and excluded | yes |
| department | soft +20 scoring boost | no |
| category | soft +15 scoring boost | no |
| brand/style/colour/disclosed slots | soft lexical/constraint scoring | no |
| negated terms / seen products / diversity | post-retrieval exclusion or diversification, not the M0 hard candidate mask | no |

Missing/unparseable catalogue price is stored as `9999.0`; therefore a real `price_max < 9999.0` excludes it. This exactly matches current M0 and the prior U2 diagnostic.

# E. Candidate-universe comparison

| Fixture | Full Top-K | Eligible catalogue | Filtered K | Original Top-500 incompatible | Overlap |
|---|---:|---:|---|---:|---|
| u1_stable_s5_final | 500 | 9794 | not constructed | 93.2% | not computed (preflight stop) |
| u1_stable_s6_final | 500 | 9794 | not constructed | 90.8% | not computed (preflight stop) |
| u2_override_s4_final | 500 | 9794 | not constructed | 92.6% | not computed (preflight stop) |
| u2_override_s7_final | 500 | 9794 | not constructed | 81.6% | not computed (preflight stop) |
| u2_override_s9_final | 500 | 9794 | not constructed | 85.4% | not computed (preflight stop) |
| u3_distractor_s3_final | 500 | 4792 | not constructed | 88.0% | not computed (preflight stop) |
| u3_distractor_s5_final | 500 | 9794 | not constructed | 90.0% | not computed (preflight stop) |
| u3_distractor_s6_final | 500 | 9794 | not constructed | 84.2% | not computed (preflight stop) |

# F. Subspace comparison

Not computed. Singular spectra, effective ranks, and principal angles would be outcome data from an underpowered arm, so execution stopped at the preregistered sample gate.

# G. Pair-level projector comparison

Not computed. `paired_results.csv` contains the frozen schema and zero rows; `paired_results.jsonl` is empty.

# H. Primary paired metrics

Not computed. The identical-subset raw/full/filtered AUROC and AUPRC blocks are explicitly null in `summary.json`.

# I. Hard-negative metrics

Not computed. Eligible preflight counts by evaluator-private hard-negative type are: contextual_requirement=12, easy_distractor=9, nearby_non_portable=7, override_conflict=6, same_category=3. These labels were joined only for the sample audit, after label-blind fixture eligibility.

# J. U2 case study

U2 remains the motivating case: budget under 120, bright/colourful current intent versus black/minimal/understated history. The prior diagnostic found 73/500 compliant, 85.4% incompatible, 14.6% neighbourhood overlap, and a materially rotated rank-16 subspace. Full-arm values remain authoritative; filtered rho/projected norms were not computed after the sample gate failed.

# K. U3 portability control

`u3_distractor_s9_final` has no current M0 hard condition, so filtering is identity and it is excluded. Its prior formal/dressy contextual memory remains at rho approximately 0.152. Candidate-universe filtering does not address U3's demonstrated portability failure.

# L. Scientific limitations

Only 8 eligible queries, 3 users, 4 positives, and 3 negatives with the strict `same_category` subtype remain. Budget is a structured filter, not an embedded feature; filtering cannot be described as QLMP understanding budget. Representation, selected-variance, and portability limitations remain unresolved.

# M. Candidate-universe verdict

`FILTERED-UNIVERSE FOLLOW-UP INCONCLUSIVE`

# N. Original projector verdict

`PROJECTOR STOP`

# O. Next-study decision

`NEW FILTERED PROJECTOR STUDY INCONCLUSIVE`

# P. Tests

Combined QLMP core plus full shopping-agent suite: 201 tests passed, 0 failed, with 115 subtests passed and 2 third-party deprecation warnings. The focused follow-up file contributed 9 passing tests. The run manifest records zero LLM and zero OpenAI calls.

# Q. Scope audit

Confirmed by the run manifest: QLMP geometry unchanged; B1/B2 unchanged; M0 ranking/routing/dense scorer unchanged; product text/model unchanged; `experiment_1` unchanged; official evaluator unchanged; Graphify not run; no commit; B3 not implemented; q-star not constructed; K=500 and rank=16 unchanged.
