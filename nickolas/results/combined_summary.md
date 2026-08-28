# Combined experiment summary

> Oracle diagnostics (1, 3, 4, 5) inspect reconstructed hidden labels only for dataset analysis. Agent-realistic claims (2, 6, 7, 8, 9, 10, 11) use only information disclosed at each turn. Experiment 11 is retrospective because its public set was inspected before evaluation.

| Experiment | Mode | Central result |
|---|---|---|
| 1. Constraint uniqueness | Oracle | Four constraints leave median 1.0 exact-phrase candidates; uniqueness 63.5%. |
| 2. Target-rank curves | Agent-realistic | exact_phrase leads: score 0.816917, Hit@10 93.0%, MRR 0.623. |
| 3. Field signal | Oracle | features has the highest exact coverage (93.9%); 62 constraints overlap fields. |
| 4. Classification | Oracle | `other` covers all 800 constraints; 19 strings trigger classifier precedence. |
| 5. Candidate shrinkage | Oracle | Exact candidates shrink to median 1.0 at four constraints; soft sets remain deliberately broader. |
| 6. Slate widths | Agent-realistic | Held-out Top-10 scores 0.524651 vs 0.409286 adaptive; full width preserves more recall. |
| 7. Residual failures | Agent-realistic + oracle-after-freeze | Calibration selected exact_stateful_bm25_rrf; held-out score 0.846109 vs exact 0.831404, with 4 rescues/2 regressions. Recommend exact_stateful_bm25_rrf. |
| 8. Intent-routed dense browsing | Agent-realistic | Held-out routed score 0.710690 vs Experiment 7 0.846109; routing accuracy 100.0%. |
| 9. Adaptive hybrid | Agent-realistic | Calibration selected structured_state_identity_fixed_other; held-out score 0.846109; promote=False. |
| 10. XTR/WARP retrieval | Agent-realistic + oracle-after-freeze | Held-out scores: exact 0.831404, Experiment 7 BM25 0.846109, WARP 0.846892; recommend exact_stateful_xtr_warp_rrf. |
| 11. Clean FTS5 candidate | Retrospective agent evaluation | Selected clean_specific_query_pagination; full score 0.894362, evaluation-partition score 0.896425. Private validation required. |

- [1. Constraint Uniqueness](experiment_01_constraint_uniqueness/summary.md) (21.793 seconds)
- [2. Target Rank Curves](experiment_02_target_rank_curves/summary.md) (189.911 seconds)
- [3. Field Signal](experiment_03_field_signal/summary.md) (140.793 seconds)
- [4. Constraint Classification](experiment_04_constraint_classification/summary.md) (0.297 seconds)
- [5. Candidate Set Shrinkage](experiment_05_candidate_set_shrinkage/summary.md) (38.355 seconds)
- [6. Slate Width Counterfactuals](experiment_06_slate_width_counterfactuals/summary.md) (3.142 seconds)
- [7. Residual Failure Analysis](experiment_07_residual_failure_analysis/summary.md) (173.276 seconds)
- [8. Intent Routed Dense Browsing](experiment_08_intent_routed_dense_browsing/summary.md) (415.536 seconds)
- [9. Adaptive Hybrid Architecture](experiment_09_adaptive_hybrid_architecture/summary.md) (208.667 seconds)
- [10. Xtr Warp Retrieval](experiment_10_xtr_warp_retrieval/summary.md) (60.448 seconds)
- [11. Clean Fts5 Candidate](experiment_11_clean_fts5_candidate/summary.md) (367.138 seconds)

## Recommendation

Prioritize **clean_specific_query_pagination** for private or newly generated validation. It passed every retrospective diagnostic gate, but Experiment 11 does not authorize production promotion because the public data were already inspected. The starter agent remains unchanged.
