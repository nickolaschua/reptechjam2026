# Query-conditioned memory portability isolation

- Fixture: 18 queries / 109 memory pairs / 4 users
- Positive-polarity label counts: {"CONFLICTING": 13, "CONTEXTUAL": 33, "IRRELEVANT": 23, "PORTABLE": 12, "REDUNDANT": 18}
- U4 negative-polarity diagnostic pairs: 10
- Candidate PORTABLE recall: Top-3 0.3333, Top-5 0.3333, Top-10 0.3333

## Primary binary comparison (judge capability)

- Raw cosine: AUROC 0.3514, AUPRC 0.1204
- Portability score: AUROC 0.4028, AUPRC 0.1505
- PORTABLE classification: precision 0.0000, recall 0.0000, F1 0.0000

## Multiclass

- Macro F1: 0.0872
- Balanced accuracy: 0.1599
- Confusion matrix: `{"CONFLICTING": {"CONFLICTING": 0, "CONTEXTUAL": 8, "IRRELEVANT": 5, "PORTABLE": 0, "REDUNDANT": 0}, "CONTEXTUAL": {"CONFLICTING": 0, "CONTEXTUAL": 2, "IRRELEVANT": 29, "PORTABLE": 1, "REDUNDANT": 1}, "IRRELEVANT": {"CONFLICTING": 0, "CONTEXTUAL": 5, "IRRELEVANT": 17, "PORTABLE": 0, "REDUNDANT": 1}, "PORTABLE": {"CONFLICTING": 0, "CONTEXTUAL": 0, "IRRELEVANT": 12, "PORTABLE": 0, "REDUNDANT": 0}, "REDUNDANT": {"CONFLICTING": 0, "CONTEXTUAL": 6, "IRRELEVANT": 8, "PORTABLE": 4, "REDUNDANT": 0}}`

## Hard negatives and no-useful-history states

- Rejection: `{"CONFLICTING": {"count": 13, "rejected": 13, "rejection_accuracy": 1.0}, "CONTEXTUAL": {"count": 33, "rejected": 32, "rejection_accuracy": 0.9696969696969697}, "IRRELEVANT": {"count": 23, "rejected": 23, "rejection_accuracy": 1.0}, "nearby_same_category": {"count": 46, "rejected": 45, "rejection_accuracy": 0.9782608695652174}}`
- No-useful-history: `{"false_portable_count": 4, "false_portable_rate": 0.08695652173913043, "fixture_ids": ["u2_override_s2_final", "u2_override_s4_final", "u2_override_s7_final", "u2_override_s9_final", "u3_distractor_s3_final", "u3_distractor_s5_final", "u4_negative_s2_final", "u4_negative_s5_final", "u4_negative_s8_final", "u4_negative_s9_final"], "pair_count": 46, "query_count": 10}`
- Error information audit: `{"AMBIGUOUS_LABEL": 0, "CONTEXT_LOST_DURING_MEMORY_DISTILLATION": 34, "ENOUGH_CONTEXT": 0, "MODEL_ERROR": 46}`

## Cost and latency

`{"failures": 0, "hosted_calls": 37, "input_tokens": 32371, "mean_latency_seconds": 2.833261408331358, "median_latency_seconds": 2.694019600006868, "output_tokens": 12573, "retries": 1, "successful_query_calls": 36}`

## Verdict

PORTABILITY INCONCLUSIVE

MEMORY PROVENANCE INSUFFICIENT: atomic MemoryItem text lost persistence/episode provenance for classification errors
