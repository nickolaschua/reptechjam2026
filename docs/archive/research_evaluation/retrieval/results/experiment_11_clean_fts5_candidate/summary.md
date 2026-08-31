# Experiment 11 — Clean FTS5 candidate

> **RETROSPECTIVE PUBLIC-SET EVALUATION.** The candidate uses only observable dialogue messages and catalog metadata. The preceding investigation inspected the public set and its frozen partition, so these numbers are diagnostic rather than an unbiased promotion test.

The current submission control reproduced its saved official score exactly at **0.840846**. Yang's original agent scored **0.896387**. Calibration selected **clean_specific_query_pagination**; it scored **0.894362** on all 200 sessions and **0.896425** on the 140-session evaluation partition.

The clean candidate removes stale-preference boosting, scopes override removal to the revoked preference, uses robust case-insensitive parsing, and applies deterministic ASIN tie-breaking. All diagnostic score, Hit@10, MRR, and rescue/regression gates passed: **True**. The starter agent was not modified because private or newly generated validation is still required.
