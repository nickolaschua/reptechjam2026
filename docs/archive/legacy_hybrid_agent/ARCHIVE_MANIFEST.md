# Legacy hybrid-agent archive manifest

This directory is the complete historical `experiment_1` tree. Its source, evaluators, result files, dashboard, ASIN image mapping, model/cache paths, and historical imports are preserved for provenance.

Ported into the active `system.shopping_agent` runtime:

- the exact FTS5 fields, tokenizer, AND/weighted-OR limits, weights, and route thresholds;
- canonical demographic normalization and hard eligibility relationships;
- price, rating, review-count, store, and explicit-negative masks;
- constraint provenance, revocation, search epochs, and epoch-local seen state;
- entropy/gain-ratio clarification selection;
- the unchanged browser `index.html`, public-sample materialization, shopper simulation, and ASIN image map.

Historical only:

- BGE loading and BGE fallback embeddings;
- heuristic/category/phrase boosts, popularity mixing, and diversification;
- old evaluator entry points and reports.

The active system uses the frozen OpenAI catalogue matrix and gated user-vector memory. It does not use BGE, persistent hard criteria, or continual embedding fine-tuning. Historical path strings and hashes in this archive are intentionally unchanged.
