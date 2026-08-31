# Memory evaluation status

The gated vector-memory implementation used by the demo is experimental. The demo intentionally freezes the current implementation and does not claim benchmark superiority.

The deterministic v2 evaluator contains 40 probes across longitudinal-positive, memory-irrelevant, current-override, and browsing-personalization slices. Its fixture validation, paired invariants, artifact hashes, offline reconstruction, deterministic rerun, and complete test-suite checks passed; it was classified **EVALUATOR NOW TRUSTWORTHY**.

Under its strict pre-registered criteria, the current untuned mechanism was classified **HARMFUL**. Overall relevant-set MRR changed from `0.021452` without memory to `0.019327` with memory at the frozen configuration.

The relevance cosine itself was highly discriminative: RELEVANT-vs-IRRELEVANT ROC AUC was `0.97`. Threshold calibration nevertheless found no valid operating region because activated fixed blending still harmed required slices. A staged Buying blend sweep found one isolated strict-pass point at memory weight `b=0.05`, but no broad stable region; Browsing and final stages therefore were not run.

Memory representation refinement, parameter validation, and further evaluator research are deferred until after demo integration. The demo keeps the existing threshold, Buying/Browsing weights, EWMA alpha, representation, embeddings, and update mechanism unchanged.

Trustworthy evidence, manifests, hashes, reports, vectors, and scripts are preserved under `archive/research_evaluation/memory/`. Legacy QLMP/projector evidence remains under `archive/legacy_qlmp/`.
