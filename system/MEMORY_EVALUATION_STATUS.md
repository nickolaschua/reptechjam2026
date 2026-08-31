# Memory evaluation status

The gated vector-memory implementation used by the demo is experimental. The demo intentionally freezes the current implementation and does not claim benchmark superiority.

The deterministic v2 evaluator contains 40 probes across longitudinal-positive, memory-irrelevant, current-override, and browsing-personalization slices. Its fixture validation, paired invariants, artifact hashes, offline reconstruction, deterministic rerun, and complete test-suite checks passed; it was classified **EVALUATOR NOW TRUSTWORTHY**.

Under its strict pre-registered criteria, the current untuned mechanism was classified **HARMFUL**. Overall relevant-set MRR changed from `0.021452` without memory to `0.019327` with memory at the frozen configuration.

The relevance cosine itself was highly discriminative: RELEVANT-vs-IRRELEVANT ROC AUC was `0.97`. Threshold calibration nevertheless found no valid operating region because activated fixed blending still harmed required slices. A staged Buying blend sweep found one isolated strict-pass point at memory weight `b=0.05`, but no broad stable region; Browsing and final stages therefore were not run.

The archived evaluation above describes the former fixed `alpha=0.30` update and remains frozen. The runtime now defaults to a bounded novelty-adaptive centroid update while retaining fixed EMA as an experimental control; its relevance threshold has been raised from `0.20` to `0.30`, while the Buying/Browsing weights, one-vector representation, and embeddings remain unchanged. The frozen offline threshold reconstruction reports improved overall and irrelevant-slice deltas at `0.30`, but the point still fails the strict operating-region criteria and is not benchmark evidence for the adaptive updater. This change must not be described as benchmark superiority unless the adaptive-vs-fixed acceptance criteria pass, including a dormant-interest slice that the frozen fixture does not cover.

Trustworthy evidence, manifests, hashes, reports, vectors, and scripts are preserved under `archive/research_evaluation/memory/`. Legacy QLMP/projector evidence remains under `archive/legacy_qlmp/`.
