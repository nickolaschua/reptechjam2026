# Eleven-experiment evaluation methodology

## Scope and datasets

The canonical inputs are `techjam-conversational-search/data/catalog.jsonl` (50,000 products) and `public_set.jsonl` (200 labeled sessions: 80 buying, 80 browsing, 30 intent-override, 10 boundary). The runner hashes both files with SHA-256 and records the hashes in `nickolas/results/manifest.json`. It reconstructs hidden intent cards and behavior by calling the participant kit's unmodified `materialize_hidden_fields`, which delegates to the official `intent_card` and `behavior_for` functions.

The seed is `20260826`. Catalog order is fixed, and ties are broken by ascending `parent_asin`. Python, package, OS, CPU, model, Git commit, worktree status, commands, and timestamps are recorded. Timings and timestamps are intentionally excluded from determinism comparisons.

## Leakage boundary

Experiments 1, 3, 4, and 5 are prominently labeled **oracle diagnostics**. They may inspect reconstructed cards, target fields, or all four hidden constraints to characterize the public data. They do not make agent-performance claims.

Experiments 2 and 6 are **agent-realistic evaluations**. At turn *t*, a query contains only the coarse catalog category and constraints disclosed by the official simulator through turn *t*. Latest-message BM25 sees only the current user message. Intent overrides remove the obsolete preference from accumulated state before retrieval. The target ASIN, target fields, and undisclosed constraints are used only to compute ranks and scores, never to construct a query. Full diagnostic traces always run for ten turns; score replay obeys normal early termination and ignores pre-override target appearances.

Experiment 7 is an **agent-realistic evaluation with oracle-after-freeze diagnostics**. Its ranker input is a frozen two-field object containing only category and active disclosed constraints. Target ASIN, scenario, sample ID, intent card, future turns, and user profile are excluded from rankers. All exact, lexical, dense, and cascade Top-10 lists are frozen before labels are joined. User profiles are copied only into residual audit records and never affect retrieval.

Experiment 8 is an **agent-realistic routed evaluation**. A binary route is selected from the first user message and locked for the session. Exploratory cues route to pure MiniLM cosine Top-10; explicit requirements and the conservative unknown fallback route to the Experiment 7 exact-first/conditional-BM25 policy. Boundary and intent-override labels are not visible to the detector. Exact and BM25 call instrumentation must remain zero on browsing-routed treatment turns. A catalog-wide observable constraint lexicon disambiguates evaluator messages where `; ` can mean either a disclosure delimiter or punctuation inside one catalog-derived constraint.

Experiment 9 is an **agent-realistic staged ablation**. Its typed state contains category, intent seed, constraint type, negation, source turn, strength, profile tags, and asked attributes. Targets, scenarios, hidden cards, and future turns never enter retrieval or clarification selection. The typed identity stage must reproduce Experiment 7 before adaptive retrieval, reranking, clarification, or profile evidence is admitted.

Experiment 10 is an **agent-realistic XTR/WARP evaluation with oracle-after-freeze diagnostics**. Its 600 unique retrieval queries contain only category plus active disclosed evidence. A pinned pretrained XTR model and pinned xtr-warp source revision build a 4-bit WARP index in Colab; no model weights are fine-tuned. Top-1000 rankings are frozen at `nprobe=32` before target ASINs, scenario labels, and split membership are joined locally. The returned archive is rejected unless every checksum, provenance value, PID-to-ASIN mapping, query, rank, and score agrees with the local inputs and manifest.

Experiment 11 is a **retrospective public-set candidate evaluation**. It reconstructs Yang's SQLite FTS5 architecture without target, scenario, hidden-card, sample-ID, profile, or future-turn inputs. The parser is case-insensitive, intent overrides remove only the revoked initial preference, stale preferences receive no boost, and all rank ties use ascending `parent_asin`. Because Yang's implementation and the public sessions were inspected before this experiment was defined, its frozen 60/140 split is useful for diagnostics and calibration discipline but is not an unbiased holdout. Experiment 11 therefore never authorizes a production edit; the selected candidate must first pass private or newly generated validation.

The deterministic policy always asks `other`. This is evaluator-specific and intentionally broad: unlike attribute-specific questions, `other` can disclose any remaining generated constraint. Boundary sessions give the official no-preference reply once. Buying sessions disclose one hard constraint in the initial message; browsing starts category-only; override sessions begin with the obsolete preference and replace it on the evaluator-selected turn.

## Normalization and retrieval definitions

Normalized phrase matching lowercases text, collapses whitespace, and strips surrounding whitespace. Catalog field values are flattened in evaluator-compatible form. Price is additionally rendered as `budget around $PRICE price PRICE` so evaluator-created budget constraints have an explicit field representation.

- **Exact/phrase:** ranks by 1,000 times the number of accumulated phrases occurring verbatim, plus mean token coverage; nonmatching documents are not retrieved.
- **All-token:** a hard candidate must contain every normalized non-stopword token from every accumulated phrase.
- **Token overlap:** the experiment-1 diagnostic retains documents whose summed per-phrase token coverage is at least 50% of the number of phrases.
- **BM25:** Okapi BM25 uses `k1=1.2`, `b=0.75`, binary query terms, and the whole searchable corpus.
- **Field-aware BM25:** sums per-field BM25 with title 6.0, categories 4.0, features 2.5, details 2.5, store 1.5, description 1.0, and price 1.0.
- **Dense:** CPU inference with `sentence-transformers/all-MiniLM-L6-v2`, a recorded 128-token maximum sequence length, L2-normalized embeddings, and cosine similarity. Dense text orders title, categories, store, features, details, description, then price so the most discriminative identity/category signal precedes any truncation. Catalog embeddings are cached as float32 NumPy arrays.
- **Hybrid:** reciprocal rank fusion of field-aware BM25 and dense results, each truncated to depth 1,000, with `RRF_K=60`. Targets outside the fused pool are `not_retrieved`.

Experiment 7 preserves the Experiment 2 exact order unless exact evidence is weak: no active constraint exists, no product matches every active phrase exactly, or the highest exact-match tier contains more than ten products. Only then does it compare four equal-weight, depth-1,000, `RRF_K=60` cascades: exact + generic BM25; exact + field-aware BM25; exact + MiniLM; and exact + generic BM25 + field-aware BM25 + MiniLM. All fused ties use ascending `parent_asin`. Field weights, fusion constants, and cases are not tuned.

Experiment 8 embeds `category + active constraints` with the same 128-token normalized float32 MiniLM model on browsing routes. Its deterministic stress suite applies frozen synonym substitution, clause reordering, lexical compression, and punctuation transformations to identical control/treatment messages. Except for punctuation-only transforms, an assertion prevents a transformed constraint from retaining its original normalized phrase.

Experiment 9 recomputes exploratory/mixed/specific state each turn and applies the preregistered weighted RRF table over exact, BM25, dense, and structured rankings. Structured budget violations are the only irreversible filter. The deterministic reranker uses the recorded `0.35/0.20/0.25/0.10/0.10/-0.30` normalized feature formula. Its final profile ablation reduces hybrid weight to 0.30, adds profile weight 0.05, and decays inferred soft evidence by `0.9^turn_age`; explicit requirements do not decay.

Experiment 10 uses the same conditional exact-first policy as Experiment 7. When fallback is active, it compares equal-weight depth-1,000 `RRF_K=60` fusion of exact + generic BM25 with exact + XTR/WARP. WARP retrieval itself is run in its required CPU mode after GPU index construction. Frozen WARP ties are score-descending then PID-ascending; fused ties are ascending `parent_asin`. Exact order is preserved whenever fallback is inactive.

Experiment 11 stores title, categories, features, description, details, and store in an in-memory SQLite FTS5 index. Its weighted OR query uses field weights `6.0/4.0/2.5/1.0/2.5/1.5`, adds `0.02 * rating_number^0.1`, then applies deterministic brand/title diversification. It compares specific versus broad `other` questions and global, query-scoped, or disabled pagination. The candidate pool is capped at 1,000, and query-scoped pagination keys shown products by the normalized active query so a genuinely changed intent can revisit relevant products.

Full-catalog ranks are calculated for lexical, exact, field-aware, and dense methods. A missing lexical target is reported literally as `not_retrieved`, never as 50,001.

## Candidate definitions

Experiment 5 keeps hard filters separate from ranking candidates. Exact and all-token are hard definitions. BM25 candidates score at least 50% of the top BM25 score. Dense candidates score at least both cosine 0.25 and 80% of the top cosine. Hybrid candidates score at least 50% of the top RRF score. The report gives mean, median, p25, p75, p90, p95, and proportions at 1, 10, and 100 candidates overall and by scenario/position.

## Metrics and counterfactuals

HitRate@10 is the fraction of sessions whose target appears in the shown slate after any required override. MRR is the mean reciprocal rank at first conversion. Misses receive turn 11 for MTTC. Efficiency is `clip((11 - MTTC) / 10, 0, 1)`. Technical score is `0.50*HitRate@10 + 0.30*MRR + 0.20*efficiency`.

Experiment 6 compares widths 1, 3, 5, and 10. A seeded 30% split is drawn independently within each scenario for calibration; all reported final comparisons use the remaining 70%. Adaptive confidence is 65% normalized hybrid top score and 35% top-two relative margin. High/medium thresholds, relative candidate inclusion, and optional low-confidence abstention are selected solely by calibration technical score, with deterministic tie-breaking.

Experiment 7 reuses and validates Experiment 6's exact 60 calibration and 140 held-out IDs. A cascade is selected only on calibration TechnicalScore, breaking ties by more hard-failure rescues, fewer regressions, higher MRR, then method name. Held-out results never reselect a route. The selected cascade is recommended only when held-out TechnicalScore is at least exact-only, it rescues a held-out hard failure, and regressions do not exceed rescues. Otherwise exact-only is retained.

Experiment 8 uses the same frozen split but does not tune the route. Official metrics are reported by route and scenario only after ranking freeze. Its paraphrases are a robustness stress test, not an official-score replacement.

Experiment 9 compares fixed-`other`, candidate-entropy, and rank-aware value-of-clarification branches on calibration only. Cumulative configuration selection uses TechnicalScore, rescues, fewer regressions, MRR, then method name. Exactly one selected treatment is evaluated on held-out sessions. Promotion additionally requires held-out TechnicalScore and MRR at least Experiment 7, at least one hard-failure rescue, and regressions no greater than rescues.

Experiment 10 does not tune or reselect any configuration. XTR/WARP is recommended only if its held-out TechnicalScore strictly exceeds the Experiment 7 BM25-RRF control, it rescues at least one exact-only held-out hard failure, and its exact-baseline regressions do not exceed its rescues. Exact-only and BM25-RRF must first reproduce Experiment 7's 2,000 Top-10 turn slates and 200 session outcomes per control bit-for-bit.

Experiment 11 evaluates four fixed configurations. Selection uses only the reused 60-session calibration partition, ordered by TechnicalScore, MRR, then configuration name; the 140-session partition does not reselect the winner. Diagnostic gates require evaluation TechnicalScore, HitRate@10, and MRR to be no lower than the current submission and regressions not to exceed rescues. Passing these gates is evidence for a new-data validation candidate, not a promotion decision. The runner also requires the current submission to reproduce its saved official score exactly and reports Yang's original implementation as a separate control.

Hard failures are exact sessions with no eligible Top-10 hit; weak successes first hit at rank 6–10. Experiment 7 labels the hard failure's final eligible turn or the weak success's original first-hit turn with non-exclusive categories for exact mismatch, ambiguity, ranking, cross-field partial evidence, normalization, dialogue state, insufficient information, and semantic opportunity. Every disclosed string is attributed independently to title, features, details, description, categories, and store. The baseline's synthetic budget/price text is reported as a separate diagnostic source. Unicode diagnostics use NFKC/casefold and punctuation-to-space normalization; they do not alter baseline ranking.

## Reproduction

From the repository root on Windows PowerShell:

```powershell
python -m unittest discover -s nickolas/experiments/tests -v
python -m nickolas.experiments.run_all
```

To rerun one experiment while keeping existing artifacts:

```powershell
python -m nickolas.experiments.run_all --only 3 --skip-baseline
```

Experiment 7 alone is reproduced with:

```powershell
python -m nickolas.experiments.run_all --only 7 --skip-baseline
```

Experiments 8 and 9 can be reproduced independently with:

```powershell
python -m nickolas.experiments.run_all --only 8 --skip-baseline
python -m nickolas.experiments.run_all --only 9 --skip-baseline
```

After placing `experiment_10_colab_output.zip` in `nickolas/colab`, validate, import, and evaluate it with:

```powershell
python -m nickolas.experiments.run_all --only 10 --skip-baseline
python -m nickolas.experiments.experiment_10_cli --summary
python -m nickolas.experiments.experiment_10_cli --demo
```

Experiment 11 can be reproduced independently with:

```powershell
python -m nickolas.experiments.run_all --only 11 --skip-baseline
```

Its evaluator/system interactions can be explored turn by turn in the terminal. Add `--run` to stream a fresh run before the viewer opens:

```powershell
python -m nickolas.experiments.experiment_07_cli
python -m nickolas.experiments.experiment_07_cli --run
```

For a presentation-ready replay of one held-out hard-failure rescue, use:

```powershell
python -m nickolas.experiments.experiment_07_cli --demo
```

The viewer identifies agent-visible retrieval inputs separately from oracle-only target and scoring fields. Because Experiment 7 evaluates a retrieval policy rather than an `Agent.respond()` implementation, the system side consists of `ask_attribute=other` plus a Top-10 recommendation slate; it has no generated prose assistant message.

The artifact bundles include all-turn rows, session outcomes, route/state diagnostics, ablation and rescue comparisons, robustness results, latency, charts, logs, input/model/source hashes, and SHA-256-verified source snapshots. Experiments 8, 9, 10, and 11 report promotion or diagnostic decisions but never edit the submission agent. Experiment 10 also imports the reusable converted WARP index into a gitignored results subdirectory.

The first dense run may download the MiniLM model. Later runs use `nickolas/results/cache/`. Each experiment writes `summary.md`, `metrics.json`, `rows.csv`, charts, and `run.log`. The runner updates the cross-experiment index after every stage, catches failures without deleting completed directories, and returns nonzero on failure.

Local execution note: the pre-existing Python environment contained `torchvision 0.23.0`, `torchaudio 2.8.0`, and `torchcodec 0.7.0` binary builds that were incompatible with its CPU-only `torch 2.12.1`. Those optional vision/audio packages were removed because they prevented the text-only `sentence-transformers` import. They are not dependencies of this suite.
