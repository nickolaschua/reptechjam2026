# BGE over OpenAI: Phase 3 embedding bake-off

## Result location and provenance

The original Colab output is stored beside this report:

- Repository-relative location: `nickolas/shopping_agent/benchmark_results/phase3_benchmark_results.zip`
- Local location: `C:\Users\nicko\Desktop\techjam26\nickolas\shopping_agent\benchmark_results\phase3_benchmark_results.zip`
- Download/open: [phase3_benchmark_results.zip](./phase3_benchmark_results.zip)
- SHA-256: `A8FEB8AE3B235440DD6008BB6581CFA29B576239CD40F01D729B5DEE7220DD4C`
- Summary generation time: `2026-08-29T15:33:02.272700+00:00`

The archive contains the original machine-readable artifacts:

- `retrieval_fixture.json` — 200 controlled retrieval fixtures.
- `retrieval_bge_20260829T152825Z.json` — BGE metrics, per-query ranks, and timings.
- `retrieval_openai_20260829T153302Z.json` — OpenAI metrics, per-query ranks, timings, and API usage.
- `comparison_summary.json` — combined machine-readable summary.
- `comparison_summary.md` — original generated comparison table.
- `openai_smoke_20260829T152343Z.json` — OpenAI dimension, normalization, latency, and usage smoke check.

The benchmark implementation is [compare_embeddings.py](../compare_embeddings.py). The original ZIP is the source of truth for every reported measurement below.

## Models compared

| Variant | Model | Query convention |
|---|---|---|
| BGE | `BAAI/bge-base-en-v1.5` | Existing BGE retrieval instruction applied internally |
| OpenAI | `text-embedding-3-large` | Same query text, without the BGE-specific instruction |

Both variants used normalized embeddings and the same dense dot-product ranking implementation. OpenAI used its default 3,072-dimensional output; the smoke test recorded norm `1.0`.

## Testing methodology

### Controlled retrieval benchmark

The primary comparison used 200 fixtures generated from the public evaluator set in its original order:

| Scenario | Queries |
|---|---:|
| Buying | 80 |
| Browsing | 80 |
| Intent override | 30 |
| Boundary | 10 |
| **Total** | **200** |

For each fixture, the runner recorded a canonical state-derived retrieval query and the target ASIN for offline scoring. The same fixture file was then supplied to both embedding backends. A post-run integrity check confirmed that all 200 records had identical sample IDs, ordering, scenario types, target ASINs, and visible query text across BGE and OpenAI.

Apart from the embedding backend, the retrieval conditions were held constant:

- identical catalog and catalog row ordering;
- identical product-text construction;
- identical canonical query content;
- normalized catalog and query vectors;
- identical NumPy dot-product similarity and descending sort;
- identical target-rank calculation;
- no target ASIN included in query text;
- no evaluator/shopper LLM generating different comparison queries;
- no reranking, routing, filtering, prompts, or response-generation differences in this retrieval-level test.

For each query, the runner embedded the query, searched the complete catalog ranking, and recorded target rank, reciprocal rank, Recall@10, Recall@50, Recall@150, query-embedding latency, and dense-search latency. Timing used `time.perf_counter()` instrumentation. Both runs reported zero failures.

### Additional paired analysis

The per-query JSON records were also compared pairwise rather than treating the two runs as unrelated samples:

- threshold disagreements were counted at ranks 1, 10, 50, and 150;
- an exact two-sided paired binomial/McNemar-style test was applied to the discordant Recall@10 outcomes;
- deterministic paired bootstrap resampling with 20,000 replicates estimated uncertainty for metric differences.

This additional analysis did not change retrieval ordering or any original result file.

### What was not tested

The archive does not contain an extended end-to-end shopper evaluation. Consequently, Extended HR@10, Extended MRR, mean turns, agent-response latency, route counts, and total evaluator wall-clock time are unavailable. Those fields are `null` in the original summary. End-to-end shopper runs would also be stochastic and should be treated as supporting evidence rather than a perfectly paired embedding test.

## Controlled retrieval results

| Metric | BGE | OpenAI | BGE minus OpenAI |
|---|---:|---:|---:|
| Dense Recall@10 | **0.7150** | 0.6350 | **+0.0800** |
| Dense Recall@50 | **0.8500** | 0.8250 | +0.0250 |
| Dense Recall@150 | **0.9450** | 0.9350 | +0.0100 |
| Dense MRR | **0.5485** | 0.5153 | +0.0332 |
| Median target rank | 2 | 2 | 0 |
| Rank-1 targets | **95/200** | 88/200 | +7 |
| Failures | 0 | 0 | 0 |

At Recall@10, the paired outcomes were:

- both backends hit: 119 queries;
- BGE-only hit: 24 queries;
- OpenAI-only hit: 8 queries;
- neither hit: 49 queries.

The Recall@10 paired exact test produced `p ≈ 0.007`. The paired bootstrap 95% interval for the BGE Recall@10 advantage was approximately `+0.025` to `+0.135`. The smaller Recall@50, Recall@150, and MRR differences had intervals crossing zero, so they are not conclusive at this sample size.

Across exact target ranks, BGE ranked the target higher in 66 queries, OpenAI ranked it higher in 53, and 81 were tied.

### Scenario-level Recall@10

| Scenario | BGE | OpenAI | Difference |
|---|---:|---:|---:|
| Buying | **0.6750** | 0.5750 | +0.1000 |
| Browsing | **0.7875** | 0.6625 | +0.1250 |
| Intent override | 0.6333 | 0.6333 | 0.0000 |
| Boundary | 0.7000 | **0.9000** | -0.2000 |

The boundary result contains only ten queries and should not be generalized without more samples.

## Latency and API usage

| Metric | BGE | OpenAI |
|---|---:|---:|
| Query embedding p50 | **15.6 ms** | 245.9 ms |
| Query embedding p95 | **26.3 ms** | 276.3 ms |
| Dense search p50 | **12.3 ms** | 44.0 ms |
| Dense search p95 | **15.9 ms** | 65.9 ms |
| Approximate combined p50 | **27.9 ms** | 289.9 ms |
| Approximate combined p95 | **42.2 ms** | 342.2 ms |
| Catalog embedding generation | 241.4 s | **167.1 s** |
| Total initialization | 259.6 s | **214.6 s** |
| Embedding API requests | 0 | 250 |
| Embedding API input tokens | 0 | 3,631,470 |

OpenAI semantic retrieval was about `10.4×` slower at the combined p50. Its faster catalog build was a one-time Colab measurement involving remote batching and should not be interpreted as cached-startup performance. Of OpenAI's 250 requests, 200 correspond to the controlled queries; the remaining 50 are consistent with batched catalog generation.

## Methodology limitation

The controlled comparison is exactly paired, but its fixture construction has an important limitation. `build_fixture()` calls `materialize_hidden_fields()`. For public samples without pre-materialized fields, that function obtains the ground-truth product and generates an intent card from its metadata. The fixture then inserts the resulting hard constraints and soft preferences into `canonical_state["disclosed_slots"]`.

The target ASIN itself is never included in query text, but the query can contain near-verbatim target-product attributes. This conflicts with the intended restriction against placing target metadata into Fast Memory and can make the retrieval task more lexical and easier than realistic shopper-derived queries. Therefore:

- the measurements validly compare BGE and OpenAI on these exact 200 target-derived queries;
- they should not be treated as a final, leakage-free estimate of production retrieval quality;
- the fixture should be corrected and the query-only comparison rerun using the existing catalog caches before finalizing the Phase 3 decision.

## Conclusion

For the benchmark that was actually run, **vanilla BGE outperformed OpenAI `text-embedding-3-large` at the most important early-retrieval threshold**:

- BGE delivered an 8 percentage-point Recall@10 advantage;
- the paired Recall@10 difference was statistically persuasive for this fixture;
- BGE semantic retrieval was roughly ten times faster at p50;
- OpenAI added 250 API requests, 3.63 million embedding input tokens, and a network dependency;
- OpenAI showed no overall retrieval-quality gain that justifies replacing BGE or abandoning future BGE tuning.

The current evidence therefore favors keeping BGE for the hackathon. This conclusion remains provisional until the target-metadata fixture issue is corrected and, ideally, both variants receive the same extended end-to-end evaluator configuration.
