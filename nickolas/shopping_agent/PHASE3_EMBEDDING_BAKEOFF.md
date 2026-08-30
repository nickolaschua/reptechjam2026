# Phase 3 embedding bake-off

The controlled retrieval benchmark is the primary embedding-quality evidence.
It presents the same canonical state-derived query text to both backends; only
the BGE backend internally adds its retrieval instruction. Ground-truth ASINs
are held in the offline fixture for scoring and are never passed to
`Agent.respond()`, Fast Memory, or an embedding query.

## Safety

- Importing either variant does not construct an OpenAI client or embed data.
- `agent_openai.Agent` loads a validated local cache by default and fails if it
  is absent or incompatible.
- A missing OpenAI catalog cache can only be built by passing
  `--allow-openai-catalog-build` to the comparison runner.
- OpenAI errors remain OpenAI failures; there is no fallback to BGE.
- All caches are stored under `nickolas/shopping_agent/embedding_cache/` and
  results under `nickolas/shopping_agent/benchmark_results/` by default.

## Commands

Build the deterministic 200-query fixture without loading an embedding model
or making an API call:

```powershell
python nickolas/shopping_agent/compare_embeddings.py fixture --samples 200
```

Run one small OpenAI query smoke test (never embeds the catalog):

```powershell
python nickolas/shopping_agent/compare_embeddings.py smoke-openai
```

Run the controlled retrieval bake-off. The explicit flag permits a batched
OpenAI catalog build only when its validated cache is missing:

```powershell
python nickolas/shopping_agent/compare_embeddings.py retrieval --samples 200 --allow-openai-catalog-build
```

Run both variants through the unchanged shared extended evaluator after both
caches exist:

```powershell
python nickolas/shopping_agent/compare_embeddings.py end-to-end --samples 200 --repeats 1
```

Run retrieval followed by the shared evaluator and produce a combined summary:

```powershell
python nickolas/shopping_agent/compare_embeddings.py all --samples 200 --repeats 1 --allow-openai-catalog-build
```

The LLM shopper is stochastic. Repeats are supported, but even repeated
end-to-end results are supporting evidence rather than a perfectly paired
embedding experiment. No winner is declared automatically.

## Phase 0 map

- BGE was initialized in `Agent.__init__` with `SentenceTransformer`.
- Product text was constructed in `_build_vector_index` from title, all
  categories, and the first three features.
- Catalog vectors were generated in `_build_vector_index`, normalized, and
  written to a model-named NPZ file beside `agent.py`.
- Query vectors were generated in `_respond_custom` after the FTS candidate
  count fell below 10.
- The BGE retrieval instruction was prepended in `_respond_custom`.
- Catalog and query normalization both used NumPy L2 norms.
- Similarity was `np.dot(catalog_embeddings, query_embedding)` and dense depth
  was the first 150 rows of descending `np.argsort`.
- The old cache trusted only its filename and loaded vectors without validating
  model, catalog, row order, text, normalization, row count, or dimension.
- `experiment_1/run_eval_v2.py` collects HR@10, MRR, MTTC, efficiency, a
  recommended score, and scenario-level metrics. It writes one fixed JSON file,
  swallows agent exceptions into empty recommendations, and did not previously
  collect route, dense-use, or agent latency metrics.
