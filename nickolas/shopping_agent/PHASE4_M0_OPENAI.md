# Phase 4: M0_OPENAI

`M0_OPENAI` is the no-longitudinal-memory **hybrid** development baseline. Its
current-session state (Fast Memory) is allowed, but historical user memory does
not affect retrieval or ranking. The production route remains:

```text
current-session state
  -> FTS/lexical retrieval
  -> dense OpenAI fallback only when the existing route requires it
  -> deterministic heuristic reranking
  -> diversification
  -> recommendations
```

The full path still invokes dense fallback only when the price-filtered FTS
candidate count is below 10. Dense fallback still scores the full catalogue,
takes Top-150, and only then applies the downstream price exclusion. This
ordering is the existing M0 behaviour; Checkpoint A does not change it.

OpenAI `text-embedding-3-large` is used intentionally for development
simplicity. Phase 3 found vanilla BGE stronger on the controlled embedding
benchmark, so M0 does not declare OpenAI the final or competition-best
embedder. The BGE implementation, cache isolation, and comparison runner remain
available.

## Dense embedding contract

- Provider: OpenAI
- Model: `text-embedding-3-large`
- Dimension: 3072
- Query and product vectors: L2-normalized
- M0 boundary dtype: `float32`
- Embedding fallback: none for the canonical M0 configuration
- Product row order: exact catalogue JSONL order
- Product text:
  `Product: {title}. Categories: {all categories}. Features: {first three features}.`

The cache identity, product text version, catalogue fingerprint, exact product
ID row order, model, dimension, and normalization metadata continue to guard
reuse of the one existing product matrix. No second catalogue matrix is built.
QLMP may later promote only its local geometry inputs to `float64`; vectors
crossing back into the M0 scorer use the catalogue's `float32` dtype.

## Dense vector interface

`Agent.embed_dense_query(query_text) -> numpy.ndarray` performs the existing
backend operation and returns the exact normalized query vector `q`.

`Agent.dense_retrieve_vector(query_embedding, top_n=150) -> DenseRetrievalResult`
accepts an already normalized vector. It validates one-dimensional shape,
catalogue dimension, finite values, and unit norm. It does not renormalize,
embed text, call OpenAI, apply filters, or read longitudinal memory. It uses the
canonical score and unchanged NumPy ordering:

```python
all_scores = np.dot(agent.catalog_embeddings, query_embedding)
row_indices = np.argsort(all_scores)[::-1][:top_n]
```

The result exposes aligned `query_embedding`, `row_indices`, `product_ids`,
`scores`, and Top-K `product_embeddings`. For every result position `j`, all
four returned product fields refer to the same catalogue row. The returned
query vector is the exact vector used by that score operation.

`Agent.dense_retrieve_text(...)` is the structured text route. The legacy
private `_dense_retrieve(...)` wrapper still returns row indices to the existing
M0 control flow, but now delegates text embedding and all scoring to the vector
interface. There is one catalogue scoring implementation.

`DenseQuerySnapshot` and `Agent.freeze_dense_query(...)` provide the minimal
replay contract: example ID, raw current user message, effective dense query
text, exact `q`, evaluation-only target product ID, and optional current scope
and category. The target ID is stored after query construction and is never an
embedding or ranking input.

## Future QLMP seam

Checkpoint A enables this later flow without changing the scorer:

```text
M0 current state -> exact q -> query-only dense result
                              -> future QLMP produces q_star
                              -> same dense_retrieve_vector scorer
```

The scorer is origin-agnostic, so a normalized alternative vector can be
passed directly. This task does **not** integrate QLMP, implement B1/B2/B3,
apply historical memory, or establish that memory improves retrieval. It also
does not show that projection beats cosine, that dense retrieval dominates
FTS, or that a future longitudinal evaluator is scientifically valid.

## Commands

Run a one-query OpenAI smoke check without loading or building the catalog:

```powershell
python nickolas/shopping_agent/run_m0.py --smoke
```

Run the exact frozen M0 evaluator configuration. A valid OpenAI cache is reused;
the explicit flag permits a missing or rejected cache to be built:

```powershell
python nickolas/shopping_agent/run_m0.py --allow-openai-catalog-build
```

`run_eval.py` is a compatibility alias for the same M0 runner.

Run the explicit BGE alternative through the retained Phase-3 infrastructure:

```powershell
python nickolas/shopping_agent/compare_embeddings.py end-to-end --backends bge --samples 200 --repeats 1
```

Completed M0 artifacts are written only to
`nickolas/shopping_agent/baseline_results/m0_openai/`.

## Checkpoint A interface freeze (2026-08-30)

The interface freeze was verified with 76/76 shopping-agent unit tests, AST
parsing of 22 Python files, and import checks for `agent`,
`embedding_backends`, `run_m0`, and `run_eval`. The expensive evaluator was not
run. SHA-256 values for the relevant implementation/config/test boundary are:

```text
agent.py                             6bad27303791a59c9740b049fbdbd34a439c8a951263b9ce9d382f82777f11a7
embedding_backends.py                3f1eb4bd286b703fa05084bc79cfa9065ee6d24f87fef6d8e9282d1437bd93ea
run_m0.py                            01489b7330528155957c6ea6c226a31e6c9a59919458fbde5ef33b2b916cda9d
run_eval.py                          9c5a78fd555ef6a03a35771c7a0f1894dd2ee251ba98e07a9204707dca971c36
configs/m0_openai.json               ad193a5363bb7c93cf2291dc0d2447773f347082a544e67286842a7132e70ee3
tests/test_dense_vector_interface.py a4e04dac85c11d9f428da293676e5fb496da218b0b06c1d9ced87494c15dc47a
tests/test_state_routing.py          0200cdd547a29e6c508c6167e7bc0630e0572e8a40602c5cbc08060cc8122d11
tests/test_m0_openai.py              1f92405eda41acfabe59e6c33e936078aff7bce51a45b86d189cc1617b09140d
tests/test_embedding_bakeoff.py      a900038955fb3951e687bec597d8b2025bd3d0c2ee573bf7b65b9a926435b214
```
