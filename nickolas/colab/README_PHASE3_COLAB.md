# Phase 3 on Google Colab

This folder contains a GPU-ready notebook and a packaging script for the
controlled BGE versus OpenAI embedding bake-off.

## Included files

- `phase3_embedding_bakeoff_colab.ipynb`: run this in Google Colab.
- `prepare_phase3_colab.ps1`: creates the upload bundle from the current local
  source and datasets.
- `requirements-phase3-colab.txt`: notebook dependencies.
- `phase3_colab_bundle.zip`: generated upload bundle after running the script.

The bundle deliberately excludes `.env`, caches, prior results, model weights,
and unrelated experiments.

## Prepare locally

From the repository root in PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File nickolas/colab/prepare_phase3_colab.ps1
```

This creates:

```text
nickolas/colab/phase3_colab_bundle.zip
```

Upload that ZIP to this Google Drive location:

```text
MyDrive/techjam26_phase3/phase3_colab_bundle.zip
```

You can instead upload the ZIP directly to the Colab session when the notebook
asks, but Drive is more reliable.

## Run in Colab

1. Open `phase3_embedding_bakeoff_colab.ipynb` in Colab.
2. Select **Runtime > Change runtime type > T4 GPU** (or a better GPU).
3. Run the cells from top to bottom.
4. Add `OPENAI_API_KEY` to Colab Secrets when prompted. The notebook reads it
   through `google.colab.userdata`; it never prints or stores the key.
5. Confirm the displayed catalog count is 50,000 and OpenAI batch estimate is
   50 before running the explicit full benchmark cell.

The notebook sets `BGE_EMBEDDING_BATCH_SIZE=64` for T4 safety. Batch size only
changes execution memory/throughput; product text, vectors, normalization,
query prefix, dot-product ranking, candidate depth, and evaluator behavior stay
unchanged.

## Persistent outputs

The notebook writes to:

```text
MyDrive/techjam26_phase3/outputs/embedding_cache/
MyDrive/techjam26_phase3/outputs/benchmark_results/
```

Copy both generated NPZ cache files back to the matching local folder when the
run finishes:

```text
nickolas/shopping_agent/embedding_cache/
```

The cache metadata includes backend/model ID, exact catalog row order, row
count, dimension, normalization flag, product-text fingerprint/version, and
catalog fingerprint, so copied caches will be rejected if the local catalog or
code text construction differs.

The controlled retrieval cell is the primary experiment. The optional
end-to-end cell is disabled by default because the LLM shopper is stochastic
and can make many additional hosted chat requests.
