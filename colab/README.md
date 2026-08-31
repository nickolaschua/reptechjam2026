# BGE catalogue artifact workflow

Open `bge_pipeline.ipynb` in Google Colab and select **Runtime → Change runtime
type → T4 GPU**. A high-memory runtime is recommended: the final float32 matrix
is roughly 147 MiB before ZIP compression.

Run the notebook from top to bottom. The configuration cell defaults to
`https://github.com/nickolaschua/reptechjam2026.git` and `main`; set `REPO_REF`
to an exact commit SHA before a release build. The required path inside the
downloaded ZIP is:

```text
system/shopping_agent/embedding_cache/catalog_cache_bge-base-en-v1.5.npz
```

On a T4, dependency installation and cloning normally take 3–6 minutes and the
50,000-row base embedding pass roughly 10–25 minutes. Optional three-epoch
fine-tuning plus base/tuned evaluation can take 45–90 minutes. Runtime and ZIP
size vary with Colab allocation.

The base section checks the 50,000-row catalogue, builds product passages by
calling the production helper, writes schema-v2 metadata, reloads the file with
the production cache loader, and performs one catalogue/query cosine check. It
then creates `techjam-bge-artifacts.zip` with the cache and
`bge_artifact_manifest.json`.

After download, copy the cache to `system/shopping_agent/embedding_cache/`, keep
the manifest beside it while verifying, and run:

```powershell
python colab/verify_bge_artifact.py system/shopping_agent/embedding_cache/catalog_cache_bge-base-en-v1.5.npz --manifest system/shopping_agent/embedding_cache/bge_artifact_manifest.json
python -m system.shopping_agent.demo --scripted
```

Add `--cosine-check` to the verifier when Sentence Transformers and the BGE
weights are installed locally. Without that flag, all cache, row-order,
fingerprint, dimension, normalization, metadata, and checksum checks remain
fully offline.

## Optional tuned model

Set `RUN_FINE_TUNING = True` only after the stock cache is complete. The notebook
uses tracked messy utterance/product pairs, a deterministic product-level 20%
holdout (`seed=20260830`), same-category hard negatives, Multiple Negatives
Ranking Loss, and three epochs. It reports hit@10 and MRR for base and tuned BGE,
then exports the tuned weights, separately named cache, held-out IDs, metrics,
and checksum manifest.

The tuned output is never copied onto the stock cache name and is never selected
by production configuration. Promotion requires a later explicit code change
after its fixed-holdout metrics have been reviewed.
