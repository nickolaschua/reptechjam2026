# Required BGE catalogue artifact

Place the validated file here:

```text
catalog_cache_bge-base-en-v1.5.npz
```

It must contain 50,000 normalized 768-dimensional vectors produced for the exact
organizer catalogue row order with `BAAI/bge-base-en-v1.5` and the production
product-text function. The runtime rejects missing, stale, reordered, or
embedding-space-incompatible caches and never regenerates one during evaluation.

Use `scripts/install_artifact.py` to download a release asset with a required
SHA-256, then run `scripts/verify_artifact.py` against the organizer catalogue.
The release URL and final SHA-256 must be inserted into `README.md` before the
submission commit is frozen.
