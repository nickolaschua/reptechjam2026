"""Regenerate the BGE catalogue cache from the mounted organizer catalogue.

This is the offline fallback for `install_artifact.py`: it produces the exact
same `catalog_cache_bge-base-en-v1.5.npz` locally instead of downloading a
published release asset. It reuses the production build path in Agent, so the
catalogue fingerprint, product-text fingerprint and embedding-space identity are
correct by construction rather than by convention.

The frozen evaluation entry point never reaches this code: `starter/agent.py`
pins ALLOW_CATALOG_EMBEDDING=false and this script is not imported by it.
"""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import sys
import time


BUNDLE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BUNDLE_ROOT))

# Must precede the runtime import: provider selection happens at import time.
os.environ["TEST_MODE"] = "false"
os.environ["ALLOW_CATALOG_EMBEDDING"] = "true"

from system.shopping_agent.agent import Agent  # noqa: E402
from system.shopping_agent.embedding_backends import cache_filename  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=BUNDLE_ROOT / "data" / "catalog.jsonl")
    parser.add_argument("--out-dir", type=Path, default=BUNDLE_ROOT / "artifacts")
    args = parser.parse_args()

    catalog = args.catalog.resolve()
    out_dir = args.out_dir.resolve()
    if not catalog.is_file():
        raise SystemExit(f"missing organizer catalogue: {catalog}")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = sum(1 for line in catalog.open("rb") if line.strip())
    print(f"[Build] catalogue {catalog} ({rows:,} rows)")
    print(f"[Build] encoding with BAAI/bge-base-en-v1.5; this takes a while on CPU.")

    started = time.perf_counter()
    agent = Agent(
        catalog_path=catalog,
        embedding_cache_dir=out_dir,
        allow_catalog_embedding=True,
    )
    elapsed = time.perf_counter() - started

    target = out_dir / cache_filename(agent.embedding_backend_id)
    if not target.is_file():
        raise SystemExit(f"build finished but no cache was written to {target}")
    print(f"[Build] wrote {target} in {elapsed / 60:.1f} min")
    print(f"[Build] shape={agent.catalog_embeddings.shape} sha256={_sha256(target)}")
    print("[Build] now run: python scripts/verify_artifact.py --catalog data/catalog.jsonl")


if __name__ == "__main__":
    main()
