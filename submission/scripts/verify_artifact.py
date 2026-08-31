"""Validate the BGE catalogue cache against the exact organizer catalogue."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
BUNDLE_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(BUNDLE_ROOT))
os.environ["TEST_MODE"] = "false"
os.environ["ALLOW_CATALOG_EMBEDDING"] = "false"

from system.shopping_agent.embedding_backends import (  # noqa: E402
    BGEEmbeddingBackend,
    CacheExpectation,
    PRODUCT_TEXT_VERSION,
    fingerprint_file,
    fingerprint_texts,
    load_embedding_cache,
    production_product_texts,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument(
        "--cache",
        type=Path,
        default=BUNDLE_ROOT / "artifacts" / "catalog_cache_bge-base-en-v1.5.npz",
    )
    args = parser.parse_args()

    products: list[dict] = []
    with args.catalog.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                products.append(json.loads(line))
    if len(products) != 50_000:
        raise SystemExit(f"expected 50,000 catalogue rows, found {len(products):,}")

    ids = [str(product["parent_asin"]) for product in products]
    texts = production_product_texts(products)
    backend = BGEEmbeddingBackend()
    expectation = CacheExpectation(
        backend_id=backend.backend_id,
        model_id=backend.model_id,
        embedding_space_id=backend.embedding_space_id,
        catalog_ids=ids,
        product_text_version=PRODUCT_TEXT_VERSION,
        product_text_fingerprint=fingerprint_texts(texts),
        catalog_fingerprint=fingerprint_file(args.catalog),
        vector_dimension=backend.vector_dimension,
        normalized=True,
    )
    vectors = load_embedding_cache(args.cache, expectation)
    print(json.dumps({
        "status": "ok",
        "catalog_rows": len(products),
        "cache_shape": list(vectors.shape),
        "model": backend.model_id,
        "embedding_space_id": backend.embedding_space_id,
        "catalog_sha256": _sha256(args.catalog),
        "cache_sha256": _sha256(args.cache),
    }, indent=2))


if __name__ == "__main__":
    main()
