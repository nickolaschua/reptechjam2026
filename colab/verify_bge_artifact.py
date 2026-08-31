"""Validate a downloaded TechJam BGE catalogue cache against production code."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


EXPECTED_ROWS = 50_000
EXPECTED_DIMENSIONS = 768
EXPECTED_CACHE_NAME = "catalog_cache_bge-base-en-v1.5.npz"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _arguments() -> argparse.Namespace:
    here = Path(__file__).resolve()
    default_root = here.parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--repo-root", type=Path, default=default_root)
    parser.add_argument("--catalog", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--expected-sha256")
    parser.add_argument("--skip-checksum", action="store_true")
    parser.add_argument("--cosine-check", action="store_true")
    return parser.parse_args()


def main() -> int:
    arguments = _arguments()
    root = arguments.repo_root.resolve()
    sys.path.insert(0, str(root))
    from system.shopping_agent.catalogue import Catalogue
    from system.shopping_agent.embedding_backends import (
        BGEEmbeddingBackend,
        CacheExpectation,
        PRODUCT_TEXT_VERSION,
        fingerprint_file,
        fingerprint_texts,
        load_embedding_cache,
        production_product_texts,
    )

    artifact = arguments.artifact.resolve()
    if artifact.name != EXPECTED_CACHE_NAME:
        raise SystemExit(
            f"base artifact must be named {EXPECTED_CACHE_NAME}, got {artifact.name}"
        )
    digest = sha256_file(artifact)
    expected_checksum = arguments.expected_sha256
    manifest_path = arguments.manifest or artifact.with_name("bge_artifact_manifest.json")
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        files = manifest.get("files", {})
        expected_checksum = files.get(artifact.name, expected_checksum)
        if not expected_checksum:
            matches = [
                checksum
                for name, checksum in files.items()
                if Path(name).name == artifact.name
            ]
            if len(matches) == 1:
                expected_checksum = matches[0]
    if not arguments.skip_checksum:
        if not expected_checksum:
            raise SystemExit(
                "checksum validation requires --expected-sha256 or "
                "bge_artifact_manifest.json (use --skip-checksum only for diagnostics)"
            )
        if digest.lower() != str(expected_checksum).lower():
            raise SystemExit("artifact SHA-256 does not match the expected checksum")

    catalog_path = arguments.catalog or (
        root / "techjam-conversational-search" / "data" / "catalog.jsonl"
    )
    catalogue = Catalogue(catalog_path)
    try:
        if len(catalogue.ids) != EXPECTED_ROWS:
            raise SystemExit(
                f"expected {EXPECTED_ROWS:,} catalogue rows, got {len(catalogue.ids):,}"
            )
        texts = production_product_texts(catalogue.products)
        backend = BGEEmbeddingBackend()
        expectation = CacheExpectation(
            backend_id=backend.backend_id,
            model_id=backend.model_id,
            embedding_space_id=backend.embedding_space_id,
            catalog_ids=catalogue.ids,
            product_text_version=PRODUCT_TEXT_VERSION,
            product_text_fingerprint=fingerprint_texts(texts),
            catalog_fingerprint=fingerprint_file(catalog_path),
            vector_dimension=EXPECTED_DIMENSIONS,
            normalized=True,
        )
        matrix = load_embedding_cache(artifact, expectation)
        with np.load(artifact, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata_json"].item()))
        if metadata.get("schema_version") != 2:
            raise SystemExit("downloaded BGE cache must use schema_version 2")
        if matrix.shape != (EXPECTED_ROWS, EXPECTED_DIMENSIONS):
            raise SystemExit(f"unexpected embedding matrix shape: {matrix.shape}")
        norms = np.linalg.norm(matrix, axis=1)
        if not np.allclose(norms, 1.0, rtol=1e-4, atol=1e-5):
            raise SystemExit("embedding rows are not L2 normalized")

        cosine = None
        if arguments.cosine_check:
            query = backend.embed_query(texts[0])
            cosine = float(matrix[0] @ query)
            if not np.isfinite(cosine) or not -1.00001 <= cosine <= 1.00001:
                raise SystemExit("catalogue/query cosine check failed")
    finally:
        catalogue.close()

    print(
        json.dumps(
            {
                "artifact": str(artifact),
                "sha256": digest,
                "rows": EXPECTED_ROWS,
                "dimensions": EXPECTED_DIMENSIONS,
                "normalized": True,
                "schema_version": 2,
                "catalog_fingerprint": metadata["catalog_fingerprint"],
                "product_text_fingerprint": metadata["product_text_fingerprint"],
                "cosine_check": cosine,
                "status": "valid",
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
