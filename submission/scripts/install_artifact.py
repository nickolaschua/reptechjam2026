"""Download the release cache, enforce its SHA-256, and install atomically."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import tempfile
import urllib.request


BUNDLE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DESTINATION = BUNDLE_ROOT / "artifacts" / "catalog_cache_bge-base-en-v1.5.npz"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION)
    args = parser.parse_args()

    expected = args.sha256.strip().lower()
    if len(expected) != 64 or any(char not in "0123456789abcdef" for char in expected):
        raise SystemExit("--sha256 must be a 64-character hexadecimal digest")

    destination = args.destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=destination.name + ".", suffix=".download", dir=destination.parent, delete=False
    ) as temporary:
        temporary_path = Path(temporary.name)
    try:
        urllib.request.urlretrieve(args.url, temporary_path)
        actual = _sha256(temporary_path)
        if actual != expected:
            raise SystemExit(f"cache SHA-256 mismatch: expected {expected}, got {actual}")
        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)
    print(f"installed {destination} ({expected})")


if __name__ == "__main__":
    main()
