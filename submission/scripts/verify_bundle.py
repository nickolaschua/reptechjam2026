"""Verify the allowlisted runtime snapshot against bundle_manifest.json."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


BUNDLE_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_ROOT = BUNDLE_ROOT / "system" / "shopping_agent"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    manifest = json.loads((BUNDLE_ROOT / "bundle_manifest.json").read_text(encoding="utf-8"))
    expected = manifest["runtime_files_sha256"]
    problems: list[str] = []
    for name, checksum in expected.items():
        path = RUNTIME_ROOT / name
        if not path.is_file():
            problems.append(f"missing {path.relative_to(BUNDLE_ROOT)}")
        elif _sha256(path) != checksum:
            problems.append(f"checksum mismatch: {path.relative_to(BUNDLE_ROOT)}")
    actual = {path.name for path in RUNTIME_ROOT.glob("*.py")}
    unexpected = sorted(actual - set(expected))
    if unexpected:
        problems.append("unexpected runtime modules: " + ", ".join(unexpected))
    if problems:
        raise SystemExit("bundle verification failed:\n  " + "\n  ".join(problems))
    print(f"bundle runtime verified: {len(expected)} files")


if __name__ == "__main__":
    main()
