"""Run the unmodified organizer evaluator with this bundle first on sys.path."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import runpy
import sys


BUNDLE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = BUNDLE_ROOT.parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--kit", type=Path, default=REPOSITORY_ROOT / "techjam-conversational-search"
    )
    parser.add_argument("--catalog", type=Path)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--output", type=Path, default=BUNDLE_ROOT / "results" / "results.json")
    args = parser.parse_args()

    kit = args.kit.resolve()
    evaluator = kit / "evaluator" / "local_evaluator.py"
    catalog = (args.catalog or kit / "data" / "catalog.jsonl").resolve()
    dataset = (args.dataset or kit / "data" / "public_set.jsonl").resolve()
    cache_dir = Path(
        os.environ.get("TECHJAM_BGE_CACHE_DIR", BUNDLE_ROOT / "artifacts")
    ).resolve()
    required = [evaluator, catalog, dataset, cache_dir / "catalog_cache_bge-base-en-v1.5.npz"]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit("missing required release files:\n  " + "\n  ".join(missing))

    os.environ["TEST_MODE"] = "false"
    os.environ["ALLOW_CATALOG_EMBEDDING"] = "false"
    os.environ["TECHJAM_CATALOG_PATH"] = str(catalog)
    os.environ["TECHJAM_BGE_CACHE_DIR"] = str(cache_dir)
    sys.path.insert(0, str(BUNDLE_ROOT))
    sys.path.insert(1, str(kit))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sys.argv = [
        str(evaluator), "--catalog", str(catalog), "--dataset", str(dataset),
        "--output", str(args.output.resolve()),
    ]
    runpy.run_path(str(evaluator), run_name="__main__")


if __name__ == "__main__":
    main()
