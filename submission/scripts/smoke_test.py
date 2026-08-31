"""Load the real release stack and execute one contract-compatible turn."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


BUNDLE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BUNDLE_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=BUNDLE_ROOT / "artifacts")
    args = parser.parse_args()
    os.environ["TECHJAM_BGE_CACHE_DIR"] = str(args.cache_dir.resolve())

    from starter.agent import Agent

    agent = Agent(args.catalog)
    try:
        agent.reset("submission-smoke", {})
        response = agent.respond(
            "submission-smoke",
            "I'm looking for women's running shoes. A key requirement is: under $100.",
            1,
            10,
        )
        required = {"message", "ask_attribute", "recommendations"}
        if not isinstance(response, dict) or not required <= response.keys():
            raise SystemExit("invalid Agent response schema")
        print(json.dumps(response, indent=2))
    finally:
        agent.close()


if __name__ == "__main__":
    main()
