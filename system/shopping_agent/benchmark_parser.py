"""Run the archived 30-probe parser benchmark through production Ollama code."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

from .category_resolver import CategoryResolver
from .catalogue import Catalogue
from .config import CATALOG_PATH, PROJECT_ROOT
from .ollama_client import OllamaClient
from .turn_parser import WinstonTurnParser


MINIMUM_SLOT_F1 = 0.441


def _archive_root() -> Path:
    candidates = (
        PROJECT_ROOT / "docs" / "archive" / "winston",
        PROJECT_ROOT / "archive" / "winston",
        PROJECT_ROOT / "winston",
    )
    for candidate in candidates:
        if (candidate / "nlp_parse.py").exists() and (candidate / "probe_gold.json").exists():
            return candidate
    raise FileNotFoundError(
        "The archived Winston probe data is unavailable; restore archive/winston "
        "before running the live 30-probe benchmark"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="optional JSON prediction path")
    parser.add_argument("--minimum-slot-f1", type=float, default=MINIMUM_SLOT_F1)
    arguments = parser.parse_args()

    archive = _archive_root()
    specification = importlib.util.spec_from_file_location(
        "archived_winston_nlp_parse", archive / "nlp_parse.py"
    )
    module = importlib.util.module_from_spec(specification)
    assert specification.loader is not None
    specification.loader.exec_module(module)
    cases = json.loads((archive / "probe_gold.json").read_text(encoding="utf-8"))

    catalogue = Catalogue(CATALOG_PATH)
    try:
        resolver = CategoryResolver(catalogue)
        client = OllamaClient()
        turn_parser = WinstonTurnParser(resolver, client=client)
        predictions = []
        scores = []
        for case in cases:
            parsed = turn_parser.parse(case["utterance"], int(case["case"]))
            prediction = dict(parsed.raw_parse)
            predictions.append({"case": case["case"], "pred": prediction})
            scores.append(
                module.score(
                    prediction,
                    case["gold"],
                    case["discard_spans"],
                    case["utterance"],
                )
            )
    finally:
        catalogue.close()

    if arguments.output:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(
            json.dumps(predictions, indent=2) + "\n", encoding="utf-8"
        )
    slot_f1 = sum(row["slot_f1"] for row in scores) / len(scores)
    print(
        json.dumps(
            {
                "model": client.model,
                "probes": len(scores),
                "slot_f1": round(slot_f1, 3),
                "minimum_slot_f1": arguments.minimum_slot_f1,
                "passed": slot_f1 >= arguments.minimum_slot_f1,
            },
            indent=2,
        )
    )
    return 0 if slot_f1 >= arguments.minimum_slot_f1 else 1


if __name__ == "__main__":
    sys.exit(main())
