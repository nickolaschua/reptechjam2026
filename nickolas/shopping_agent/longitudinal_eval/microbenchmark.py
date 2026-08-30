"""Deterministic Phase 6 memory-selection microbenchmark metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


DEFAULT_CASES = Path(__file__).resolve().parent / "memory_microbenchmark.json"


def load_cases(path: str | Path = DEFAULT_CASES) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_cases(payload: Mapping[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    cases = payload.get("cases")
    if not isinstance(cases, list) or not cases:
        return {"valid": False, "errors": ["cases must be a non-empty list"]}
    seen_cases: set[str] = set()
    for case in cases:
        case_id = str(case.get("case_id", ""))
        if not case_id or case_id in seen_cases:
            errors.append("case_id values must be unique and non-empty")
        seen_cases.add(case_id)
        memories = case.get("visible_prior_memories", [])
        ids = [value.get("id") for value in memories if isinstance(value, Mapping)]
        if len(ids) != len(memories) or len(ids) != len(set(ids)):
            errors.append(f"{case_id}: memory IDs must be unique")
        known = set(ids)
        for field in (
            "relevant_memory_ids", "distractor_memory_ids",
            "should_suppress_memory_ids", "negative_memory_ids",
        ):
            values = case.get(field)
            if not isinstance(values, list) or not set(values).issubset(known):
                errors.append(f"{case_id}: invalid {field}")
        if set(case.get("relevant_memory_ids", [])).intersection(case.get("distractor_memory_ids", [])):
            errors.append(f"{case_id}: relevant and distractor labels overlap")
    return {"valid": not errors, "errors": errors, "case_count": len(cases)}


def score_selection(case: Mapping[str, Any], selected_memory_ids: Iterable[str]) -> dict[str, float]:
    selected = set(selected_memory_ids)
    relevant = set(case.get("relevant_memory_ids", []))
    distractors = set(case.get("distractor_memory_ids", []))
    suppress = set(case.get("should_suppress_memory_ids", []))
    negatives = set(case.get("negative_memory_ids", []))

    def recall(expected: set[str]) -> float:
        return 1.0 if not expected else len(selected.intersection(expected)) / len(expected)

    return {
        "relevant_memory_selection_recall": recall(relevant),
        "distractor_selection_rate": 0.0 if not distractors else len(selected.intersection(distractors)) / len(distractors),
        "override_contamination": 0.0 if not suppress else len(selected.intersection(suppress)) / len(suppress),
        "negative_memory_handling_recall": recall(negatives),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the deterministic memory microbenchmark")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    args = parser.parse_args()
    result = validate_cases(load_cases(args.cases))
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["valid"] else 1)


if __name__ == "__main__":
    main()


__all__ = ["load_cases", "score_selection", "validate_cases"]
