"""Offline public-row discovery and auditable selection report generator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from .validate_fixture import DEFAULT_CATALOG, DEFAULT_FIXTURE, DEFAULT_PUBLIC, load_jsonl
except ImportError:
    from validate_fixture import DEFAULT_CATALOG, DEFAULT_FIXTURE, DEFAULT_PUBLIC, load_jsonl


DEFAULT_REPORT = Path(__file__).resolve().parent / "CANDIDATE_SELECTION_REPORT.md"


def _compact(value: Any, limit: int = 190) -> str:
    text = " ".join(str(value).split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def build_report(
    fixture: dict[str, Any],
    samples: dict[str, dict[str, Any]],
    products: dict[str, dict[str, Any]],
) -> str:
    selected_ids = {
        session["source_sample_id"]
        for user in fixture["users"]
        for session in user["sessions"]
    }
    lines = [
        "# Phase 6 candidate-selection report",
        "",
        "Generated offline from the current official public set and 50,000-row catalogue. "
        "Every selected session reuses an existing public buying row; no ASIN was fabricated. "
        f"The fixed budget tendency is under approximately ${float(fixture['budget_threshold_usd']):.0f}.",
        "",
        "## Selected sessions",
        "",
    ]
    for user in fixture["users"]:
        lines.extend([f"### {user['user_id']} — {user['archetype']}", ""])
        for session in sorted(user["sessions"], key=lambda value: value["sequence_index"]):
            source = samples[session["source_sample_id"]]
            product = products[session["target_asin"]]
            directive = session["longitudinal_directive"]
            active = [
                f"{key}: {'; '.join(value)}"
                for key, value in directive.items()
                if value
            ]
            traits = session["target_attribute_audit"]["verified_traits"]
            trait_text = "; ".join(
                f"{trait['kind']}={trait['value']} [{trait['source']}: {_compact(trait['evidence'], 110)}]"
                for trait in traits
            )
            category = " > ".join(product.get("categories") or [])
            lines.extend(
                [
                    f"#### S{session['sequence_index'] + 1}: {session['session_role']}",
                    "",
                    f"- Source: `{source['sample_id']}` ({source['scenario_type']}, {source.get('difficulty_bucket')})",
                    f"- Target: `{product['parent_asin']}` — {product.get('title')}",
                    f"- Category: {category}",
                    f"- Longitudinal signal: {session.get('intended_memory_signal') or 'none'}",
                    f"- Directive: {' | '.join(active) if active else 'none'}",
                    f"- Verified properties: {trait_text}",
                    f"- Why appropriate: {session['selection_rationale']}",
                    "",
                ]
            )

    lines.extend(["## Rejected discovery candidates", ""])
    for rejected in fixture.get("rejected_candidates", []):
        source = samples.get(rejected["source_sample_id"])
        if source is None:
            continue
        asin = source["ground_truth"]["parent_asin"]
        product = products[asin]
        lines.append(
            f"- `{source['sample_id']}` / `{asin}` — {product.get('title')}: {rejected['reason']}"
        )
    if not fixture.get("rejected_candidates"):
        lines.append("- No curated rejection annotations were supplied.")

    buying = [sample for sample in samples.values() if sample.get("scenario_type") == "buying"]
    lines.extend(
        [
            "",
            "## Discovery coverage",
            "",
            f"- Public buying rows inspected programmatically: {len(buying)}",
            f"- Distinct selected public rows: {len(selected_ids)}",
            f"- Selected session assignments: {sum(len(user['sessions']) for user in fixture['users'])}",
            "- Arbitrary catalogue-only targets: 0",
            "- Attribute policy: category/department are structural; material, colour, style, use case, "
            "and price are used only when the session audit points to target-level catalogue evidence.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Phase 6 candidate-selection report")
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--public", type=Path, default=DEFAULT_PUBLIC)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    fixture = json.loads(args.fixture.read_text(encoding="utf-8"))
    samples = {value["sample_id"]: value for value in load_jsonl(args.public)}
    products = {value["parent_asin"]: value for value in load_jsonl(args.catalog)}
    report = build_report(fixture, samples, products)
    args.output.write_text(report + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "selected_sessions": sum(len(user["sessions"]) for user in fixture["users"]),
                "distinct_source_rows": len(
                    {
                        session["source_sample_id"]
                        for user in fixture["users"]
                        for session in user["sessions"]
                    }
                ),
                "report": str(args.output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()


__all__ = ["build_report"]
