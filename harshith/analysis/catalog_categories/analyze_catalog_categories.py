#!/usr/bin/env python3
"""Extract product category paths and build exact occurrence/hierarchy summaries."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


HERE = Path(__file__).resolve().parent
CATALOG = HERE.parents[1] / "data" / "catalog.jsonl"


def clean_categories(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def main() -> None:
    label_counts: Counter[str] = Counter()
    depth_label_counts: dict[int, Counter[str]] = defaultdict(Counter)
    prefix_counts: Counter[tuple[str, ...]] = Counter()
    terminal_counts: Counter[tuple[str, ...]] = Counter()
    product_rows: list[dict[str, object]] = []
    depth_counts: Counter[int] = Counter()

    with CATALOG.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            product = json.loads(line)
            categories = clean_categories(product.get("categories"))
            depth_counts[len(categories)] += 1
            for depth, label in enumerate(categories, 1):
                label_counts[label] += 1
                depth_label_counts[depth][label] += 1
                prefix_counts[tuple(categories[:depth])] += 1
            if categories:
                terminal_counts[tuple(categories)] += 1
            product_rows.append(
                {
                    "line_number": line_number,
                    "parent_asin": product.get("parent_asin", ""),
                    "title": product.get("title", ""),
                    "category_depth": len(categories),
                    "category_path": " > ".join(categories),
                    "categories_json": json.dumps(categories, ensure_ascii=False),
                }
            )

    with (HERE / "product_categories.csv").open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=product_rows[0].keys())
        writer.writeheader()
        writer.writerows(product_rows)

    with (HERE / "category_label_occurrences.csv").open("w", newline="", encoding="utf-8") as output:
        fields = ["category", "occurrences", "depths_seen", "depth_occurrences_json"]
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for label, count in sorted(label_counts.items(), key=lambda item: (-item[1], item[0].casefold())):
            by_depth = {
                str(depth): counts[label]
                for depth, counts in sorted(depth_label_counts.items())
                if counts[label]
            }
            writer.writerow(
                {
                    "category": label,
                    "occurrences": count,
                    "depths_seen": ",".join(by_depth),
                    "depth_occurrences_json": json.dumps(by_depth, separators=(",", ":")),
                }
            )

    with (HERE / "category_hierarchy_nodes.csv").open("w", newline="", encoding="utf-8") as output:
        fields = ["path", "depth", "parent_path", "category", "product_count", "terminal_product_count"]
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for path, count in sorted(prefix_counts.items(), key=lambda item: (len(item[0]), item[0])):
            writer.writerow(
                {
                    "path": " > ".join(path),
                    "depth": len(path),
                    "parent_path": " > ".join(path[:-1]),
                    "category": path[-1],
                    "product_count": count,
                    "terminal_product_count": terminal_counts[path],
                }
            )

    children: dict[tuple[str, ...], list[tuple[str, ...]]] = defaultdict(list)
    for path in prefix_counts:
        children[path[:-1]].append(path)

    def node(path: tuple[str, ...]) -> dict[str, object]:
        result: dict[str, object] = {
            "name": path[-1] if path else "All products",
            "count": prefix_counts[path] if path else len(product_rows),
        }
        direct = terminal_counts[path]
        if direct:
            result["direct_count"] = direct
        child_nodes = [node(child) for child in sorted(children[path], key=lambda p: (-prefix_counts[p], p[-1].casefold()))]
        if child_nodes:
            result["children"] = child_nodes
        return result

    tree = node(())
    with (HERE / "category_tree.json").open("w", encoding="utf-8") as output:
        json.dump(tree, output, ensure_ascii=False, separators=(",", ":"))
        output.write("\n")

    summary = {
        "catalog_products": len(product_rows),
        "products_with_categories": sum(count for depth, count in depth_counts.items() if depth > 0),
        "unique_category_labels": len(label_counts),
        "unique_full_paths": len(terminal_counts),
        "unique_hierarchy_nodes": len(prefix_counts),
        "maximum_depth": max(depth_counts),
        "depth_distribution": dict(sorted(depth_counts.items())),
        "root_category_counts": {
            path[0]: count for path, count in prefix_counts.items() if len(path) == 1
        },
    }
    with (HERE / "summary.json").open("w", encoding="utf-8") as output:
        json.dump(summary, output, ensure_ascii=False, indent=2)
        output.write("\n")


if __name__ == "__main__":
    main()
