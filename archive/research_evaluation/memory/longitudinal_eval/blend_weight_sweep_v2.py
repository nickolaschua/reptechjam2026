"""Staged, mode-conditioned frozen-vector blend-weight experiment.

The experiment is deliberately sequential: Buying is swept and locked before
Browsing is inspected.  It imports only offline analysis helpers and reads the
immutable v2 forensic bundle plus its hash-matched catalogue.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

try:  # Package import in tests; direct import when this file is run as a script.
    from .threshold_calibration_v2 import (
        DEFAULT_CATALOG, DEFAULT_INPUT, RELATIONS, SCENARIOS, load_catalog, read_json,
        relevant_rank, sha256_file, slice_metrics, stable_rank, subsets, write_csv, write_json,
    )
except ImportError:
    from threshold_calibration_v2 import (
        DEFAULT_CATALOG, DEFAULT_INPUT, RELATIONS, SCENARIOS, load_catalog, read_json,
        relevant_rank, sha256_file, slice_metrics, stable_rank, subsets, write_csv, write_json,
    )


HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / "results" / "v2_blend_weight_sweep"
FIXED_GATE_THRESHOLD = 0.20
BUYING_WEIGHTS = (0.00, 0.025, 0.05, 0.075, 0.10, 0.125, 0.15, 0.175, 0.20, 0.25, 0.30, 0.40, 0.50)
BROWSING_WEIGHTS = (0.00, 0.025, 0.05, 0.075, 0.10, 0.125, 0.15, 0.175, 0.20, 0.25, 0.30, 0.40,
                    0.50, 0.60, 0.70, 0.80)
MIN_STABLE_POINTS = 3


def array_hash(array: np.ndarray) -> str:
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def reconstruct(
    probes: Sequence[Mapping[str, Any]], arrays: Mapping[str, np.ndarray], asins: np.ndarray,
    buying_memory_weight: float, browsing_memory_weight: float, gate_threshold: float,
) -> list[dict[str, Any]]:
    """Reconstruct one mode-conditioned system without mutating frozen arrays."""
    rows: list[dict[str, Any]] = []
    for probe in probes:
        refs = probe["array_references"]
        s1 = arrays[refs["s1"]["npz_key"]]
        s2 = arrays[refs["s2"]["npz_key"]]
        mask = arrays[refs["eligibility"]["npz_key"]]
        b = buying_memory_weight if probe["buyer_mode"] == "Buying" else browsing_memory_weight
        scores = (1.0 - b) * s1 + b * s2 if float(probe["gate_cosine"]) >= gate_threshold else s1
        ranked = stable_rank(scores, mask, asins)
        raw, penalized = relevant_rank(ranked, set(probe["relevant_asins"]), asins)
        rows.append({"session_id": probe["session_id"], "scenario_class": probe["scenario_class"],
                     "buyer_mode": probe["buyer_mode"], "memory_relation": probe["memory_relation"],
                     "gate_cosine": probe["gate_cosine"],
                     "gate_passed": float(probe["gate_cosine"]) >= gate_threshold,
                     "m0_rank": probe["m0_relevant_rank"], "m0_penalized_rank": probe["m0_penalized_rank"],
                     "m3_rank": raw, "m3_penalized_rank": penalized})
    return rows


def persisted_original_rows(probes: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [{"session_id": probe["session_id"], "scenario_class": probe["scenario_class"],
             "buyer_mode": probe["buyer_mode"], "memory_relation": probe["memory_relation"],
             "gate_cosine": probe["gate_cosine"], "gate_passed": probe["gate_passed"],
             "m0_rank": probe["m0_relevant_rank"], "m0_penalized_rank": probe["m0_penalized_rank"],
             "m3_rank": probe["m3_relevant_rank"], "m3_penalized_rank": probe["m3_penalized_rank"]}
            for probe in probes]


def gate_counts(probes: Sequence[Mapping[str, Any]], mode: str, threshold: float) -> dict[str, int]:
    selected = [probe for probe in probes if probe["buyer_mode"] == mode]
    counts: dict[str, int] = {}
    for relation in RELATIONS:
        relation_rows = [probe for probe in selected if probe["memory_relation"] == relation]
        counts[f"{relation.lower()}_gate_total"] = len(relation_rows)
        counts[f"{relation.lower()}_gate_pass"] = sum(float(probe["gate_cosine"]) >= threshold
                                                         for probe in relation_rows)
    return counts


def buying_sweep_row(weight: float, metrics: Mapping[str, Mapping[str, Any]], counts: Mapping[str, int]) -> dict[str, Any]:
    buying = metrics["Buying"]
    row = {"b_buying": weight, "a_buying": 1.0 - weight,
           "longitudinal_positive_delta": metrics["LONGITUDINAL_POSITIVE"]["mean_rr_delta"],
           "memory_irrelevant_delta": metrics["MEMORY_IRRELEVANT"]["mean_rr_delta"],
           "current_override_delta": metrics["CURRENT_OVERRIDE"]["mean_rr_delta"],
           "buying_mrr_delta": buying["mean_rr_delta"], "buying_help_rate": buying["help_rate"],
           "buying_harm_rate": buying["harm_rate"], "buying_unchanged_rate": buying["unchanged_rate"], **counts}
    row["strict_pass"] = (row["longitudinal_positive_delta"] > 0
                          and row["memory_irrelevant_delta"] >= 0
                          and row["current_override_delta"] >= 0
                          and row["buying_mrr_delta"] >= 0)
    return row


def browsing_sweep_row(weight: float, metrics: Mapping[str, Mapping[str, Any]], counts: Mapping[str, int]) -> dict[str, Any]:
    browsing = metrics["Browsing"]
    row = {"b_browsing": weight, "a_browsing": 1.0 - weight,
           "browsing_personalization_delta": metrics["BROWSING_PERSONALIZATION"]["mean_rr_delta"],
           "browsing_mrr_delta": browsing["mean_rr_delta"], "browsing_help_rate": browsing["help_rate"],
           "browsing_harm_rate": browsing["harm_rate"], "browsing_unchanged_rate": browsing["unchanged_rate"],
           "available_browsing_safety_slices": "none_in_frozen_fixture", **counts}
    row["strict_pass"] = row["browsing_personalization_delta"] > 0 and row["browsing_mrr_delta"] >= 0
    return row


def contiguous_regions(rows: Sequence[Mapping[str, Any]], weight_key: str) -> list[list[Mapping[str, Any]]]:
    regions: list[list[Mapping[str, Any]]] = []
    current: list[Mapping[str, Any]] = []
    for row in rows:
        if row["strict_pass"]:
            current.append(row)
        elif current:
            regions.append(current)
            current = []
    if current:
        regions.append(current)
    return [region for region in regions if len(region) >= MIN_STABLE_POINTS]


def lock_region(rows: Sequence[Mapping[str, Any]], weight_key: str) -> dict[str, Any] | None:
    """Select the widest valid run, then its tested interior midpoint."""
    regions = contiguous_regions(rows, weight_key)
    if not regions:
        return None
    region = sorted(regions, key=lambda run: (-len(run),
                                              -(float(run[-1][weight_key]) - float(run[0][weight_key])),
                                              float(run[0][weight_key])))[0]
    low, high = float(region[0][weight_key]), float(region[-1][weight_key])
    midpoint = (low + high) / 2.0
    interior = region[1:-1]
    assert interior
    selected = min(interior, key=lambda row: (abs(float(row[weight_key]) - midpoint), float(row[weight_key])))
    return {"lower": low, "upper": high, "point_count": len(region),
            "tested_values": [float(row[weight_key]) for row in region], "locked_weight": float(selected[weight_key]),
            "selection_rule": ("Choose the widest contiguous run of at least three strict-pass sweep points; "
                               "break region ties by lower endpoint; lock the tested interior point nearest the "
                               "endpoint midpoint, breaking point ties toward the lower weight. MRR is not a selector.")}


def flattened_system(label: str, metrics: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for name in ("OVERALL",) + SCENARIOS + ("Buying", "Browsing"):
        metric = metrics[name]
        output.append({"system": label, "slice": name, "count": metric["count"],
                       "m0_mrr": metric["m0"]["mrr"], "system_mrr": metric["m3"]["mrr"],
                       "mrr_delta_vs_m0": metric["mean_rr_delta"], "help_rate": metric["help_rate"],
                       "harm_rate": metric["harm_rate"], "unchanged_rate": metric["unchanged_rate"]})
    return output


def render_report(result: Mapping[str, Any]) -> str:
    lines = ["# Frozen v2 staged mode-conditioned blend-weight sweep", "",
             "## Experimental controls", "",
             f"- Fixed gate: `gate_cosine >= {result['fixed_gate_threshold']}` (the immutable frozen production threshold).",
             "- Buying and Browsing use independent weights. No common weight and no two-dimensional grid were evaluated.",
             "- Sequence: Buying sweep → Buying lock → Browsing sweep → Browsing lock → final evaluation.",
             "- Memory vectors, EWMA, embeddings, fixture, catalogue, eligibility masks, and retrieval behavior are frozen.",
             f"- Broad stability requires at least {MIN_STABLE_POINTS} adjacent strict-pass sweep points, ensuring a tested interior lock candidate.", "",
             "## Stage 1 — Buying", "",
             "| b | a | LP delta | MI delta | CO delta | Buying delta | Help | Harm | Unchanged | Rel pass/total | Irrel pass/total | Strict |",
             "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for row in result["stage_1_buying"]["sweep"]:
        lines.append(f"| {row['b_buying']:.3f} | {row['a_buying']:.3f} | {row['longitudinal_positive_delta']:+.9f} | "
                     f"{row['memory_irrelevant_delta']:+.9f} | {row['current_override_delta']:+.9f} | "
                     f"{row['buying_mrr_delta']:+.9f} | {row['buying_help_rate']:.3f} | {row['buying_harm_rate']:.3f} | "
                     f"{row['buying_unchanged_rate']:.3f} | {row['relevant_gate_pass']}/{row['relevant_gate_total']} | "
                     f"{row['irrelevant_gate_pass']}/{row['irrelevant_gate_total']} | {row['strict_pass']} |")
    stage1 = result["stage_1_buying"]
    if stage1["lock"] is None:
        lines += ["", f"**Stage 1 outcome: STOP — {stage1['failure_classification']}**", "",
                  stage1["failure_detail"], "",
                  "Stages 2 and 3 were not run, as required by the staged protocol.", ""]
        return "\n".join(lines)
    lock1 = stage1["lock"]
    lines += ["", f"Stable region: `{lock1['tested_values']}`. Locked `b_buying={lock1['locked_weight']}` and `a_buying={1-lock1['locked_weight']}`.", "",
              f"Lock rule: {lock1['selection_rule']}", "", "## Stage 2 — Browsing", "",
              "| b | a | BP delta | Browsing delta | Help | Harm | Unchanged | Rel pass/total | Original b=0.80 | Strict |",
              "|---:|---:|---:|---:|---:|---:|---:|---:|---|---|"]
    for row in result["stage_2_browsing"]["sweep"]:
        lines.append(f"| {row['b_browsing']:.3f} | {row['a_browsing']:.3f} | {row['browsing_personalization_delta']:+.9f} | "
                     f"{row['browsing_mrr_delta']:+.9f} | {row['browsing_help_rate']:.3f} | {row['browsing_harm_rate']:.3f} | "
                     f"{row['browsing_unchanged_rate']:.3f} | {row['relevant_gate_pass']}/{row['relevant_gate_total']} | "
                     f"{row['b_browsing'] == .80} | {row['strict_pass']} |")
    stage2 = result["stage_2_browsing"]
    if stage2["lock"] is None:
        lines += ["", f"**Stage 2 outcome: STOP — {stage2['failure_classification']}**", "",
                  stage2["failure_detail"], "", "Stage 3 was not run, as required by the staged protocol.", ""]
        return "\n".join(lines)
    lock2 = stage2["lock"]
    lines += ["", f"Stable region: `{lock2['tested_values']}`. Locked `b_browsing={lock2['locked_weight']}` and `a_browsing={1-lock2['locked_weight']}`.", "",
              f"Lock rule: {lock2['selection_rule']}", "", "## Stage 3 — Locked system", "",
              f"Strict operating-region result: **{result['stage_3_final']['strict_pass']}**.", "",
              "| System | Slice | M0 MRR | System MRR | Delta vs M0 | Help | Harm | Unchanged |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in result["stage_3_final"]["comparison"]:
        lines.append(f"| {row['system']} | {row['slice']} | {row['m0_mrr']:.9f} | {row['system_mrr']:.9f} | "
                     f"{row['mrr_delta_vs_m0']:+.9f} | {row['help_rate']:.3f} | {row['harm_rate']:.3f} | {row['unchanged_rate']:.3f} |")
    return "\n".join(lines) + "\n"


def run(input_dir: Path, output_dir: Path, catalog_path: Path) -> dict[str, Any]:
    manifest = read_json(input_dir / "manifest.json")
    frozen_before = {name: sha256_file(input_dir / name) for name in manifest["artifact_hashes"]}
    assert frozen_before == manifest["artifact_hashes"]
    records = [json.loads(line) for line in (input_dir / "sessions.jsonl").read_text(encoding="utf-8").splitlines() if line]
    probes = [row for row in records if row["record_type"] == "probe"]
    assert len(probes) == 40
    catalog_asins, _ = load_catalog(catalog_path, manifest["catalog_sha256"])
    with np.load(input_dir / "vectors.npz", allow_pickle=False) as loaded:
        arrays = {key: loaded[key] for key in loaded.files}
    asins = arrays["catalog_asins"]
    assert asins.tolist() == catalog_asins and len(asins) == 50_000
    protected = {key: array_hash(value) for key, value in arrays.items()
                 if key == "catalog_asins" or key.endswith(("_probe_s1", "_probe_s2", "_probe_eligibility"))}

    buying_counts = gate_counts(probes, "Buying", FIXED_GATE_THRESHOLD)
    stage1_rows = []
    for b_buying in BUYING_WEIGHTS:
        rows = reconstruct(probes, arrays, asins, b_buying, .80, FIXED_GATE_THRESHOLD)
        metrics = {name: slice_metrics(group) for name, group in subsets(rows).items()}
        stage1_rows.append(buying_sweep_row(b_buying, metrics, buying_counts))
    buying_lock = lock_region(stage1_rows, "b_buying")
    stage1: dict[str, Any] = {"sweep": stage1_rows, "lock": buying_lock}
    if buying_lock is None:
        passing = [row["b_buying"] for row in stage1_rows if row["strict_pass"]]
        stage1.update({"failure_classification": "NO_BROAD_STABLE_BUYING_REGION",
                       "failure_detail": (f"Strict-pass Buying weights were {passing}, but none formed at least "
                                          f"{MIN_STABLE_POINTS} adjacent sweep points. The result is too narrow "
                                          "to lock without selecting an edge or an isolated numerical pocket.")})

    result: dict[str, Any] = {"schema_version": "v2-staged-mode-blend-sweep-1",
                              "fixed_gate_threshold": FIXED_GATE_THRESHOLD,
                              "stable_region_minimum_adjacent_points": MIN_STABLE_POINTS,
                              "stage_1_buying": stage1, "stage_2_browsing": None, "stage_3_final": None,
                              "controls": {"joint_2d_grid_run": False, "common_mode_weight_forced": False,
                                           "buying_retuned_after_stage_1": False, "production_calls": 0,
                                           "production_behavior_changed": False},
                              "frozen_manifest": manifest}

    if buying_lock is not None:
        locked_buying = buying_lock["locked_weight"]
        browsing_counts = gate_counts(probes, "Browsing", FIXED_GATE_THRESHOLD)
        stage2_rows = []
        # This loop is intentionally reached only after Buying has been locked.
        for b_browsing in BROWSING_WEIGHTS:
            rows = reconstruct(probes, arrays, asins, locked_buying, b_browsing, FIXED_GATE_THRESHOLD)
            metrics = {name: slice_metrics(group) for name, group in subsets(rows).items()}
            stage2_rows.append(browsing_sweep_row(b_browsing, metrics, browsing_counts))
        browsing_lock = lock_region(stage2_rows, "b_browsing")
        stage2: dict[str, Any] = {"locked_buying_weight": locked_buying, "sweep": stage2_rows, "lock": browsing_lock,
                                  "original_browsing_reference": next(row for row in stage2_rows if row["b_browsing"] == .80)}
        if browsing_lock is None:
            passing = [row["b_browsing"] for row in stage2_rows if row["strict_pass"]]
            stage2.update({"failure_classification": "NO_BROAD_STABLE_BROWSING_REGION",
                           "failure_detail": (f"Strict-pass Browsing weights were {passing}, but none formed at least "
                                              f"{MIN_STABLE_POINTS} adjacent sweep points.")})
        result["stage_2_browsing"] = stage2

        if browsing_lock is not None:
            locked_browsing = browsing_lock["locked_weight"]
            locked_rows = reconstruct(probes, arrays, asins, locked_buying, locked_browsing, FIXED_GATE_THRESHOLD)
            locked_metrics = {name: slice_metrics(group) for name, group in subsets(locked_rows).items()}
            original_metrics = {name: slice_metrics(group) for name, group in subsets(persisted_original_rows(probes)).items()}
            comparison = flattened_system("locked_mode_conditioned", locked_metrics)
            comparison += flattened_system("original_frozen_M3_tau_0.20", original_metrics)
            strict = (locked_metrics["LONGITUDINAL_POSITIVE"]["mean_rr_delta"] > 0
                      and locked_metrics["BROWSING_PERSONALIZATION"]["mean_rr_delta"] > 0
                      and locked_metrics["MEMORY_IRRELEVANT"]["mean_rr_delta"] >= 0
                      and locked_metrics["CURRENT_OVERRIDE"]["mean_rr_delta"] >= 0
                      and locked_metrics["OVERALL"]["mean_rr_delta"] >= 0)
            result["stage_3_final"] = {"locked_weights": {"buying": {"a": 1-locked_buying, "b": locked_buying},
                                                            "browsing": {"a": 1-locked_browsing, "b": locked_browsing}},
                                       "strict_pass": strict, "comparison": comparison}

    assert all(array_hash(arrays[key]) == digest for key, digest in protected.items())
    assert {name: sha256_file(input_dir / name) for name in manifest["artifact_hashes"]} == frozen_before
    result["verification"] = {"frozen_artifact_hashes": frozen_before,
                              "catalog_sha256": manifest["catalog_sha256"], "catalog_rows_aligned": 50_000,
                              "protected_arrays_unmutated": True, "sequential_lock_order_enforced": True}
    output_dir.mkdir(parents=True, exist_ok=True)
    if result["stage_2_browsing"] is None:
        (output_dir / "stage_2_browsing.csv").unlink(missing_ok=True)
    if result["stage_3_final"] is None:
        (output_dir / "stage_3_comparison.csv").unlink(missing_ok=True)
    write_json(output_dir / "results.json", result)
    write_csv(output_dir / "stage_1_buying.csv", stage1_rows)
    if result["stage_2_browsing"] is not None:
        write_csv(output_dir / "stage_2_browsing.csv", result["stage_2_browsing"]["sweep"])
    if result["stage_3_final"] is not None:
        write_csv(output_dir / "stage_3_comparison.csv", result["stage_3_final"]["comparison"])
    (output_dir / "report.md").write_text(render_report(result), encoding="utf-8")
    output_hashes = {path.name: sha256_file(path) for path in sorted(output_dir.iterdir())
                     if path.is_file() and path.name != "manifest.json"}
    write_json(output_dir / "manifest.json", {"schema_version": result["schema_version"],
                                               "frozen_artifact_hashes": frozen_before,
                                               "catalog_sha256": manifest["catalog_sha256"],
                                               "output_hashes": output_hashes})
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    args = parser.parse_args()
    result = run(args.input.resolve(), args.output.resolve(), args.catalog.resolve())
    if result["stage_1_buying"]["lock"] is None:
        print(result["stage_1_buying"]["failure_classification"])
    elif result["stage_2_browsing"]["lock"] is None:
        print(result["stage_2_browsing"]["failure_classification"])
    else:
        print(f"Final strict pass: {result['stage_3_final']['strict_pass']}")
    print(f"Wrote {args.output.resolve()}")


if __name__ == "__main__":
    main()
