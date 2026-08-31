"""Offline relevance-threshold calibration for the immutable v2 bundle.

This module deliberately imports no production agent, memory, embedding, or LLM
code.  It reconstructs rankings exclusively from the frozen forensic artifacts
and a catalogue whose SHA-256 matches the frozen manifest.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


HERE = Path(__file__).resolve().parent
DEFAULT_INPUT = HERE / "results" / "v2"
DEFAULT_OUTPUT = HERE / "results" / "v2_threshold_calibration"
DEFAULT_CATALOG = HERE.parents[2] / "techjam-conversational-search" / "data" / "catalog.jsonl"

# The preregistered 17-point sweep.  The diagnostic cutoffs below additionally
# evaluate every unique observed cosine, so the coarse sweep is not used to
# claim the best possible classifier cutoff.
THRESHOLDS = tuple(round(i * 0.05, 2) for i in range(17))
SCENARIOS = (
    "LONGITUDINAL_POSITIVE",
    "MEMORY_IRRELEVANT",
    "CURRENT_OVERRIDE",
    "BROWSING_PERSONALIZATION",
)
SLICE_NAMES = ("OVERALL",) + SCENARIOS + ("Buying", "Browsing")
RELATIONS = ("RELEVANT", "IRRELEVANT", "CONFLICTING")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(value.tobytes(order="C")).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns: list[str] = []
    for row in rows:
        columns.extend(key for key in row if key not in columns)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False, sort_keys=True)
                             if isinstance(value, (dict, list)) else value for key, value in row.items()})


def load_catalog(path: Path, expected_hash: str) -> tuple[list[str], dict[str, dict[str, Any]]]:
    actual = sha256_file(path)
    assert actual == expected_hash, f"catalogue hash mismatch: {actual} != {expected_hash}"
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    asins = [str(row["parent_asin"]) for row in rows]
    assert len(rows) == 50_000 and len(set(asins)) == 50_000
    return asins, {asin: row for asin, row in zip(asins, rows)}


def product_metadata(row: Mapping[str, Any]) -> dict[str, Any]:
    return {key: row.get(key) for key in (
        "parent_asin", "title", "categories", "features", "description", "price",
        "average_rating", "rating_number", "store", "details",
    )}


def stable_rank(scores: np.ndarray, mask: np.ndarray, asins: np.ndarray) -> np.ndarray:
    eligible = np.flatnonzero(mask)
    order = np.lexsort((asins[eligible], -scores[eligible]))
    return eligible[order]


def relevant_rank(ranked_rows: np.ndarray, relevant: set[str], asins: np.ndarray) -> tuple[int | None, int]:
    raw = next((rank for rank, row in enumerate(ranked_rows, 1) if str(asins[row]) in relevant), None)
    return raw, len(ranked_rows) + 1 if raw is None else raw


def rr(rank: int | None) -> float:
    return 0.0 if rank is None else 1.0 / rank


def arm_metrics(raw_ranks: Sequence[int | None], penalties: Sequence[int]) -> dict[str, Any]:
    return {
        "mrr": statistics.fmean(rr(rank) for rank in raw_ranks),
        "hit_at_1": statistics.fmean(rank is not None and rank <= 1 for rank in raw_ranks),
        "hit_at_5": statistics.fmean(rank is not None and rank <= 5 for rank in raw_ranks),
        "hit_at_10": statistics.fmean(rank is not None and rank <= 10 for rank in raw_ranks),
        "mean_penalized_rank": statistics.fmean(penalties),
        "median_penalized_rank": statistics.median(penalties),
    }


def slice_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    m0_raw = [row["m0_rank"] for row in rows]
    m3_raw = [row["m3_rank"] for row in rows]
    deltas = [rr(right) - rr(left) for left, right in zip(m0_raw, m3_raw)]
    result = {
        "count": len(rows),
        "m0": arm_metrics(m0_raw, [int(row["m0_penalized_rank"]) for row in rows]),
        "m3": arm_metrics(m3_raw, [int(row["m3_penalized_rank"]) for row in rows]),
        "mean_rr_delta": statistics.fmean(deltas),
        "help_rate": statistics.fmean(delta > 0 for delta in deltas),
        "harm_rate": statistics.fmean(delta < 0 for delta in deltas),
        "unchanged_rate": statistics.fmean(delta == 0 for delta in deltas),
    }
    assert abs(result["help_rate"] + result["harm_rate"] + result["unchanged_rate"] - 1.0) < 1e-12
    return result


def subsets(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    return {
        "OVERALL": list(rows),
        **{name: [row for row in rows if row["scenario_class"] == name] for name in SCENARIOS},
        **{name: [row for row in rows if row["buyer_mode"] == name] for name in ("Buying", "Browsing")},
    }


def distribution(values: Sequence[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": len(values), "min": float(array.min()), "p10": float(np.quantile(array, .10, method="linear")),
        "p25": float(np.quantile(array, .25, method="linear")), "median": float(np.quantile(array, .50, method="linear")),
        "p75": float(np.quantile(array, .75, method="linear")), "p90": float(np.quantile(array, .90, method="linear")),
        "max": float(array.max()), "mean": float(array.mean()), "sample_stddev": float(array.std(ddof=1)),
    }


def classifier_metrics(probes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    binary = [row for row in probes if row["memory_relation"] in ("RELEVANT", "IRRELEVANT")]
    positives = [float(row["gate_cosine"]) for row in binary if row["memory_relation"] == "RELEVANT"]
    negatives = [float(row["gate_cosine"]) for row in binary if row["memory_relation"] == "IRRELEVANT"]
    auc = statistics.fmean((p > n) + 0.5 * (p == n) for p in positives for n in negatives)
    ranked = sorted(binary, key=lambda row: (-float(row["gate_cosine"]), str(row["session_id"])))
    precisions: list[float] = []
    true_positives = 0
    for index, row in enumerate(ranked, 1):
        if row["memory_relation"] == "RELEVANT":
            true_positives += 1
            precisions.append(true_positives / index)
    average_precision = statistics.fmean(precisions)

    def confusion(tau: float) -> dict[str, Any]:
        tp = sum(row["memory_relation"] == "RELEVANT" and row["gate_cosine"] >= tau for row in binary)
        fp = sum(row["memory_relation"] == "IRRELEVANT" and row["gate_cosine"] >= tau for row in binary)
        fn, tn = len(positives) - tp, len(negatives) - fp
        return {"threshold": tau, "tp": tp, "fp": fp, "tn": tn, "fn": fn,
                "tpr": tp / len(positives), "fpr": fp / len(negatives),
                "precision": tp / (tp + fp) if tp + fp else 0.0}

    cutoffs = [confusion(score) for score in sorted({float(row["gate_cosine"]) for row in binary}, reverse=True)]
    for row in cutoffs:
        row["youden_j"] = row["tpr"] - row["fpr"]
    best_j = max(row["youden_j"] for row in cutoffs)
    youden = max((row for row in cutoffs if row["youden_j"] == best_j), key=lambda row: row["threshold"])
    feasible = [row for row in cutoffs if row["tpr"] >= .90 and row["fpr"] <= .10]
    return {"positive_relation": "RELEVANT", "negative_relation": "IRRELEVANT",
            "excluded_relation": "CONFLICTING", "roc_auc": auc, "pr_auc_average_precision": average_precision,
            "tau_0_20": confusion(.20), "youden_j_cutoff": youden,
            "recall_90_fpr_10_cutoffs": feasible, "exact_score_cutoffs": cutoffs}


def flatten_metric(tau: float | str, slice_name: str, metric: Mapping[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {"threshold": tau, "slice": slice_name, "count": metric["count"]}
    for arm in ("m0", "m3"):
        row.update({f"{arm}_{key}": value for key, value in metric[arm].items()})
    row.update({key: metric[key] for key in ("mean_rr_delta", "help_rate", "harm_rate", "unchanged_rate")})
    return row


def assert_published_metric_parity(actual: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    """Require every published relevant-set metric to match reconstruction."""
    assert actual["count"] == expected["count"]
    for arm in ("m0", "m3"):
        for local, frozen in (
            ("mrr", "relevant_set_mrr"), ("hit_at_1", "relevant_set_hit_at_1"),
            ("hit_at_5", "relevant_set_hit_at_5"), ("hit_at_10", "relevant_set_hit_at_10"),
            ("mean_penalized_rank", "relevant_set_mean_penalized_rank"),
            ("median_penalized_rank", "relevant_set_median_penalized_rank"),
        ):
            assert abs(float(actual[arm][local]) - float(expected[arm][frozen])) < 1e-15
    for local, frozen in (
        ("mean_rr_delta", "relevant_set_mean_reciprocal_rank_delta"),
        ("help_rate", "relevant_set_help_rate"), ("harm_rate", "relevant_set_harm_rate"),
        ("unchanged_rate", "relevant_set_unchanged_rate"),
    ):
        assert abs(float(actual[local]) - float(expected[frozen])) < 1e-15


def reconstruct(
    probes: Sequence[Mapping[str, Any]], arrays: Mapping[str, np.ndarray], asins: np.ndarray, tau: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    activation = Counter()
    for probe in probes:
        refs = probe["array_references"]
        s1 = arrays[refs["s1"]["npz_key"]]
        s2 = arrays[refs["s2"]["npz_key"]]
        mask = arrays[refs["eligibility"]["npz_key"]]
        passed = float(probe["gate_cosine"]) >= tau
        activation[probe["memory_relation"]] += int(passed)
        if passed:
            current = .8 if probe["buyer_mode"] == "Buying" else .2
            memory = 1.0 - current
            scores = current * s1 + memory * s2
        else:
            scores = s1
        ranked = stable_rank(scores, mask, asins)
        raw, penalty = relevant_rank(ranked, set(probe["relevant_asins"]), asins)
        rows.append({"session_id": probe["session_id"], "scenario_class": probe["scenario_class"],
                     "buyer_mode": probe["buyer_mode"], "memory_relation": probe["memory_relation"],
                     "gate_cosine": probe["gate_cosine"], "gate_passed": passed,
                     "m0_rank": probe["m0_relevant_rank"], "m0_penalized_rank": probe["m0_penalized_rank"],
                     "m3_rank": raw, "m3_penalized_rank": penalty})
    relation_counts = Counter(row["memory_relation"] for row in probes)
    rates = {relation: activation[relation] / relation_counts[relation] for relation in RELATIONS}
    return rows, {"relevant_activation_rate": rates["RELEVANT"],
                  "irrelevant_activation_rate": rates["IRRELEVANT"],
                  "conflicting_activation_rate": rates["CONFLICTING"]}


def classify_contamination(probe: Mapping[str, Any]) -> tuple[str, str]:
    query_terms = {term.casefold() for term in probe["update_scope_annotations"].get("session_specific", [])}
    texts = [str(item["text"]).casefold() for item in probe["contributing_memory_lineage"]]
    contaminated = any(term and term in text for term in query_terms for text in texts)
    if contaminated:
        return "B", "Embedded update text contains the probe-domain/session-specific query annotation."
    return "A", ("Embedded update text contains no probe-domain or temporary/session-specific query content; "
                 "the cosine and rank movement arise from cross-domain trait semantics and promoted catalogue products.")


def forensic_rows(
    probes: Sequence[Mapping[str, Any]], setup: Sequence[Mapping[str, Any]], arrays: Mapping[str, np.ndarray],
    asins: np.ndarray, catalog: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    setup_by_user: dict[str, list[Mapping[str, Any]]] = {}
    for row in setup:
        setup_by_user.setdefault(str(row["user_id"]), []).append(row)
    asin_to_row = {str(asin): index for index, asin in enumerate(asins)}
    output: list[dict[str, Any]] = []
    for probe in probes:
        if probe["scenario_class"] != "MEMORY_IRRELEVANT" or probe["gate_cosine"] < .20:
            continue
        refs = probe["array_references"]
        s1, s2 = arrays[refs["s1"]["npz_key"]], arrays[refs["s2"]["npz_key"]]
        current, memory = (.8, .2) if probe["buyer_mode"] == "Buying" else (.2, .8)
        s3 = current * s1 + memory * s2
        label, reason = classify_contamination(probe)
        persisted = probe["overtakers"]
        overtakers = []
        for item in persisted:
            row_index = asin_to_row[str(item["asin"])]
            overtakers.append({**item, "metadata": product_metadata(catalog[str(item["asin"])]),
                               "s1": float(s1[row_index]), "s2": float(s2[row_index]), "s3": float(s3[row_index])})
        setup_lineage = []
        for item in sorted(setup_by_user[probe["user_id"]], key=lambda row: int(row["sequence_index"])):
            setup_lineage.append({"session_id": item["session_id"], "update_id": item["update_id"],
                                  "sequence_index": item["sequence_index"], "update_text": item["update_text"],
                                  "fact_scope_annotations": item["fact_scope_annotations"],
                                  "contributing_sequences": item["contributing_sequences"],
                                  "vector_references": item["array_references"]})
        output.append({"session_id": probe["session_id"], "user_id": probe["user_id"],
                       "classification": label, "classification_reason": reason,
                       "gate_cosine": probe["gate_cosine"], "query": probe["scripted_turns"],
                       "canonical_parsed_state": probe["canonical_parsed_state"],
                       "update_scope_annotations": probe["update_scope_annotations"],
                       "contributing_memory_lineage": probe["contributing_memory_lineage"],
                       "setup_textual_vector_lineage": setup_lineage,
                       "probe_vector_references": {key: refs[key] for key in ("v1", "v2", "s1", "s2", "s3", "eligibility")},
                       "target_asin": probe["target_asin"], "target_metadata": product_metadata(catalog[probe["target_asin"]]),
                       "relevant_asins": probe["relevant_asins"],
                       "relevant_metadata": [product_metadata(catalog[asin]) for asin in probe["relevant_asins"]],
                       "m0_relevant_rank": probe["m0_relevant_rank"], "m3_relevant_rank": probe["m3_relevant_rank"],
                       "m0_rr": rr(probe["m0_relevant_rank"]), "m3_rr": rr(probe["m3_relevant_rank"]),
                       "rr_delta": probe["relevant_rr_delta"], "persisted_overtaker_count": len(persisted),
                       "overtakers": overtakers})
    return output


def verify_bundle(root: Path, manifest: Mapping[str, Any], records: Sequence[Mapping[str, Any]],
                  arrays: Mapping[str, np.ndarray], asins: np.ndarray) -> dict[str, Any]:
    artifact_hashes = {name: sha256_file(root / name) for name in manifest["artifact_hashes"]}
    assert artifact_hashes == manifest["artifact_hashes"]
    probes = [row for row in records if row["record_type"] == "probe"]
    setup = [row for row in records if row["record_type"] == "setup_update"]
    assert len(probes) == 40 and len(setup) == 80
    assert Counter(row["scenario_class"] for row in probes) == Counter({name: 10 for name in SCENARIOS})
    assert Counter(row["buyer_mode"] for row in probes) == Counter({"Buying": 30, "Browsing": 10})
    assert Counter(row["memory_relation"] for row in probes) == Counter({"RELEVANT": 20, "IRRELEVANT": 10, "CONFLICTING": 10})
    assert len(asins) == len(set(asins.tolist())) == 50_000
    for record in records:
        for ref in record["array_references"].values():
            value = arrays[ref["npz_key"]]
            assert list(value.shape) == ref["shape"] and str(value.dtype) == ref["dtype"]
            assert array_sha256(value) == ref["sha256"]
    return {"artifact_hashes_verified": artifact_hashes, "probe_count": 40, "setup_record_count": 80,
            "scenario_counts": dict(Counter(row["scenario_class"] for row in probes)),
            "buyer_mode_counts": dict(Counter(row["buyer_mode"] for row in probes)),
            "relation_counts": dict(Counter(row["memory_relation"] for row in probes)), "catalog_rows_aligned": 50_000}


def metric_markdown(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    lines = ["| Threshold | Slice | n | M0 MRR | M3 MRR | Delta | M0 H@1/5/10 | M3 H@1/5/10 | M0 mean/median rank | M3 mean/median rank | Help/Harm/Unchanged |",
             "|---:|---|---:|---:|---:|---:|---|---|---|---|---|"]
    for row in rows:
        lines.append(f"| {row['threshold']} | {row['slice']} | {row['count']} | {row['m0_mrr']:.6f} | {row['m3_mrr']:.6f} | {row['mean_rr_delta']:+.6f} | "
                     f"{row['m0_hit_at_1']:.3f}/{row['m0_hit_at_5']:.3f}/{row['m0_hit_at_10']:.3f} | "
                     f"{row['m3_hit_at_1']:.3f}/{row['m3_hit_at_5']:.3f}/{row['m3_hit_at_10']:.3f} | "
                     f"{row['m0_mean_penalized_rank']:.1f}/{row['m0_median_penalized_rank']:.1f} | "
                     f"{row['m3_mean_penalized_rank']:.1f}/{row['m3_median_penalized_rank']:.1f} | "
                     f"{row['help_rate']:.3f}/{row['harm_rate']:.3f}/{row['unchanged_rate']:.3f} |")
    return lines


def render_report(result: Mapping[str, Any]) -> str:
    cls = result["classification"]
    lines = ["# Frozen v2 relevance-threshold calibration", "",
             f"**Required conclusion: {result['conclusion']}**", "",
             f"**Unimplemented follow-up experiment:** {result['follow_up_experiment']}", "",
             "## Frozen-input verification", "",
             "All four v2 artifact hashes and the catalogue hash match the frozen manifest. Offline rank reconstruction passed. "
             "No embedding or LLM calls were made, and the source bundle was unchanged.", "",
             f"Thresholds: `{list(THRESHOLDS)}`; comparator: `gate_cosine >= tau`; ranking: stable `(-score, ASIN)` over the persisted eligibility mask.", "",
             "## Seven-slice M0/M3 baseline (tau 0.20)", ""]
    lines += metric_markdown(result["baseline_table"])
    lines += ["", "Help/harm/unchanged above use each probe's relevant-set reciprocal-rank delta. Hit rates and penalized-rank summaries are arm-specific.", "",
              "## Gate distributions", "",
              "Quantiles use linear interpolation; standard deviations are sample standard deviations (`ddof=1`).", "",
              "| Scenario class | n | min | p10 | p25 | median | p75 | p90 | max | mean | sample sd |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, row in result["gate_distributions"].items():
        lines.append(f"| {name} | {row['count']} | {row['min']:.6f} | {row['p10']:.6f} | {row['p25']:.6f} | {row['median']:.6f} | {row['p75']:.6f} | {row['p90']:.6f} | {row['max']:.6f} | {row['mean']:.6f} | {row['sample_stddev']:.6f} |")
    lines += ["", "### All probes by descending exact cosine", "",
              "| # | Session | Scenario | Relation | Mode | Cosine | tau-0.20 pass |",
              "|---:|---|---|---|---|---:|---|"]
    for index, row in enumerate(result["sorted_probes"], 1):
        lines.append(f"| {index} | {row['session_id']} | {row['scenario_class']} | {row['memory_relation']} | {row['buyer_mode']} | {row['gate_cosine']:.9f} | {row['gate_passed']} |")
    lines += ["", "## Classification diagnostics", "",
              f"RELEVANT-vs-IRRELEVANT ROC AUC: **{cls['roc_auc']:.6f}**. PR AUC (average precision): **{cls['pr_auc_average_precision']:.6f}**. CONFLICTING probes are excluded.", "",
              f"At tau 0.20: TP={cls['tau_0_20']['tp']}, FP={cls['tau_0_20']['fp']}, TN={cls['tau_0_20']['tn']}, FN={cls['tau_0_20']['fn']}, TPR={cls['tau_0_20']['tpr']:.3f}, FPR={cls['tau_0_20']['fpr']:.3f}.", "",
              f"Highest-threshold Youden-J tie winner: tau={cls['youden_j_cutoff']['threshold']:.9f}, J={cls['youden_j_cutoff']['youden_j']:.3f}, TPR={cls['youden_j_cutoff']['tpr']:.3f}, FPR={cls['youden_j_cutoff']['fpr']:.3f}.", "",
              "All exact-score diagnostic cutoffs are in `classification_cutoffs.csv` and `results.json`.", "",
              "## Threshold sweep", ""]
    lines += metric_markdown(result["threshold_metrics"])
    lines += ["", "### Activation and operating-region test", "",
              "| tau | Relevant activation (n=20) | Irrelevant activation (n=10) | Conflicting activation (n=10) | LP delta | BP delta | MI delta | CO delta | Overall delta | Qualifies |",
              "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for row in result["threshold_summary"]:
        lines.append(f"| {row['threshold']:.2f} | {row['relevant_activation_rate']:.3f} | {row['irrelevant_activation_rate']:.3f} | {row['conflicting_activation_rate']:.3f} | {row['longitudinal_positive_delta']:+.6f} | {row['browsing_personalization_delta']:+.6f} | {row['memory_irrelevant_delta']:+.6f} | {row['current_override_delta']:+.6f} | {row['overall_delta']:+.6f} | {row['operating_region']} |")
    qualifying = result["qualifying_thresholds"]
    lines += ["", f"Qualifying thresholds (reported without maximum-MRR selection): **{qualifying if qualifying else 'none'}**.", "",
              "## Tau-0.20 MEMORY_IRRELEVANT passers", ""]
    for item in result["forensics"]:
        lines += [f"### {item['session_id']} — class {item['classification']}", "",
                  f"- Query: `{item['query']}`", f"- Gate cosine: `{item['gate_cosine']:.9f}`",
                  f"- Relevant rank/RR: {item['m0_relevant_rank']} ({item['m0_rr']:.9f}) → {item['m3_relevant_rank']} ({item['m3_rr']:.9f}); delta {item['rr_delta']:+.9f}",
                  f"- Classification basis: {item['classification_reason']}",
                  f"- Target: `{item['target_asin']}` — {item['target_metadata'].get('title')}",
                  f"- Relevant ASINs: `{item['relevant_asins']}`", f"- Persisted overtakers: {item['persisted_overtaker_count']}", "",
                  "Textual/vector lineage, complete target/relevant catalogue metadata, parsed query state, and every hash-joined overtaker with s1/s2/s3 are in `forensics.json` and `forensics.csv`.", ""]
    lines += ["## Interpretation", "", result["conclusion_basis"], "",
              "The conclusion follows the required precedence rule; no threshold was selected by maximizing MRR.", ""]
    return "\n".join(lines)


def run(input_dir: Path, output_dir: Path, catalog_path: Path) -> dict[str, Any]:
    manifest = read_json(input_dir / "manifest.json")
    records = [json.loads(line) for line in (input_dir / "sessions.jsonl").read_text(encoding="utf-8").splitlines() if line]
    probes = [row for row in records if row["record_type"] == "probe"]
    setup = [row for row in records if row["record_type"] == "setup_update"]
    catalog_asins, catalog = load_catalog(catalog_path, manifest["catalog_sha256"])
    with np.load(input_dir / "vectors.npz", allow_pickle=False) as loaded:
        arrays = {key: loaded[key] for key in loaded.files}
    asins = arrays["catalog_asins"]
    assert asins.tolist() == catalog_asins
    frozen_hashes_before = {name: sha256_file(input_dir / name) for name in manifest["artifact_hashes"]}
    verification = verify_bundle(input_dir, manifest, records, arrays, asins)

    # Save byte-level sentinels and prove the arrays used as inputs are not mutated.
    immutable_keys = [ref[key]["npz_key"] for probe in probes for ref in [probe["array_references"]]
                      for key in ("s1", "s2", "eligibility")]
    input_array_hashes = {key: array_sha256(arrays[key]) for key in immutable_keys}

    sweep: dict[str, Any] = {}
    metric_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for tau in THRESHOLDS:
        reconstructed, activation = reconstruct(probes, arrays, asins, tau)
        metrics = {name: slice_metrics(rows) for name, rows in subsets(reconstructed).items()}
        sweep[f"{tau:.2f}"] = {"activation": activation, "slices": metrics}
        metric_rows.extend(flatten_metric(tau, name, metrics[name]) for name in SLICE_NAMES)
        deltas = {name: metrics[name]["mean_rr_delta"] for name in ("OVERALL",) + SCENARIOS}
        qualifies = (deltas["LONGITUDINAL_POSITIVE"] > 0 and deltas["BROWSING_PERSONALIZATION"] > 0
                     and deltas["MEMORY_IRRELEVANT"] >= 0 and deltas["CURRENT_OVERRIDE"] >= 0
                     and deltas["OVERALL"] >= 0)
        summary_rows.append({"threshold": tau, **activation,
                             "longitudinal_positive_delta": deltas["LONGITUDINAL_POSITIVE"],
                             "browsing_personalization_delta": deltas["BROWSING_PERSONALIZATION"],
                             "memory_irrelevant_delta": deltas["MEMORY_IRRELEVANT"],
                             "current_override_delta": deltas["CURRENT_OVERRIDE"],
                             "overall_delta": deltas["OVERALL"], "operating_region": qualifies})

    tau20_rows, _ = reconstruct(probes, arrays, asins, .20)
    for reconstructed, persisted in zip(tau20_rows, probes):
        assert reconstructed["gate_passed"] == persisted["gate_passed"]
        assert reconstructed["m3_rank"] == persisted["m3_relevant_rank"]
        assert reconstructed["m3_penalized_rank"] == persisted["m3_penalized_rank"]
    baseline = {name: slice_metrics(rows) for name, rows in subsets(tau20_rows).items()}
    published = read_json(input_dir / "metrics.json")
    assert_published_metric_parity(baseline["OVERALL"], published["overall"]["relevant_set"])
    for name in SCENARIOS:
        assert_published_metric_parity(baseline[name], published["by_scenario_class"][name]["relevant_set"])
    for name in ("Buying", "Browsing"):
        assert_published_metric_parity(baseline[name], published["by_buyer_mode"][name]["relevant_set"])
    tau80_rows, _ = reconstruct(probes, arrays, asins, .80)
    assert all(row["m3_rank"] == row["m0_rank"] and row["m3_penalized_rank"] == row["m0_penalized_rank"] for row in tau80_rows)
    assert all(array_sha256(arrays[key]) == digest for key, digest in input_array_hashes.items())

    cls = classifier_metrics(probes)
    forensics = forensic_rows(probes, setup, arrays, asins, catalog)
    harmful_passers = [row for row in forensics if row["rr_delta"] < 0]
    qualifying = [row["threshold"] for row in summary_rows if row["operating_region"]]
    tau20_summary = next(row for row in summary_rows if row["threshold"] == .20)
    safety_improved = any(row["irrelevant_activation_rate"] < tau20_summary["irrelevant_activation_rate"]
                          and row["memory_irrelevant_delta"] > tau20_summary["memory_irrelevant_delta"]
                          and row["current_override_delta"] >= tau20_summary["current_override_delta"]
                          for row in summary_rows)
    diagnostic_cutoff_results = []
    for cutoff in cls["recall_90_fpr_10_cutoffs"]:
        cutoff_rows, activation = reconstruct(probes, arrays, asins, cutoff["threshold"])
        metrics = {name: slice_metrics(rows) for name, rows in subsets(cutoff_rows).items()}
        diagnostic_cutoff_results.append({"threshold": cutoff["threshold"], "classification": cutoff,
                                          "activation": activation,
                                          "slice_rr_deltas": {name: metrics[name]["mean_rr_delta"]
                                                              for name in ("OVERALL",) + SCENARIOS}})
    diagnostic_feasible_but_harmful = any(
        row["slice_rr_deltas"]["OVERALL"] < 0
        or row["slice_rr_deltas"]["LONGITUDINAL_POSITIVE"] <= 0
        or row["slice_rr_deltas"]["BROWSING_PERSONALIZATION"] <= 0
        for row in diagnostic_cutoff_results
    )
    if qualifying:
        conclusion = "THRESHOLD MISCALIBRATION IS SUFFICIENT TO EXPLAIN HARM"
        follow_up = "Independent holdout confirmation."
        basis = "The strict operating region is non-empty."
    elif safety_improved and any(row["classification"] == "B" for row in harmful_passers):
        conclusion = "THRESHOLD HELPS BUT MEMORY REPRESENTATION STILL CAUSES MATERIAL HARM"
        follow_up = "Persistent-scope-only reconstruction."
        basis = "Thresholding improves tau-0.20 safety, but at least one harmful passer has representation contamination."
    elif diagnostic_feasible_but_harmful:
        conclusion = "GATE COSINE IS DISCRIMINATIVE BUT FIXED BLENDING REMAINS HARMFUL"
        follow_up = "Frozen-vector blend-weight sweep."
        basis = ("An exact diagnostic cutoff reaches at least 90% relevant recall and at most 10% irrelevant FPR, "
                 "but the strict operating region is empty because activated memory still harms benefit/overall slices.")
    else:
        conclusion = "CURRENT ARCHITECTURE CANNOT BE MADE SAFE BY THRESHOLD CALIBRATION ALONE"
        follow_up = "Offline continuous-attenuation comparison."
        basis = "No earlier conclusion condition is satisfied."

    sorted_probes = [{key: row[key] for key in ("session_id", "scenario_class", "memory_relation", "buyer_mode", "gate_cosine", "gate_passed")}
                     for row in sorted(probes, key=lambda row: (-float(row["gate_cosine"]), str(row["session_id"])))]
    result = {"schema_version": "v2-threshold-calibration-1", "input_directory": str(input_dir),
              "catalog_path": str(catalog_path), "frozen_manifest": manifest,
              "verification": {**verification, "catalog_sha256_verified": manifest["catalog_sha256"],
                               "tau_0_20_reproduces_persisted_gate_and_ranks": True,
                               "tau_0_20_reproduces_published_m3_metrics": True,
                               "tau_0_80_reproduces_m0": True, "input_arrays_unmutated": True,
                               "rate_triplets_sum_to_one": True, "relation_denominators_verified": True},
              "thresholds": list(THRESHOLDS), "baseline": baseline,
              "baseline_table": [flatten_metric(.20, name, baseline[name]) for name in SLICE_NAMES],
              "gate_distributions": {name: distribution([row["gate_cosine"] for row in probes if row["scenario_class"] == name]) for name in SCENARIOS},
              "sorted_probes": sorted_probes, "classification": cls, "sweep": sweep,
              "diagnostic_cutoff_outcomes": diagnostic_cutoff_results,
              "threshold_metrics": metric_rows, "threshold_summary": summary_rows,
              "qualifying_thresholds": qualifying, "forensics": forensics,
              "conclusion": conclusion, "conclusion_basis": basis, "follow_up_experiment": follow_up}

    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "results.json", result)
    write_csv(output_dir / "baseline_metrics.csv", result["baseline_table"])
    write_csv(output_dir / "threshold_metrics.csv", metric_rows)
    write_csv(output_dir / "threshold_summary.csv", summary_rows)
    write_csv(output_dir / "gate_distributions.csv", [{"scenario_class": name, **row} for name, row in result["gate_distributions"].items()])
    write_csv(output_dir / "sorted_probes.csv", sorted_probes)
    write_csv(output_dir / "classification_cutoffs.csv", cls["exact_score_cutoffs"])
    write_json(output_dir / "forensics.json", forensics)
    write_csv(output_dir / "forensics.csv", forensics)
    (output_dir / "report.md").write_text(render_report(result), encoding="utf-8")

    frozen_hashes_after = {name: sha256_file(input_dir / name) for name in manifest["artifact_hashes"]}
    assert frozen_hashes_after == frozen_hashes_before == manifest["artifact_hashes"]
    output_hashes = {path.name: sha256_file(path) for path in sorted(output_dir.iterdir())
                     if path.is_file() and path.name != "manifest.json"}
    output_manifest = {"schema_version": result["schema_version"], "frozen_artifact_hashes": frozen_hashes_after,
                       "catalog_sha256": manifest["catalog_sha256"], "output_hashes": output_hashes,
                       "production_calls": {"embedding": 0, "llm": 0}, "production_configuration_changed": False}
    write_json(output_dir / "manifest.json", output_manifest)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    args = parser.parse_args()
    result = run(args.input.resolve(), args.output.resolve(), args.catalog.resolve())
    print(result["conclusion"])
    print(f"Qualifying thresholds: {result['qualifying_thresholds']}")
    print(f"Wrote {args.output.resolve()}")


if __name__ == "__main__":
    main()
