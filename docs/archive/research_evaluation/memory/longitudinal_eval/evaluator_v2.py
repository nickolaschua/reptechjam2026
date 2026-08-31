"""Deterministic M0-versus-M3 longitudinal evaluator and forensic bundle.

The module deliberately owns evaluation annotations and update lineage.  The
production memory remains one aggregate vector per user and is never made to
pretend that it contains item records.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import tempfile
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from ..agent import Agent, ForensicRankingSnapshot
from ..memory_store import InMemoryVectorMemoryStore
from ..vector_memory import DEFAULT_VECTOR_MEMORY_CONFIG


SCHEMA_VERSION = "nickolas-longitudinal-eval-v2"
FIXTURE_SCHEMA_VERSION = "nickolas-longitudinal-fixture-v2"
SCENARIO_CLASSES = (
    "LONGITUDINAL_POSITIVE", "MEMORY_IRRELEVANT", "CURRENT_OVERRIDE",
    "BROWSING_PERSONALIZATION",
)
EXPECTED_BEHAVIORS = {
    "LONGITUDINAL_POSITIVE": "HELP",
    "MEMORY_IRRELEVANT": "IGNORE",
    "CURRENT_OVERRIDE": "DO_NOT_OVERRIDE",
    "BROWSING_PERSONALIZATION": "PERSONALIZE",
}
RELATIONS = {
    "LONGITUDINAL_POSITIVE": "RELEVANT",
    "MEMORY_IRRELEVANT": "IRRELEVANT",
    "CURRENT_OVERRIDE": "CONFLICTING",
    "BROWSING_PERSONALIZATION": "RELEVANT",
}
FAST_PREFIXES = (
    "i'm looking for ", "for that, what matters is:",
    "actually, ignore my earlier preference. what i need is:",
    "i don't have a preference for", "i don't have an additional preference for",
)


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _messages(session: Mapping[str, Any]) -> list[str]:
    value = session.get("scripted_turns")
    if not isinstance(value, list) or not value or any(not isinstance(v, str) or not v.strip() for v in value):
        raise ValueError("every session requires non-empty scripted_turns")
    return value


def validate_fixture_v2(
    fixture: Mapping[str, Any], catalog: Mapping[str, Mapping[str, Any]] | None = None
) -> dict[str, Any]:
    errors: list[str] = []
    users = fixture.get("users")
    if fixture.get("schema_version") != FIXTURE_SCHEMA_VERSION:
        errors.append("fixture schema_version is invalid")
    if not isinstance(users, list) or len(users) != 40:
        errors.append("fixture must contain exactly 40 isolated timelines")
        users = [] if not isinstance(users, list) else users
    counts: Counter[str] = Counter()
    modes: Counter[str] = Counter()
    seen_users: set[str] = set()
    all_relevant: list[str] = []
    audited: set[str] = set()
    setup_count = probe_count = 0
    for user in users:
        uid = str(user.get("user_id", ""))
        scenario = str(user.get("scenario_class", ""))
        mode = str(user.get("buyer_mode", ""))
        if not uid or uid in seen_users:
            errors.append(f"duplicate/empty user_id {uid!r}")
        seen_users.add(uid)
        counts[scenario] += 1
        modes[mode] += 1
        if scenario not in SCENARIO_CLASSES:
            errors.append(f"{uid}: unknown scenario class")
        if user.get("expected_behavior") != EXPECTED_BEHAVIORS.get(scenario):
            errors.append(f"{uid}: expected behavior does not match class")
        expected_mode = "Browsing" if scenario == "BROWSING_PERSONALIZATION" else "Buying"
        if mode != expected_mode:
            errors.append(f"{uid}: buyer mode must be {expected_mode}")
        if user.get("memory_relation") != RELATIONS.get(scenario):
            errors.append(f"{uid}: memory relation does not match class")
        sessions = user.get("sessions")
        if not isinstance(sessions, list) or len(sessions) != 3:
            errors.append(f"{uid}: timeline must have two setup sessions and one probe")
            continue
        indices = [s.get("sequence_index") for s in sessions if isinstance(s, Mapping)]
        if indices != [0, 1, 2]:
            errors.append(f"{uid}: sequence indices must be exactly [0, 1, 2]")
        roles = [s.get("session_role") for s in sessions]
        if roles != ["setup", "setup", "probe"]:
            errors.append(f"{uid}: session roles must be setup/setup/probe")
        setup_count += sum(role == "setup" for role in roles)
        probe_count += sum(role == "probe" for role in roles)
        for session in sessions:
            try:
                messages = _messages(session)
            except ValueError as exc:
                errors.append(f"{uid}: {exc}")
                continue
            if any(not message.casefold().startswith(FAST_PREFIXES) and
                   "those options are not quite right yet" not in message.casefold()
                   for message in messages):
                errors.append(f"{uid}: scripted message is not deterministic fast-path grammar")
            scopes = session.get("fact_scope_annotations")
            if not isinstance(scopes, Mapping) or set(scopes) != {"persistent", "session_specific", "unknown"}:
                errors.append(f"{uid}: missing persistent/session_specific/unknown scopes")
            for field in ("setup_sequence_references",):
                refs = session.get(field, [])
                if any(not isinstance(v, int) or v < 0 or v >= session["sequence_index"] for v in refs):
                    errors.append(f"{uid}: invalid or future setup reference")
        probe = sessions[2]
        relevant = probe.get("relevant_asins")
        target = str(probe.get("target_asin", ""))
        if not isinstance(relevant, list) or not 3 <= len(relevant) <= 8:
            errors.append(f"{uid}: relevant_asins must contain 3-8 alternatives")
            relevant = []
        if len(relevant) != len(set(relevant)):
            errors.append(f"{uid}: relevant_asins contains duplicates")
        if target not in relevant:
            errors.append(f"{uid}: target_asin must belong to relevant_asins")
        all_relevant.extend(str(v) for v in relevant)
        evidence = probe.get("relevant_asin_audit", {})
        if not isinstance(evidence, Mapping) or set(relevant) - set(evidence):
            errors.append(f"{uid}: every relevant ASIN requires audit evidence")
        else:
            audited.update(evidence)
        if catalog is not None:
            unresolved = [asin for asin in relevant if asin not in catalog]
            if unresolved:
                errors.append(f"{uid}: unresolved relevant ASINs {unresolved}")
        sufficient = bool(probe.get("current_query_alone_sufficient"))
        if sufficient != (scenario in {"MEMORY_IRRELEVANT", "CURRENT_OVERRIDE"}):
            errors.append(f"{uid}: current_query_alone_sufficient is incorrect")
        probe_text = " ".join(_messages(probe)).casefold()
        history_facts = [str(v).casefold() for v in probe.get("intended_historical_facts", [])]
        leakage_terms = [str(v).casefold() for v in probe.get("historical_fact_leakage_terms", history_facts)]
        if scenario == "LONGITUDINAL_POSITIVE" and any(term and term in probe_text for term in leakage_terms):
            errors.append(f"{uid}: longitudinal-positive query leaks historical facts")
        if scenario == "CURRENT_OVERRIDE":
            if len(_messages(probe)) < 2 or "what i need is:" not in _messages(probe)[-1].casefold():
                errors.append(f"{uid}: current override lacks an explicit override turn")
            if not probe.get("history_matching_current_conflicting_asins"):
                errors.append(f"{uid}: current override lacks conflicting-history safety labels")
        if scenario == "MEMORY_IRRELEVANT" and not probe.get("semantic_separation_audit"):
            errors.append(f"{uid}: memory-irrelevant row lacks semantic separation audit")
    if counts != Counter({name: 10 for name in SCENARIO_CLASSES}):
        errors.append(f"scenario balance must be 10/10/10/10; got {dict(counts)}")
    if modes != Counter({"Buying": 30, "Browsing": 10}):
        errors.append(f"mode balance must be Buying=30/Browsing=10; got {dict(modes)}")
    if setup_count != 80 or probe_count != 40:
        errors.append("fixture must contain 80 setup interactions and 40 probes")
    return {"valid": not errors, "errors": errors, "scenario_counts": dict(counts),
            "mode_counts": dict(modes), "setup_count": setup_count, "probe_count": probe_count,
            "relevant_asin_reference_count": len(all_relevant), "audited_asin_count": len(audited)}


def rank_rows(scores: np.ndarray, eligible_mask: np.ndarray, asins: Sequence[str]) -> np.ndarray:
    values = np.asarray(scores)
    mask = np.asarray(eligible_mask, dtype=bool)
    if values.ndim != 1 or mask.shape != values.shape or len(asins) != len(values):
        raise ValueError("scores, mask, and ASINs must align")
    eligible = np.flatnonzero(mask)
    return np.asarray(sorted(eligible.tolist(), key=lambda row: (-float(values[row]), str(asins[row]))), dtype=np.int64)


def assert_paired_ranking_invariants(
    snapshot: ForensicRankingSnapshot, catalog_asins: Sequence[str],
    product_embeddings: np.ndarray, *, required_rows: int = 50_000,
) -> dict[str, bool]:
    """Fail on configuration/catalogue/mask/embedding drift, not rank changes."""
    matrix = np.asarray(product_embeddings)
    checks = {
        "row_count": len(catalog_asins) == required_rows == matrix.shape[0] == len(snapshot.s1),
        "embedding_scores": np.allclose(matrix @ snapshot.v1, snapshot.s1, rtol=1e-6, atol=1e-6),
        "shared_hard_masks": np.array_equal(snapshot.eligibility_mask,
                                             snapshot.price_mask & snapshot.negative_mask),
        "m0_stable_order": np.array_equal(snapshot.m0_ranked_rows,
                                           rank_rows(snapshot.s1, snapshot.eligibility_mask, catalog_asins)),
        "m3_stable_order": np.array_equal(snapshot.m3_ranked_rows,
                                           rank_rows(snapshot.s3, snapshot.eligibility_mask, catalog_asins)),
        "frozen_threshold": DEFAULT_VECTOR_MEMORY_CONFIG.relevance_threshold == 0.20,
        "frozen_weights": (DEFAULT_VECTOR_MEMORY_CONFIG.buying_current_weight,
                           DEFAULT_VECTOR_MEMORY_CONFIG.buying_memory_weight,
                           DEFAULT_VECTOR_MEMORY_CONFIG.browsing_current_weight,
                           DEFAULT_VECTOR_MEMORY_CONFIG.browsing_memory_weight) == (.8, .2, .2, .8),
    }
    if snapshot.s2 is not None and snapshot.v2 is not None:
        checks["memory_embedding_scores"] = np.allclose(matrix @ snapshot.v2, snapshot.s2,
                                                        rtol=1e-6, atol=1e-6)
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError("paired ranking invariant drift: " + ", ".join(failed))
    return checks


def raw_and_penalized_rank(ranked_asins: Sequence[str], relevant_asins: Iterable[str], eligible_count: int) -> tuple[int | None, int]:
    relevant = frozenset(str(v) for v in relevant_asins)
    raw = next((index for index, asin in enumerate(ranked_asins, start=1) if asin in relevant), None)
    return raw, (eligible_count + 1 if raw is None else raw)


def _quantile(values: Sequence[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * probability
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def distribution(values: Sequence[float]) -> dict[str, float | None]:
    clean = [float(v) for v in values]
    return {"min": min(clean) if clean else None, "p10": _quantile(clean, .10),
            "p25": _quantile(clean, .25), "median": _quantile(clean, .50),
            "p75": _quantile(clean, .75), "p90": _quantile(clean, .90),
            "max": max(clean) if clean else None,
            "mean": statistics.fmean(clean) if clean else None,
            "sample_stddev": statistics.stdev(clean) if len(clean) > 1 else None,
            "count": len(clean)}


def tied_auc(positive_scores: Sequence[float], negative_scores: Sequence[float]) -> float | None:
    """Mann-Whitney AUC: ties contribute one half."""
    positives, negatives = list(map(float, positive_scores)), list(map(float, negative_scores))
    if not positives or not negatives:
        return None
    wins = sum(1.0 if p > n else 0.5 if p == n else 0.0 for p in positives for n in negatives)
    return wins / (len(positives) * len(negatives))


def metric_summary(rows: Sequence[Mapping[str, Any]], prefix: str = "relevant_set") -> dict[str, Any]:
    if not rows:
        return {"count": 0}
    m0 = [row["m0_relevant_rank"] for row in rows]
    m3 = [row["m3_relevant_rank"] for row in rows]
    p0 = [int(row["m0_penalized_rank"]) for row in rows]
    p3 = [int(row["m3_penalized_rank"]) for row in rows]
    rr0 = [0.0 if rank is None else 1.0 / rank for rank in m0]
    rr3 = [0.0 if rank is None else 1.0 / rank for rank in m3]
    deltas = [right - left for left, right in zip(rr0, rr3)]
    rank_deltas = [left - right for left, right in zip(p0, p3)]
    out: dict[str, Any] = {"count": len(rows)}
    for label, ranks, penalized, rr in (("m0", m0, p0, rr0), ("m3", m3, p3, rr3)):
        out[label] = {
            f"{prefix}_hit_at_1": sum(r is not None and r <= 1 for r in ranks) / len(rows),
            f"{prefix}_hit_at_5": sum(r is not None and r <= 5 for r in ranks) / len(rows),
            f"{prefix}_hit_at_10": sum(r is not None and r <= 10 for r in ranks) / len(rows),
            f"{prefix}_mrr": statistics.fmean(rr),
            f"{prefix}_mean_penalized_rank": statistics.fmean(penalized),
            f"{prefix}_median_penalized_rank": statistics.median(penalized),
        }
    out.update({
        f"{prefix}_mean_reciprocal_rank_delta": statistics.fmean(deltas),
        f"{prefix}_mean_rank_delta": statistics.fmean(rank_deltas),
        f"{prefix}_help_rate": sum(v > 0 for v in deltas) / len(deltas),
        f"{prefix}_harm_rate": sum(v < 0 for v in deltas) / len(deltas),
        f"{prefix}_unchanged_rate": sum(v == 0 for v in deltas) / len(deltas),
        "gate_activation_rate": sum(bool(row.get("gate_passed")) for row in rows) / len(rows),
    })
    return out


def classify_mechanism(rows: Sequence[Mapping[str, Any]]) -> str:
    grouped = {name: [r for r in rows if r["scenario_class"] == name] for name in SCENARIO_CLASSES}
    summaries = {name: metric_summary(group) for name, group in grouped.items()}
    overall = metric_summary(rows)
    def harmful(summary: Mapping[str, Any]) -> bool:
        return (summary.get("relevant_set_mean_reciprocal_rank_delta", 0) < 0 and
                summary.get("relevant_set_harm_rate", 0) > summary.get("relevant_set_help_rate", 0))
    if harmful(overall) or any(harmful(summaries[name]) for name in ("MEMORY_IRRELEVANT", "CURRENT_OVERRIDE")):
        return "HARMFUL"
    positive = all(
        summaries[name].get("relevant_set_mean_reciprocal_rank_delta", 0) > 0 and
        summaries[name].get("relevant_set_help_rate", 0) > summaries[name].get("relevant_set_harm_rate", 0)
        for name in ("LONGITUDINAL_POSITIVE", "BROWSING_PERSONALIZATION")
    )
    if positive:
        return "SHOWS POSITIVE LONGITUDINAL SIGNAL"
    if all(summaries[name].get("relevant_set_help_rate", 0) == 0 for name in ("LONGITUDINAL_POSITIVE", "BROWSING_PERSONALIZATION")):
        return "NO POSITIVE SIGNAL"
    return "INCONCLUSIVE"


def vector_reference(key: str, value: np.ndarray) -> dict[str, Any]:
    array = np.ascontiguousarray(value)
    return {"npz_key": key, "dtype": str(array.dtype), "shape": list(array.shape),
            "l2_norm": float(np.linalg.norm(array)), "sha256": sha256_bytes(array.tobytes(order="C"))}


def verify_vector_reference(reference: Mapping[str, Any], arrays: Mapping[str, np.ndarray]) -> bool:
    key = str(reference["npz_key"])
    if key not in arrays:
        return False
    return vector_reference(key, arrays[key]) == dict(reference)


def _product_record(agent: Agent, row: int, snapshot: ForensicRankingSnapshot, rank: int) -> dict[str, Any]:
    asin = agent.catalog_ids[row]
    meta = agent.catalog_metadata[asin]
    return {"asin": asin, "title": meta.get("title"), "categories": sorted(agent.catalog_categories_set[row]),
            "s1": float(snapshot.s1[row]), "s2": None if snapshot.s2 is None else float(snapshot.s2[row]),
            "s3": float(snapshot.s3[row]), "rank": rank}


def _lineage(user: Mapping[str, Any], embedded_updates: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [{"update_id": row["update_id"], "text": row["text"], "scopes": deepcopy(row["scopes"]),
             "sequence_index": row["sequence_index"],
             "contributing_sequences": list(range(row["sequence_index"] + 1))}
            for row in embedded_updates]


def run_v2(
    agent: Agent, fixture: Mapping[str, Any], *, top_k: int = 10,
    response_stub: str = '{"message":"Forensic replay.","ask_attribute":"other"}',
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    """Run 80 setup sessions and score the final turn of 40 probes."""
    validation = validate_fixture_v2(fixture, agent.catalog_metadata)
    if not validation["valid"]:
        raise ValueError("invalid v2 fixture: " + "; ".join(validation["errors"]))
    agent._call_llm = lambda *args, **kwargs: response_stub
    arrays: dict[str, np.ndarray] = {"catalog_asins": np.asarray(agent.catalog_ids, dtype="U")}
    rows: list[dict[str, Any]] = []
    for user in fixture["users"]:
        embedded_updates: list[dict[str, Any]] = []
        for session in user["sessions"]:
            sid = f"{user['user_id']}_s{session['sequence_index']}"
            agent.reset(sid, user.get("constant_profile", {}), user_id=user["user_id"],
                        sequence_index=session["sequence_index"])
            agent.enable_forensic_memory_update(sid)
            is_probe = session["session_role"] == "probe"
            if is_probe:
                agent.enable_forensic_ranking(sid)
            canonical_states: list[dict[str, Any]] = []
            for turn, message in enumerate(_messages(session), 1):
                response = agent.respond(sid, message, turn, top_k,
                                         buyer_mode=user["buyer_mode"].casefold())
                canonical_states.append(deepcopy(agent._sessions[sid]))
            before = agent.memory_store.get_state(user["user_id"])
            if is_probe:
                snap = agent.get_forensic_ranking_snapshots(sid)[-1]
                assert_paired_ranking_invariants(snap, agent.catalog_ids, agent.catalog_embeddings)
                expected_m0 = rank_rows(snap.s1, snap.eligibility_mask, agent.catalog_ids)
                expected_m3 = rank_rows(snap.s3, snap.eligibility_mask, agent.catalog_ids)
                if not np.array_equal(expected_m0, snap.m0_ranked_rows) or not np.array_equal(expected_m3, snap.m3_ranked_rows):
                    raise RuntimeError("rank order violates (-score, ASIN) contract")
                m0_asins = [agent.catalog_ids[i] for i in snap.m0_ranked_rows]
                m3_asins = [agent.catalog_ids[i] for i in snap.m3_ranked_rows]
                relevant = session["relevant_asins"]
                m0_raw, m0_penalty = raw_and_penalized_rank(m0_asins, relevant, len(m0_asins))
                m3_raw, m3_penalty = raw_and_penalized_rank(m3_asins, relevant, len(m3_asins))
                target = session["target_asin"]
                target_m0 = m0_asins.index(target) + 1 if target in m0_asins else None
                target_m3 = m3_asins.index(target) + 1 if target in m3_asins else None
                relevant_m0_rank = {asin: m0_asins.index(asin) + 1 for asin in relevant if asin in m0_asins}
                relevant_m3_rank = {asin: m3_asins.index(asin) + 1 for asin in relevant if asin in m3_asins}
                threshold = min(relevant_m0_rank.values()) if relevant_m0_rank else len(m0_asins) + 1
                overtakers = []
                for rank, asin in enumerate(m3_asins, 1):
                    old_rank = m0_asins.index(asin) + 1
                    if old_rank > threshold and rank < (m3_raw or len(m3_asins) + 1):
                        overtakers.append({"asin": asin, "m0_rank": old_rank, "m3_rank": rank})
                prefix = f"{user['user_id']}_probe"
                refs = {}
                for suffix, value in (("v1", snap.v1), ("v2", snap.v2), ("s1", snap.s1),
                                      ("s2", snap.s2), ("s3", snap.s3),
                                      ("price_mask", snap.price_mask),
                                      ("negative_mask", snap.negative_mask),
                                      ("eligibility", snap.eligibility_mask)):
                    if value is not None:
                        key = f"{prefix}_{suffix}"
                        arrays[key] = np.array(value, copy=True)
                        refs[suffix] = vector_reference(key, arrays[key])
                row = {
                    "record_type": "probe",
                    "session_id": sid, "user_id": user["user_id"],
                    "scenario_class": user["scenario_class"], "expected_behavior": user["expected_behavior"],
                    "buyer_mode": user["buyer_mode"], "memory_relation": user["memory_relation"],
                    "scripted_turns": list(session["scripted_turns"]),
                    "canonical_parsed_state": snap.canonical_state,
                    "disclosed_slots": snap.canonical_state.get("disclosed_slots", {}),
                    "update_scope_annotations": deepcopy(session["fact_scope_annotations"]),
                    "contributing_memory_lineage": _lineage(user, embedded_updates),
                    "relevant_asins": list(relevant), "target_asin": target,
                    "eligible_count": len(m0_asins), "catalog_rows_scored": len(snap.s1),
                    "m0_relevant_rank": m0_raw, "m3_relevant_rank": m3_raw,
                    "m0_penalized_rank": m0_penalty, "m3_penalized_rank": m3_penalty,
                    "m0_target_rank": target_m0, "m3_target_rank": target_m3,
                    "relevant_rr_delta": (0 if m3_raw is None else 1/m3_raw) - (0 if m0_raw is None else 1/m0_raw),
                    "gate_cosine": snap.gate_cosine, "gate_passed": snap.gate_passed,
                    "gate_threshold": DEFAULT_VECTOR_MEMORY_CONFIG.relevance_threshold,
                    "weights": {"current": snap.current_weight, "memory": snap.memory_weight},
                    "array_references": refs,
                    "m0_top_results": [_product_record(agent, int(r), snap, i) for i, r in enumerate(snap.m0_ranked_rows[:top_k], 1)],
                    "m3_top_results": [_product_record(agent, int(r), snap, i) for i, r in enumerate(snap.m3_ranked_rows[:top_k], 1)],
                    "overtakers": overtakers,
                    "hard_masks": {"price_filtered": int(np.count_nonzero(~snap.price_mask)),
                                   "negative_filtered": int(np.count_nonzero(snap.price_mask & ~snap.negative_mask))},
                }
            else:
                row = None
            preference_text = __import__("nickolas.shopping_agent.vector_memory", fromlist=["positive_slot_text"]).positive_slot_text(
                agent._sessions[sid].get("disclosed_slots", {}))
            commit = agent.end_session(sid)
            update_vector = agent.get_forensic_memory_update(sid)
            update_reference = None
            if update_vector is not None:
                update_key = f"{user['user_id']}_embedded_update_{session['sequence_index']}"
                arrays[update_key] = np.array(update_vector, copy=True)
                update_reference = vector_reference(update_key, arrays[update_key])
            if session["session_role"] == "setup" and preference_text:
                update_id = f"{user['user_id']}:u{session['sequence_index']}"
                embedded_updates.append({"update_id": update_id, "text": preference_text,
                                         "scopes": deepcopy(session["fact_scope_annotations"]),
                                         "sequence_index": session["sequence_index"]})
                state = commit.state
                key = f"{user['user_id']}_update_{session['sequence_index']}"
                arrays[key] = np.array(state.vector, copy=True)
                state_reference = vector_reference(key, arrays[key])
                rows.append({"record_type": "setup_update", "session_id": sid,
                             "user_id": user["user_id"], "sequence_index": session["sequence_index"],
                             "scenario_class": user["scenario_class"], "buyer_mode": user["buyer_mode"],
                             "update_id": update_id, "update_text": preference_text,
                             "fact_scope_annotations": deepcopy(session["fact_scope_annotations"]),
                             "contributing_sequences": list(range(session["sequence_index"] + 1)),
                             "array_references": {"embedded_update": update_reference,
                                                  "post_update_memory": state_reference}})
            if is_probe and before is not None:
                key = f"{user['user_id']}_post_update"
                final = commit.state if commit and commit.state else before
                arrays[key] = np.array(final.vector, copy=True)
                row["array_references"]["post_update_memory"] = vector_reference(key, arrays[key])
                if update_reference is not None:
                    row["array_references"]["embedded_update"] = update_reference
            if is_probe:
                rows.append(row)
    return rows, arrays


def build_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    probes = [r for r in rows if r.get("record_type", "probe") == "probe"]
    def summaries(group: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        relevant = metric_summary(group)
        exact_rows = []
        for row in group:
            copied = dict(row)
            copied["m0_relevant_rank"] = row.get("m0_target_rank")
            copied["m3_relevant_rank"] = row.get("m3_target_rank")
            copied["m0_penalized_rank"] = row.get("m0_target_rank") or row["eligible_count"] + 1
            copied["m3_penalized_rank"] = row.get("m3_target_rank") or row["eligible_count"] + 1
            exact_rows.append(copied)
        exact = metric_summary(exact_rows, prefix="exact_target")
        return {"exact_target": exact, "relevant_set": relevant, **relevant}
    by_class = {name: summaries([r for r in probes if r["scenario_class"] == name]) for name in SCENARIO_CLASSES}
    by_mode = {mode: summaries([r for r in probes if r["buyer_mode"] == mode]) for mode in ("Buying", "Browsing")}
    gate_by_relation = {relation: distribution([r["gate_cosine"] for r in probes if r["memory_relation"] == relation and r["gate_cosine"] is not None])
                        for relation in ("RELEVANT", "IRRELEVANT", "CONFLICTING")}
    relevant = [r["gate_cosine"] for r in probes if r["memory_relation"] == "RELEVANT"]
    irrelevant = [r["gate_cosine"] for r in probes if r["memory_relation"] == "IRRELEVANT"]
    return {"overall": summaries(probes), "by_scenario_class": by_class, "by_buyer_mode": by_mode,
            "gate": {"by_relation": gate_by_relation, "relevant_vs_irrelevant_auc": tied_auc(relevant, irrelevant),
                     "conflicting_reported_separately": True},
            "mechanism_classification": classify_mechanism(probes)}


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)


def _render_report(metrics: Mapping[str, Any], rows: Sequence[Mapping[str, Any]],
                   trustworthy: bool, checks: Mapping[str, bool]) -> str:
    probes = [r for r in rows if r.get("record_type", "probe") == "probe"]
    def pct(value: float) -> str: return f"{100 * value:.1f}%"
    def metric_line(label: str, summary: Mapping[str, Any]) -> str:
        rel = summary["relevant_set"]
        return (f"| {label} | {rel['count']} | {rel['m0']['relevant_set_mrr']:.6f} | "
                f"{rel['m3']['relevant_set_mrr']:.6f} | "
                f"{rel['relevant_set_mean_reciprocal_rank_delta']:+.6f} | "
                f"{pct(rel['relevant_set_help_rate'])} | {pct(rel['relevant_set_harm_rate'])} | "
                f"{pct(rel['gate_activation_rate'])} |")
    table = ["| Slice | n | M0 MRR | M3 MRR | RR delta | Help | Harm | Gate |",
             "|---|---:|---:|---:|---:|---:|---:|---:|",
             metric_line("Overall", metrics["overall"])]
    table += [metric_line(name, metrics["by_scenario_class"][name]) for name in SCENARIO_CLASSES]
    table += [metric_line(mode, metrics["by_buyer_mode"][mode]) for mode in ("Buying", "Browsing")]
    gates = ["| Relation | min | p10 | p25 | median | p75 | p90 | max | mean | sample SD |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for relation in ("RELEVANT", "IRRELEVANT", "CONFLICTING"):
        d = metrics["gate"]["by_relation"][relation]
        gates.append("| " + relation + " | " + " | ".join(
            f"{d[key]:.6f}" if d[key] is not None else "n/a"
            for key in ("min", "p10", "p25", "median", "p75", "p90", "max", "mean", "sample_stddev")) + " |")
    def rr_delta(row: Mapping[str, Any]) -> float:
        if "relevant_rr_delta" in row:
            return float(row["relevant_rr_delta"])
        left, right = row.get("m0_relevant_rank"), row.get("m3_relevant_rank")
        return (0.0 if right is None else 1.0 / right) - (0.0 if left is None else 1.0 / left)
    ordered = sorted(probes, key=rr_delta)
    harm = ordered[0]; help_row = ordered[-1]
    unchanged = min(probes, key=lambda r: abs(rr_delta(r)))
    override = min((r for r in probes if r["scenario_class"] == "CURRENT_OVERRIDE"),
                   key=rr_delta)
    def example(label: str, row: Mapping[str, Any]) -> str:
        top0 = (row.get("m0_top_results") or [{"asin": "n/a", "title": "not recorded"}])[0]
        top3 = (row.get("m3_top_results") or [{"asin": "n/a", "title": "not recorded"}])[0]
        return (f"- **{label} — {row['session_id']} ({row['scenario_class']}):** relevant rank "
                f"{row['m0_relevant_rank']} → {row['m3_relevant_rank']}; RR delta "
                f"{rr_delta(row):+.6f}; gate={float(row.get('gate_cosine') or 0):.6f}. "
                f"M0 top: `{top0['asin']}` {top0['title']}; M3 top: `{top3['asin']}` {top3['title']}.")
    verification_lines = "\n".join(f"- {name}: {'PASS' if passed else 'FAIL'}" for name, passed in checks.items())
    return "\n".join([
        "# Trustworthy Longitudinal Evaluation v2", "",
        "## Outcome", "",
        f"- Mechanism: **{metrics['mechanism_classification']}**",
        f"- Evaluator: **{'EVALUATOR NOW TRUSTWORTHY' if trustworthy else 'NOT TRUSTWORTHY'}**",
        "- Frozen algorithm: alpha=0.30; gate=0.20; Buying=0.8/0.2; Browsing=0.2/0.8.",
        "- No tuning, threshold recommendation, memory redesign, or retrieval change was performed.", "",
        "## Fixture and determinism", "",
        "The frozen fixture contains 40 isolated timelines: 10 LONGITUDINAL_POSITIVE, 10 MEMORY_IRRELEVANT, "
        "10 CURRENT_OVERRIDE, and 10 BROWSING_PERSONALIZATION. Each has two setup sessions and one scored "
        "probe (80 setup records, 40 probes). Buying has 30 probes and Browsing exactly 10. All messages use "
        "the local parser grammar; response prose is stubbed identically. Both arms share canonical state, v1, "
        "catalogue order, product embeddings, price/negative masks, and eligibility. They differ only in v2, "
        "gate consequence, s3, and permissible rank outcomes.", "",
        "## Relevant-set results", "", *table, "",
        "Exact-target metrics are retained separately under `exact_target` in `metrics.json`; relevant-set metrics "
        "use the best-ranked member of each frozen 3–8 ASIN set. Nullable misses use `eligible_count + 1` only "
        "for penalized rank summaries.", "",
        "## Gate diagnostics", "", *gates, "",
        f"RELEVANT-vs-IRRELEVANT rank AUC (ties count 0.5): **{metrics['gate']['relevant_vs_irrelevant_auc']:.6f}**. "
        "CONFLICTING is reported separately and is not included in the ROC/AUC diagnostic.", "",
        "## Forensic examples", "", example("Helping", help_row), example("Ignored/least changed", unchanged),
        example("Harming", harm), example("Override safety", override), "",
        "Every probe record contains full update lineage; canonical parsed state and disclosed slots; hard-mask "
        "counts; gate and weights; M0/M3 top results with ASIN/title/categories/s1/s2/s3/rank; raw and penalized "
        "target/relevant ranks; and every below-relevant M0 product that overtook the relevant set in M3.", "",
        "## Verification", "", verification_lines, "",
        "The bundle is self-contained: `vectors.npz` includes catalogue ASIN order, v1, pre-query v2, update "
        "vectors, post-update vectors, s1/s2/s3, and all masks. Array references record key, dtype, shape, L2 "
        "norm, and SHA-256. `manifest.json` fingerprints fixture, catalogue, embedding cache, sources, and artifacts.", "",
        "## Bugs fixed and files", "",
        "- Replaced obsolete item-like `visible_matches` serialization with truthful aggregate-vector descriptions "
        "and evaluator-owned update lineage.",
        "- Restricted paired LLM tapes to parser calls and made response prose identical across arms.",
        "- Added opt-in immutable forensic snapshots without exposing vectors in normal response/debug payloads.",
        "- Added v2 fixture generation/validation, relevant-set metrics, gate/AUC diagnostics, atomic bundle writing, "
        "tamper detection, and offline reconstruction.",
        "- Primary implementation: `agent.py`, `run_longitudinal_eval.py`, `run_longitudinal_eval_v2.py`, "
        "`longitudinal_eval/evaluator_v2.py`, `build_fixture_v2.py`, `users_40_v2.json`, and tests.", "",
        "## Limitations", "",
        "This is a diagnostic fixture, not an unbiased estimate of user utility. Relevant sets are catalogue-text "
        "heuristics frozen before M3, and repeated category/trait templates reduce ecological diversity. Aggregate "
        "vector memory has no fact-level deletion, so lineage is evaluator evidence rather than retrievable memory "
        "items. The pre-registered HARMFUL result is a diagnosis of this frozen configuration and fixture—not a "
        "recommendation to tune the threshold.", "",
    ])


def write_bundle(output: str | Path, *, rows: Sequence[Mapping[str, Any]], arrays: Mapping[str, np.ndarray],
                 fixture_path: str | Path, catalog_path: str | Path, embedding_cache_path: str | Path,
                 embedding_space_id: str, source_paths: Sequence[str | Path] = (),
                 verification: Mapping[str, bool] | None = None) -> dict[str, Any]:
    root = Path(output); root.mkdir(parents=True, exist_ok=True)
    sessions_data = b"".join(canonical_json(row) + b"\n" for row in rows)
    metrics = build_metrics(rows); metrics_data = json.dumps(metrics, indent=2, sort_keys=True).encode()
    vector_tmp = root / ".vectors.tmp.npz"
    np.savez_compressed(vector_tmp, **arrays)
    vectors_data = vector_tmp.read_bytes(); vector_tmp.unlink()
    _atomic_write(root / "sessions.jsonl", sessions_data)
    _atomic_write(root / "metrics.json", metrics_data)
    _atomic_write(root / "vectors.npz", vectors_data)
    checks = dict(verification or {})
    trustworthy = all(checks.get(name, False) for name in (
        "fixture_validation", "paired_invariants", "artifact_hashes",
        "offline_reconstruction", "deterministic_rerun", "complete_test_suite"))
    probe_count = sum(row.get("record_type", "probe") == "probe" for row in rows)
    setup_count = sum(row.get("record_type") == "setup_update" for row in rows)
    report = _render_report(metrics, rows, trustworthy, checks)
    _atomic_write(root / "report.md", report.encode())
    artifacts = {name: sha256_file(root / name) for name in ("sessions.jsonl", "metrics.json", "vectors.npz", "report.md")}
    source_hashes = {str(Path(p)): sha256_file(p) for p in source_paths if Path(p).exists()}
    manifest = {"schema_version": SCHEMA_VERSION, "fixture_sha256": sha256_file(fixture_path),
                "catalog_sha256": sha256_file(catalog_path), "embedding_cache_sha256": sha256_file(embedding_cache_path),
                "embedding_space_id": embedding_space_id, "catalog_row_count": 50_000,
                "frozen_parameters": asdict(DEFAULT_VECTOR_MEMORY_CONFIG), "source_hashes": source_hashes,
                "code_revision": os.environ.get("GIT_COMMIT", "working-tree"), "artifact_hashes": artifacts,
                "verification": checks,
                "evaluator_classification": "EVALUATOR NOW TRUSTWORTHY" if trustworthy else "NOT TRUSTWORTHY"}
    _atomic_write(root / "manifest.json", json.dumps(manifest, indent=2, sort_keys=True).encode())
    return manifest


def verify_bundle(root: str | Path, *, reconstruct: bool = True) -> dict[str, Any]:
    path = Path(root); manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    mismatches = [name for name, digest in manifest["artifact_hashes"].items() if sha256_file(path / name) != digest]
    vector_failures: list[str] = []
    rank_failures: list[str] = []
    with np.load(path / "vectors.npz", allow_pickle=False) as arrays:
        rows = [json.loads(line) for line in (path / "sessions.jsonl").read_text(encoding="utf-8").splitlines() if line]
        for row in rows:
            for reference in row.get("array_references", {}).values():
                if reference is not None and not verify_vector_reference(reference, arrays):
                    vector_failures.append(reference["npz_key"])
            if reconstruct and row.get("record_type", "probe") == "probe":
                refs = row["array_references"]
                mask = arrays[refs["eligibility"]["npz_key"]]
                asins = arrays["catalog_asins"].tolist()
                for arm, score_key in (("m0", "s1"), ("m3", "s3")):
                    scores = arrays[refs[score_key]["npz_key"]]
                    ranked = rank_rows(scores, mask, asins)
                    ranked_asins = [asins[int(index)] for index in ranked]
                    raw, penalized = raw_and_penalized_rank(
                        ranked_asins, row["relevant_asins"], int(np.count_nonzero(mask)))
                    if (raw != row[f"{arm}_relevant_rank"] or
                            penalized != row[f"{arm}_penalized_rank"]):
                        rank_failures.append(f"{row['session_id']}:{arm}")
    metrics_rows = [json.loads(line) for line in (path / "sessions.jsonl").read_text(encoding="utf-8").splitlines() if line]
    metric_match = build_metrics(metrics_rows) == json.loads((path / "metrics.json").read_text(encoding="utf-8"))
    return {"valid": not mismatches and not vector_failures and not rank_failures and metric_match,
            "artifact_mismatches": mismatches, "vector_reference_failures": vector_failures,
            "rank_reconstruction_failures": rank_failures, "metrics_reconstructed": metric_match}


__all__ = ["SCHEMA_VERSION", "FIXTURE_SCHEMA_VERSION", "build_metrics", "classify_mechanism",
           "assert_paired_ranking_invariants", "distribution", "metric_summary", "rank_rows", "raw_and_penalized_rank", "run_v2",
           "sha256_file", "tied_auc", "validate_fixture_v2", "vector_reference",
           "verify_bundle", "verify_vector_reference", "write_bundle"]
