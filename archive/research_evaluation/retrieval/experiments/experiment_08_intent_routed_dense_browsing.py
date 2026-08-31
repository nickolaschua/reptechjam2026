from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
import re
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .config import DENSE_MAX_SEQ_LENGTH, MAX_TURNS, MODEL_ID, RRF_DEPTH, TOP_K
from .experiment_07_residual_failure_analysis import (
    deterministic_rrf,
    fallback_reasons,
    load_frozen_split,
)
from .harness import (
    Harness,
    experiment_logger,
    normalize,
    percentile_summary,
    rank_metrics,
    sha256,
    write_csv,
    write_json,
)


EXPERIMENT_NUMBER = 8
EXPERIMENT_SLUG = "intent_routed_dense_browsing"
EXPERIMENT_DIRECTORY = f"experiment_{EXPERIMENT_NUMBER:02d}_{EXPERIMENT_SLUG}"
CONTROL_METHOD = "exp7_same_parser_control"
TREATMENT_METHOD = "intent_routed_dense_browsing"
ROUTE_BUYING = "buying"
ROUTE_BROWSING = "browsing"

INITIAL_RE = re.compile(
    r"^\s*i(?:'|’)?m\s+looking\s+for\s+(.+?)(?:(?:,\s*)(?:but\s+)?(.+)|[.!?]\s*(.*))?$",
    re.IGNORECASE,
)
DISCLOSURE_RE = re.compile(r"^\s*for\s+that,?\s+what\s+matters\s+is\s*:\s*(.+)$", re.IGNORECASE)
OVERRIDE_RE = re.compile(
    r"^\s*actually,?\s*(?:please\s+)?(?:ignore|forget)\s+my\s+earlier\s+preference[.!?\s]+"
    r"what\s+i\s+need\s+is\s*:\s*(.+)$",
    re.IGNORECASE,
)
EXPLORATORY_CUES = (
    "still exploring",
    "just browsing",
    "browsing for now",
    "open to options",
    "weighing my options",
    "weighing options",
    "looking around",
    "comparing options",
    "not ready to buy",
    "haven't decided",
    "have not decided",
    "no fixed requirements",
)
BUYING_CUES = (
    "key requirement",
    "must have",
    "i need",
    "what i need",
    "required",
    "requirement is",
    "budget is",
    "under $",
    "instead",
)


def _clean(value: object) -> str:
    # Evaluator messages append a period. Exclamation/question marks may be
    # meaningful characters inside the catalog-derived constraint itself.
    return re.sub(r"\s+", " ", str(value or "")).strip().rstrip(".").strip()


def catalog_constraint_lexicon(h: Harness) -> set[str]:
    """Observable catalog-wide phrase vocabulary used to disambiguate `; `."""

    values: set[str] = set()
    def evaluator_clean(value: object) -> str:
        return re.sub(r"\s+", " ", str(value)).strip(" -;,.\t\n")[:180].rstrip()

    for product in h.products:
        features = product.get("features") or []
        if isinstance(features, list):
            values.update(normalize(evaluator_clean(item)) for item in features if normalize(item))
        details = product.get("details") or {}
        if isinstance(details, dict):
            values.update(normalize(evaluator_clean(f"{key}: {item}")) for key, item in details.items() if normalize(item))
        price = product.get("price")
        if price not in (None, ""):
            values.add(normalize(f"budget around ${price}"))
    values.update(MATERIALS_FOR_LEXICON)
    values.update(f"color: {color}" for color in COLORS_FOR_LEXICON)
    return values


MATERIALS_FOR_LEXICON = {"cotton", "polyester", "nylon", "leather", "wool", "spandex", "silk", "rayon", "fabric"}
COLORS_FOR_LEXICON = {"black", "white", "blue", "red", "pink", "green", "brown", "gray", "grey", "purple", "yellow", "orange"}


def split_disclosure(payload: str, known_constraints: set[str] | None = None) -> list[str]:
    """Recover up to two evaluator disclosures despite semicolons inside values."""

    value = re.sub(r"\s+", " ", str(payload or "")).strip()
    if value.endswith("."):
        value = value[:-1].strip()
    if not value or ";" not in value:
        return [value] if value else []
    if known_constraints:
        if normalize(value) in known_constraints:
            return [value]
        positions = [match.start() for match in re.finditer(r";\s*", value)]
        candidates: list[tuple[int, int, str, str]] = []
        for position in positions:
            left, right = value[:position].strip(), value[position + 1 :].strip()
            left_known = normalize(left) in known_constraints
            right_known = normalize(right) in known_constraints
            if left_known and right_known:
                candidates.append((int(left_known) + int(right_known), len(left) + len(right), left, right))
        if candidates:
            _, _, left, right = sorted(candidates, key=lambda item: (-item[0], -item[1], item[2], item[3]))[0]
            return [left, right]
    return [item.strip() for item in re.split(r"\s*;\s*", value) if item.strip()]


def detect_initial_intent(first_message: str) -> str:
    """Conservative, session-lockable binary detector using only the first message."""

    value = normalize(first_message)
    if any(cue in value for cue in BUYING_CUES):
        return ROUTE_BUYING
    if any(cue in value for cue in EXPLORATORY_CUES):
        return ROUTE_BROWSING
    # A concrete sentence after the category is an initial preference. Unknowns
    # deliberately fall through to the conservative buying route.
    match = INITIAL_RE.match(first_message)
    if match and _clean(match.group(3) or ""):
        return ROUTE_BUYING
    return ROUTE_BUYING


@dataclass(frozen=True)
class ObservableRetrievalInput:
    """The only fields admitted to Experiment 8 ranking before the oracle join."""

    category: str
    active_constraints: tuple[str, ...]
    locked_intent: str

    @property
    def phrases(self) -> tuple[str, ...]:
        return (self.category, *self.active_constraints)

    @property
    def query(self) -> str:
        return " ".join(self.phrases).strip()


@dataclass
class ObservableSessionState:
    category: str = ""
    locked_intent: str = ""
    constraints: list[str] = None  # type: ignore[assignment]
    raw_constraint_text: list[str] = None  # type: ignore[assignment]
    override_seed: str | None = None
    history: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.constraints = [] if self.constraints is None else self.constraints
        self.raw_constraint_text = [] if self.raw_constraint_text is None else self.raw_constraint_text
        self.history = [] if self.history is None else self.history


class ObservableStateParser:
    """Deterministic parser mirroring the Experiment 7 submission state parser."""

    def __init__(self, known_constraints: set[str] | None = None) -> None:
        self.state = ObservableSessionState()
        self.known_constraints = known_constraints

    @staticmethod
    def _append(state: ObservableSessionState, value: str) -> None:
        cleaned = re.sub(r"\s+", " ", str(value or "")).strip()
        if not cleaned:
            return
        state.raw_constraint_text.append(cleaned)
        if cleaned not in state.constraints:
            state.constraints.append(cleaned)

    def update(self, message: str, turn: int) -> ObservableRetrievalInput:
        state = self.state
        state.history.append(message)
        if turn == 1:
            state.locked_intent = detect_initial_intent(message)
            match = INITIAL_RE.match(message)
            if match:
                state.category = _clean(match.group(1))
                trailing = _clean(match.group(3) or "")
                if trailing:
                    if ":" in trailing and any(cue in normalize(trailing) for cue in BUYING_CUES):
                        trailing = trailing.split(":", 1)[1]
                    self._append(state, trailing)
                    state.override_seed = _clean(trailing) if state.locked_intent == ROUTE_BUYING else None
            else:
                state.category = "clothing item"
            return self.snapshot()

        disclosure = DISCLOSURE_RE.match(message)
        if disclosure:
            payload = disclosure.group(1)
            values = split_disclosure(payload, self.known_constraints)
            for value in values:
                self._append(state, value)
            return self.snapshot()

        override = OVERRIDE_RE.match(message)
        if override:
            if state.override_seed:
                seed = normalize(state.override_seed)
                state.constraints = [value for value in state.constraints if normalize(value) != seed]
            self._append(state, _clean(override.group(1)))
            state.override_seed = None
            return self.snapshot()

        # Generic negative/replace wording is retained for audit and removes an
        # explicitly named active phrase without inventing any new evidence.
        lowered = normalize(message)
        if any(cue in lowered for cue in ("ignore ", "forget ", "doesn't matter", "doesnt matter", "instead")):
            for existing in list(state.constraints):
                if normalize(existing) in lowered:
                    state.constraints.remove(existing)
        return self.snapshot()

    def snapshot(self) -> ObservableRetrievalInput:
        return ObservableRetrievalInput(
            self.state.category or "clothing item",
            tuple(self.state.constraints),
            self.state.locked_intent or ROUTE_BUYING,
        )


def exp7_exact_stateful_bm25(h: Harness, item: ObservableRetrievalInput) -> tuple[np.ndarray, np.ndarray, dict]:
    """The Experiment 7 selected cascade, without computing unused components."""

    exact_counts, _, _ = h.exact_scores(item.phrases)
    exact_order, exact_scores = h.exact_ranked(item.phrases)
    highest = int(exact_counts.max()) if exact_counts.size else 0
    highest_tier = int(np.count_nonzero(exact_counts == highest)) if highest else 0
    all_exact = int(np.count_nonzero(exact_counts == len(item.phrases)))
    reasons = fallback_reasons(item.active_constraints, all_exact, highest_tier)
    bm25_called = False
    if reasons:
        bm25_called = True
        bm25, _ = h.lexical.ranked(item.query, "bm25", RRF_DEPTH)
        order, scores = deterministic_rrf((exact_order[:RRF_DEPTH], bm25), h.ids)
    else:
        order, scores = exact_order, exact_scores
    return order[:TOP_K], scores[:TOP_K], {
        "exact_called": True,
        "bm25_called": bm25_called,
        "dense_called": False,
        "fallback_reasons": list(reasons),
    }


def routed_rank(h: Harness, item: ObservableRetrievalInput) -> tuple[np.ndarray, np.ndarray, dict]:
    if item.locked_intent == ROUTE_BROWSING:
        order, scores = h.dense.ranked(item.query, TOP_K)
        return order, scores, {
            "exact_called": False,
            "bm25_called": False,
            "dense_called": True,
            "fallback_reasons": [],
        }
    return exp7_exact_stateful_bm25(h, item)


def parse_observable_traces(
    h: Harness,
    messages: Mapping[str, Sequence[str]] | None = None,
    known_constraints: set[str] | None = None,
) -> list[ObservableRetrievalInput]:
    parsed: list[ObservableRetrievalInput] = []
    grouped = h.trace_by_session()
    lexicon = known_constraints if known_constraints is not None else catalog_constraint_lexicon(h)
    for sample in h.samples:
        sid = sample["sample_id"]
        parser = ObservableStateParser(lexicon)
        turn_messages = list(messages[sid]) if messages is not None else [state.message for state in grouped[sid]]
        if len(turn_messages) != MAX_TURNS:
            raise RuntimeError(f"Observable trace for {sid} has {len(turn_messages)} turns")
        for turn, message in enumerate(turn_messages, 1):
            parsed.append(parser.update(message, turn))
    return parsed


def assert_parser_identity(h: Harness, parsed: Sequence[ObservableRetrievalInput]) -> None:
    mismatches = []
    for index, (state, item) in enumerate(zip(h.traces, parsed)):
        if normalize(state.category) != normalize(item.category) or tuple(map(normalize, state.active_constraints)) != tuple(map(normalize, item.active_constraints)):
            mismatches.append((index, state.sample_id, state.turn, state.phrases, item.phrases))
    if len(parsed) != len(h.traces) or mismatches:
        raise RuntimeError(f"Same-parser identity failed: {mismatches[:5]}")


def _freeze(
    h: Harness,
    parsed: Sequence[ObservableRetrievalInput],
    ranker: Callable[[Harness, ObservableRetrievalInput], tuple[np.ndarray, np.ndarray, dict]],
) -> tuple[list[tuple[int, ...]], list[dict], list[float]]:
    slates: list[tuple[int, ...]] = []
    diagnostics: list[dict] = []
    latencies: list[float] = []
    cache: dict[ObservableRetrievalInput, tuple[tuple[int, ...], dict, float]] = {}
    for item in parsed:
        cached = cache.get(item)
        if cached is None:
            started = time.perf_counter()
            order, _, diagnostic = ranker(h, item)
            elapsed = (time.perf_counter() - started) * 1000.0
            cached = (tuple(int(value) for value in order[:TOP_K]), diagnostic, elapsed)
            cache[item] = cached
        slate, diagnostic, elapsed = cached
        slates.append(slate)
        diagnostics.append(dict(diagnostic))
        latencies.append(elapsed)
    return slates, diagnostics, latencies


def _outcomes(h: Harness, slates: Sequence[Sequence[int]]) -> list[dict]:
    sessions: list[dict] = []
    offset = 0
    for sample in h.samples:
        turns = h.traces[offset : offset + MAX_TURNS]
        rankings = slates[offset : offset + MAX_TURNS]
        offset += MAX_TURNS
        first_hit = best_rank = None
        for state, ranking in zip(turns, rankings):
            values = list(ranking)
            target = h.id_to_idx[state.target_asin]
            if state.override_applied and target in values:
                first_hit, best_rank = state.turn, values.index(target) + 1
                break
        sessions.append({
            "sample_id": sample["sample_id"],
            "scenario_type": sample["scenario_type"],
            "hit": first_hit is not None,
            "first_hit_turn": first_hit,
            "best_rank": best_rank,
            "reciprocal_rank": 0.0 if best_rank is None else 1.0 / best_rank,
        })
    return sessions


def _metric_subset(rows: Sequence[dict], ids: set[str]) -> dict:
    selected = [row for row in rows if row["sample_id"] in ids]
    result = rank_metrics(selected)
    result["scenario_metrics"] = {
        scenario: rank_metrics([row for row in selected if row["scenario_type"] == scenario])
        for scenario in sorted({row["scenario_type"] for row in selected})
    }
    return result


def _route_metrics(rows: Sequence[dict], route_by_id: Mapping[str, str]) -> dict:
    return {
        route: rank_metrics([row for row in rows if route_by_id[row["sample_id"]] == route])
        for route in (ROUTE_BUYING, ROUTE_BROWSING)
    }


def compare_sessions(baseline: Sequence[dict], candidate: Sequence[dict], ids: set[str]) -> dict:
    left = {row["sample_id"]: row for row in baseline if row["sample_id"] in ids}
    right = {row["sample_id"]: row for row in candidate if row["sample_id"] in ids}
    rescues = sorted(sid for sid in left if not left[sid]["hit"] and right[sid]["hit"])
    regressions = sorted(sid for sid in left if left[sid]["hit"] and not right[sid]["hit"])
    return {
        "baseline_hard_failures": sum(not row["hit"] for row in left.values()),
        "hard_failure_rescues": len(rescues),
        "rescued_sample_ids": rescues,
        "regressions": len(regressions),
        "regressed_sample_ids": regressions,
    }


SYNONYMS = {
    "black": "ebony", "white": "ivory", "blue": "azure", "red": "crimson",
    "pink": "rose", "green": "verdant", "brown": "umber", "gray": "charcoal",
    "grey": "charcoal", "purple": "violet", "yellow": "golden", "orange": "tangerine",
    "cotton": "natural fibre", "polyester": "synthetic fibre", "nylon": "polyamide",
    "leather": "hide", "wool": "fleece", "spandex": "elastane", "silk": "satin-like fibre",
    "rayon": "viscose", "budget": "price ceiling", "color": "shade", "fit": "cut",
    "size": "dimensions", "style": "look", "machine": "automatic", "wash": "launder",
    "waterproof": "weatherproof", "running": "jogging", "hiking": "trekking",
}


def _break_single_token(value: str) -> str:
    if len(value) < 2:
        return f"descriptor {ord(value) if value else 0}"
    pivot = max(1, len(value) // 2)
    return value[:pivot] + "-" + value[pivot:]


def transform_constraint(value: str, transform: str) -> str:
    original = _clean(value)
    words = re.findall(r"[A-Za-z0-9$.-]+", original)
    if transform == "synonym_substitution":
        changed = [SYNONYMS.get(word.lower(), word) for word in words]
        result = " ".join(changed)
    elif transform == "clause_reordering":
        result = " ".join(reversed(words))
    elif transform == "lexical_compression":
        result = " ".join(words[::2])
    elif transform == "punctuation_changes":
        result = f"({original})!"
    else:
        raise ValueError(f"Unknown robustness transform: {transform}")
    if transform != "punctuation_changes" and normalize(original) in normalize(result):
        if len(words) == 1:
            result = _break_single_token(words[0])
        else:
            result = " / ".join(reversed(words))
    if transform != "punctuation_changes" and normalize(original) in normalize(result):
        raise RuntimeError(f"Robustness transform retained original phrase: {original!r} -> {result!r}")
    return result or _break_single_token(original)


ROBUSTNESS_TRANSFORMS = (
    "synonym_substitution",
    "clause_reordering",
    "lexical_compression",
    "punctuation_changes",
)


def transform_message(message: str, transform: str, known_constraints: set[str] | None = None) -> str:
    initial = INITIAL_RE.match(message)
    if initial:
        category = _clean(initial.group(1))
        exploratory = _clean(initial.group(2) or "")
        trailing = _clean(initial.group(3) or "")
        if exploratory:
            cue = "weighing my options" if transform != "punctuation_changes" else "still exploring!"
            return f"I'm looking for {category}, but I'm {cue}."
        if trailing:
            prefix, separator, payload = trailing.partition(":")
            transformed = transform_constraint(payload if separator else trailing, transform)
            return f"I'm looking for {category}. {prefix + ':' if separator else ''} {transformed}."
        return message
    disclosure = DISCLOSURE_RE.match(message)
    if disclosure:
        values = split_disclosure(disclosure.group(1), known_constraints)
        changed = [transform_constraint(value, transform) for value in values]
        return "For that, what matters is: " + "; ".join(changed) + "."
    override = OVERRIDE_RE.match(message)
    if override:
        changed = transform_constraint(override.group(1), transform)
        return f"Actually, ignore my earlier preference. What I need is: {changed}."
    return message


def _load_exp7_rows(path: Path) -> list[dict]:
    if not path.exists():
        raise RuntimeError(f"Frozen Experiment 7 turn rows are missing: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 200 * MAX_TURNS:
        raise RuntimeError("Frozen Experiment 7 turn rows are incomplete")
    return rows


def _hash_tree(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted((value for value in path.rglob("*") if value.is_file()), key=lambda value: value.as_posix()):
        digest.update(item.relative_to(path).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256(item)))
    return digest.hexdigest()


def _plot(path: Path, metrics: Mapping[str, dict]) -> None:
    labels = list(metrics)
    scores = [metrics[name]["technical_score"] for name in labels]
    mrr = [metrics[name]["mrr"] for name in labels]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    ax.bar(x - 0.18, scores, 0.36, label="TechnicalScore", color="#277da1")
    ax.bar(x + 0.18, mrr, 0.36, label="MRR", color="#f8961e")
    ax.set_xticks(x, labels, rotation=18)
    ax.set_ylim(0, 1)
    ax.set_title("Experiment 8 held-out comparison")
    ax.grid(axis="y", alpha=.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def reproduce_submission_baseline(h: Harness, directory: Path, logger: logging.Logger) -> dict:
    """Replay the untouched submission agent and require full JSON identity."""

    project = h.repo / "techjam-conversational-search"
    evaluator = project / "evaluator" / "local_evaluator.py"
    frozen = project / "current_agent_results.json"
    output = directory / "submission_baseline_reproduction.json"
    if not frozen.exists():
        raise RuntimeError(f"Frozen current submission result is missing: {frozen}")
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(project)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    command = [
        sys.executable,
        str(evaluator),
        "--catalog", str(h.catalog_path),
        "--dataset", str(h.public_path),
        "--output", str(output),
    ]
    logger.info("Reproducing frozen current submission result before Experiment 8 comparison")
    subprocess.run(command, cwd=project, env=environment, check=True, stdout=subprocess.DEVNULL)
    expected = json.loads(frozen.read_text(encoding="utf-8"))
    actual = json.loads(output.read_text(encoding="utf-8"))
    if actual != expected:
        raise RuntimeError("Current submission baseline reproduction changed")
    return {
        "passed": True,
        "full_json_identity": True,
        "session_identity_count": len(actual.get("sessions", [])),
        "recommended_technical_score": actual["recommended_technical_score"],
        "mrr": actual["mrr"],
        "frozen_path": str(frozen.relative_to(h.repo)),
        "frozen_sha256": sha256(frozen),
        "reproduction_path": str(output.relative_to(h.repo)),
        "reproduction_sha256": sha256(output),
        "command": subprocess.list2cmdline(command),
    }


def run_08(h: Harness, root_logger: logging.Logger) -> dict:
    directory = h.results_dir / EXPERIMENT_DIRECTORY
    logger = experiment_logger(root_logger, directory)
    started = time.perf_counter()
    logger.info("AGENT-REALISTIC EVALUATION: beginning intent-routed dense browsing")

    exp7_dir = h.results_dir / "experiment_07_residual_failure_analysis"
    exp7_source = h.repo / "nickolas" / "experiments" / "experiment_07_residual_failure_analysis.py"
    submission_agent = h.repo / "techjam-conversational-search" / "starter" / "agent.py"
    split_path = h.results_dir / "experiment_06_slate_width_counterfactuals" / "metrics.json"
    public_ids = [sample["sample_id"] for sample in h.samples]
    calibration, evaluation, split = load_frozen_split(split_path, public_ids)
    submission_reproduction = reproduce_submission_baseline(h, directory, logger)

    h.dense.ensure()
    if h.dense.embeddings is None or h.dense.embeddings.dtype != np.float32:
        raise RuntimeError("Experiment 8 requires normalized float32 MiniLM catalog embeddings")

    lexicon = catalog_constraint_lexicon(h)
    parsed = parse_observable_traces(h, known_constraints=lexicon)
    assert_parser_identity(h, parsed)
    h.dense.preload_queries(item.query for item in parsed if item.locked_intent == ROUTE_BROWSING)

    control_slates, control_diag, control_latency = _freeze(h, parsed, exp7_exact_stateful_bm25)
    treatment_slates, treatment_diag, treatment_latency = _freeze(h, parsed, routed_rank)

    # Verify every buying turn is bit-for-bit identical to both the local control
    # and the frozen Experiment 7 artifact.
    exp7_rows = _load_exp7_rows(exp7_dir / "rows.csv")
    buying_turns = 0
    for index, (item, control, treatment, frozen_row) in enumerate(zip(parsed, control_slates, treatment_slates, exp7_rows)):
        expected_ids = tuple(json.loads(frozen_row["exact_stateful_bm25_rrf_top_10"]))
        actual_ids = tuple(h.ids[value] for value in control)
        if actual_ids != expected_ids:
            raise RuntimeError(f"Experiment 7 control identity failed at turn row {index}")
        if item.locked_intent == ROUTE_BUYING:
            buying_turns += 1
            if treatment != control:
                raise RuntimeError(f"Buying route changed Experiment 7 output at turn row {index}")
    if any(diagnostic["exact_called"] or diagnostic["bm25_called"] for item, diagnostic in zip(parsed, treatment_diag) if item.locked_intent == ROUTE_BROWSING):
        raise RuntimeError("A browsing treatment turn invoked exact or BM25 retrieval")

    control_sessions = _outcomes(h, control_slates)
    treatment_sessions = _outcomes(h, treatment_slates)
    route_by_id = {
        sample["sample_id"]: parsed[index * MAX_TURNS].locked_intent
        for index, sample in enumerate(h.samples)
    }
    expected_route = {
        sample["sample_id"]: (ROUTE_BROWSING if sample["scenario_type"] in {"browsing", "boundary"} else ROUTE_BUYING)
        for sample in h.samples
    }
    routing_correct = {sid: route_by_id[sid] == expected_route[sid] for sid in route_by_id}

    method_metrics = {
        CONTROL_METHOD: {
            "full": _metric_subset(control_sessions, set(public_ids)),
            "calibration": _metric_subset(control_sessions, calibration),
            "evaluation": _metric_subset(control_sessions, evaluation),
            "route_metrics": _route_metrics(control_sessions, route_by_id),
        },
        TREATMENT_METHOD: {
            "full": _metric_subset(treatment_sessions, set(public_ids)),
            "calibration": _metric_subset(treatment_sessions, calibration),
            "evaluation": _metric_subset(treatment_sessions, evaluation),
            "route_metrics": _route_metrics(treatment_sessions, route_by_id),
        },
    }
    comparisons = {
        "full": compare_sessions(control_sessions, treatment_sessions, set(public_ids)),
        "calibration": compare_sessions(control_sessions, treatment_sessions, calibration),
        "evaluation": compare_sessions(control_sessions, treatment_sessions, evaluation),
    }

    # Frozen deterministic browsing robustness suite. Only detector-routed
    # browsing sessions are included; control and treatment consume the same
    # transformed messages and parsing output.
    grouped = h.trace_by_session()
    browsing_ids = {sid for sid, route in route_by_id.items() if route == ROUTE_BROWSING}
    robustness_rows: list[dict] = []
    robustness_sessions: dict[str, dict[str, list[dict]]] = {}
    original_browsing_control = [row for row in control_sessions if row["sample_id"] in browsing_ids]
    original_browsing_treatment = [row for row in treatment_sessions if row["sample_id"] in browsing_ids]
    for transform in ROBUSTNESS_TRANSFORMS:
        transformed_messages = {
            sid: [transform_message(state.message, transform, lexicon) for state in grouped[sid]]
            for sid in browsing_ids
        }
        # Non-browsing messages are retained solely to preserve the fixed 200 x
        # ten row alignment used by the replay helper.
        all_messages = {
            sid: transformed_messages.get(sid, [state.message for state in grouped[sid]])
            for sid in grouped
        }
        transformed = parse_observable_traces(h, all_messages, known_constraints=set())
        for sample_index, sample in enumerate(h.samples):
            if sample["sample_id"] in browsing_ids and transformed[sample_index * MAX_TURNS].locked_intent != ROUTE_BROWSING:
                raise RuntimeError(f"Robustness transform unlocked browsing route for {sample['sample_id']}")
        h.dense.preload_queries(
            item.query for sample_index, sample in enumerate(h.samples)
            if sample["sample_id"] in browsing_ids
            for item in transformed[sample_index * MAX_TURNS : (sample_index + 1) * MAX_TURNS]
        )
        transformed_control, _, _ = _freeze(h, transformed, exp7_exact_stateful_bm25)
        transformed_treatment, transformed_diag, _ = _freeze(h, transformed, routed_rank)
        if any(
            diagnostic["exact_called"] or diagnostic["bm25_called"]
            for sample_index, sample in enumerate(h.samples) if sample["sample_id"] in browsing_ids
            for diagnostic in transformed_diag[sample_index * MAX_TURNS : (sample_index + 1) * MAX_TURNS]
        ):
            raise RuntimeError("A transformed browsing treatment turn invoked exact or BM25")
        control_out = [row for row in _outcomes(h, transformed_control) if row["sample_id"] in browsing_ids]
        treatment_out = [row for row in _outcomes(h, transformed_treatment) if row["sample_id"] in browsing_ids]
        robustness_sessions[transform] = {CONTROL_METHOD: control_out, TREATMENT_METHOD: treatment_out}
        for method, original, changed in (
            (CONTROL_METHOD, original_browsing_control, control_out),
            (TREATMENT_METHOD, original_browsing_treatment, treatment_out),
        ):
            before, after = rank_metrics(original), rank_metrics(changed)
            robustness_rows.append({
                "transform": transform,
                "method": method,
                "sample_count": len(changed),
                "technical_score": after["technical_score"],
                "mrr": after["mrr"],
                "hit_rate_at_10": after["hit_rate_at_10"],
                "technical_score_degradation": round(before["technical_score"] - after["technical_score"], 6),
                "mrr_degradation": round(before["mrr"] - after["mrr"], 6),
            })

    turn_rows: list[dict] = []
    route_rows: list[dict] = []
    for index, (state, item, control, treatment, cdiag, tdiag) in enumerate(
        zip(h.traces, parsed, control_slates, treatment_slates, control_diag, treatment_diag)
    ):
        turn_rows.append({
            "sample_id": state.sample_id,
            "split": "calibration" if state.sample_id in calibration else "evaluation",
            "turn": state.turn,
            "locked_route": item.locked_intent,
            "category": item.category,
            "active_constraints": list(item.active_constraints),
            "query": item.query,
            "control_top_10": [h.ids[value] for value in control],
            "treatment_top_10": [h.ids[value] for value in treatment],
            "control_exact_called": cdiag["exact_called"],
            "control_bm25_called": cdiag["bm25_called"],
            "treatment_exact_called": tdiag["exact_called"],
            "treatment_bm25_called": tdiag["bm25_called"],
            "treatment_dense_called": tdiag["dense_called"],
            "oracle_joined_after_freeze": True,
            "oracle_scenario_type": state.scenario_type,
            "oracle_target_asin": state.target_asin,
        })
        if state.turn == 1:
            route_rows.append({
                "sample_id": state.sample_id,
                "first_message": state.message,
                "detected_route": item.locked_intent,
                "expected_route_after_freeze": expected_route[state.sample_id],
                "correct": routing_correct[state.sample_id],
                "route_locked_all_turns": len({value.locked_intent for value in parsed[index : index + MAX_TURNS]}) == 1,
                "oracle_scenario_type": state.scenario_type,
            })

    dense_cache = h.results_dir / "cache" / f"dense_{h.catalog_hash[:16]}_minilm_seq{DENSE_MAX_SEQ_LENGTH}.npy"
    model_dir = h.results_dir / "cache" / "models"
    hashes = {
        "experiment_07_baseline_source": {"path": str(exp7_source.relative_to(h.repo)), "sha256": sha256(exp7_source)},
        "submission_agent_baseline": {"path": str(submission_agent.relative_to(h.repo)), "sha256": sha256(submission_agent)},
        "submission_baseline_result": {"path": submission_reproduction["frozen_path"], "sha256": submission_reproduction["frozen_sha256"]},
        "frozen_split": {"path": str(split_path.relative_to(h.repo)), "sha256": sha256(split_path)},
        "catalog": {"path": str(h.catalog_path.relative_to(h.repo)), "sha256": h.catalog_hash},
        "public_set": {"path": str(h.public_path.relative_to(h.repo)), "sha256": h.public_hash},
        "dense_embeddings": {"path": str(dense_cache.relative_to(h.repo)), "sha256": sha256(dense_cache)},
        "dense_model": {"model_id": MODEL_ID, "tree_sha256": _hash_tree(model_dir)},
    }
    source_path = Path(__file__).resolve()
    hashes["source"] = {"path": str(source_path.relative_to(h.repo)), "sha256": sha256(source_path)}

    latency = {
        CONTROL_METHOD: percentile_summary(control_latency),
        TREATMENT_METHOD: percentile_summary(treatment_latency),
        "unit": "milliseconds_per_unique-state ranking (cache hits retain originating measurement)",
    }
    metrics = {
        "experiment": EXPERIMENT_NUMBER,
        "slug": EXPERIMENT_SLUG,
        "split": split,
        "model": {"id": MODEL_ID, "max_sequence_length": DENSE_MAX_SEQ_LENGTH, "dtype": "float32", "normalized": True, "similarity": "cosine"},
        "retrieval_contract": {
            "input_fields_before_freeze": ["category", "active_constraints", "locked_intent"],
            "excluded_before_freeze": ["sample_id", "scenario_type", "target_asin", "oracle_card", "user_profile"],
            "ask_attribute": "other",
            "top_k": TOP_K,
            "browsing_exact_calls": 0,
            "browsing_bm25_calls": 0,
            "rankings_frozen_before_oracle_join": True,
        },
        "identity": {
            "same_parser_rankings_match_frozen_experiment_07": True,
            "buying_turns_bit_for_bit_identical": buying_turns,
            "route_locked": all(row["route_locked_all_turns"] for row in route_rows),
        },
        "submission_baseline_reproduction": submission_reproduction,
        "routing": {
            "accuracy": round(sum(routing_correct.values()) / len(routing_correct), 6),
            "correct": sum(routing_correct.values()),
            "total": len(routing_correct),
            "confusion": {
                expected: {actual: sum(expected_route[sid] == expected and route_by_id[sid] == actual for sid in route_by_id) for actual in (ROUTE_BUYING, ROUTE_BROWSING)}
                for expected in (ROUTE_BUYING, ROUTE_BROWSING)
            },
        },
        "method_metrics": method_metrics,
        "comparisons_to_control": comparisons,
        "latency": latency,
        "paraphrase_robustness": robustness_rows,
        "hashes": hashes,
        "submission_agent_modified": False,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }

    _plot(directory / "route_comparison.png", {
        CONTROL_METHOD: method_metrics[CONTROL_METHOD]["evaluation"],
        TREATMENT_METHOD: method_metrics[TREATMENT_METHOD]["evaluation"],
    })
    write_json(directory / "metrics.json", metrics)
    write_csv(directory / "rows.csv", turn_rows)
    write_json(directory / "sessions.json", {CONTROL_METHOD: control_sessions, TREATMENT_METHOD: treatment_sessions})
    write_csv(directory / "route_diagnostics.csv", route_rows)
    write_json(directory / "route_diagnostics.json", route_rows)
    write_csv(directory / "comparisons.csv", [{"split": key, **value} for key, value in comparisons.items()])
    write_json(directory / "comparisons.json", comparisons)
    write_csv(directory / "paraphrase_results.csv", robustness_rows)
    write_json(directory / "paraphrase_results.json", {"summary": robustness_rows, "sessions": robustness_sessions})
    write_json(directory / "latency.json", latency)
    write_json(directory / "hashes.json", hashes)
    (directory / "source_snapshot.py").write_bytes(source_path.read_bytes())
    write_json(directory / "source_snapshot.json", {
        "source": hashes["source"],
        "snapshot_sha256": sha256(directory / "source_snapshot.py"),
        "identical": sha256(directory / "source_snapshot.py") == hashes["source"]["sha256"],
    })

    held = method_metrics[TREATMENT_METHOD]["evaluation"]
    control_held = method_metrics[CONTROL_METHOD]["evaluation"]
    summary = f"""# Experiment 8 — Intent-routed dense browsing

> **AGENT-REALISTIC RANKING.** The session-locked route and rankings were computed from the first message, category, and active observable constraints. Scenario labels and targets were joined only after rankings were frozen.

The unchanged submission agent first reproduced its frozen **{submission_reproduction['recommended_technical_score']:.6f}** score and all **{submission_reproduction['session_identity_count']} session outcomes** exactly. The same-parser research control then reproduced all frozen Experiment 7 turn slates. All **{buying_turns} buying-routed turns** were bit-for-bit identical in the routed treatment. Browsing used normalized float32 `{MODEL_ID}` embeddings and pure cosine Top-10; instrumentation recorded **zero exact and zero BM25 calls** on that route.

Intent routing accuracy was **{metrics['routing']['accuracy']:.1%}**. On the frozen 140-session held-out set, the routed treatment scored **{held['technical_score']:.6f}** TechnicalScore and **{held['mrr']:.6f}** MRR versus **{control_held['technical_score']:.6f}** and **{control_held['mrr']:.6f}** for Experiment 7. It rescued **{comparisons['evaluation']['hard_failure_rescues']}** control failures and introduced **{comparisons['evaluation']['regressions']}** regressions.

The paraphrase suite is a deterministic browsing stress test, not a replacement for official scoring. The submission agent was not modified.
"""
    (directory / "summary.md").write_text(summary, encoding="utf-8")
    logger.info("Completed %s in %.2fs", EXPERIMENT_DIRECTORY, metrics["elapsed_seconds"])
    return metrics
