"""Distil intent_of into a classifier that needs no LLM parse.

Why bother, when the pipeline parses every turn anyway: the parse costs ~18s of
grammar-constrained decode. A browsing turn can go straight to dense retrieval on
the RAW utterance (fuse_rank's dense_raw path needs no parse), so knowing intent
before parsing lets a browsing turn skip the parse entirely. That makes browsing
PRECISION the metric that matters - a buying turn mislabelled browsing loses its
filters, while the reverse only wastes a parse we would have done anyway.

Labels come from bolt_on.intent_of, so the ceiling is the rule. The regex parts of
the rule (exploring cue, compatibility, model code) are exact and free, so they
stay as overrides; the model only has to learn the slot-count residual.

    python3 intent_clf.py              # train + report
    python3 intent_clf.py --curve      # also learning curve
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, cross_val_predict
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion, make_pipeline
from sklearn.preprocessing import StandardScaler

BENCH = Path(__file__).resolve().parent
LAB = BENCH.parent
WINSTON = LAB.parent
for p in (WINSTON / "experiments", WINSTON, LAB, BENCH):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from bolt_on import intent_of, model_code, _COMPAT_RE, _EXPLORING_RE  # noqa: E402
from nlp_parse import (_MATERIALS, _SIZE_NUMERIC, _SIZE_WORDS,  # noqa: E402
                       catalog_stores)

# The only lexicon nlp_parse does not already own. Colour is a buying signal in the
# problem statement but never a hard filter, so tier_of never needed a list.
_COLORS = frozenset({
    "black", "white", "grey", "gray", "red", "blue", "navy", "green", "olive",
    "yellow", "orange", "purple", "violet", "pink", "brown", "tan", "beige",
    "cream", "ivory", "gold", "silver", "bronze", "maroon", "burgundy", "teal",
    "turquoise", "coral", "mint", "lavender", "khaki", "charcoal", "rose",
})
_PRICE_RE = re.compile(r"\$\s*\d|\b(?:under|below|less than|over|above|around|about)\s+\$?\d|\b\d+\s*(?:dollars|bucks|usd)\b", re.I)
_TOK = re.compile(r"[a-z0-9$.]+")
_BARE_SIZES = frozenset({"s", "m", "l"})


def lex_features(message: str) -> list[float]:
    """One binary per buying signal the problem statement names, read straight off
    the text with the lexicons nlp_parse already uses to validate slots.

    This is the question 'can regex alone reproduce a label defined as a count of
    parse slots' - if yes, intent needs no 18s LLM parse.
    """
    low = message.lower()
    toks = _TOK.findall(low)
    tset = set(toks)
    bigrams = {f"{a} {b}" for a, b in zip(toks, toks[1:])}
    stores = catalog_stores()
    # "i'm" -> ["i", "m"], and "m" is a _SIZE_WORD. The bare letter sizes and any
    # numeric size only count when the word "size" is actually in the message.
    said_size = "size" in tset
    has_size = (any(t in _SIZE_WORDS and (t not in _BARE_SIZES or said_size) for t in toks)
                or (said_size and any(_SIZE_NUMERIC.match(t) for t in toks)))
    return [
        float(bool(tset & _MATERIALS)),
        float(bool(tset & _COLORS)),
        float(has_size),
        float(bool(_PRICE_RE.search(low))),
        float(bool((tset | bigrams) & stores)),
        float(bool(model_code(message))),
        float(bool(_COMPAT_RE.search(message))),
        float(bool(_EXPLORING_RE.search(message))),
        float(len(toks)),
    ]


class LexFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.asarray([lex_features(m) for m in X])

SEED = 20260830
HOLDOUT_PATH = BENCH / ".cache" / "holdout_asins.json"
HOLDOUT_FRAC = 0.2


def holdout_asins() -> set[str]:
    """A fifth of the PRODUCTS, frozen to disk the first time and never resampled.

    Sampled from cases.jsonl in full, not from the parsed subset, so the split does
    not drift as parse_all.py fills parses.jsonl in. Written once: everything after
    this is measured on products the model has never been fitted on, and it stays
    honest only as long as nobody tunes against it.
    """
    if HOLDOUT_PATH.exists():
        return set(json.loads(HOLDOUT_PATH.read_text()))
    asins = sorted({json.loads(l)["asin"] for l in (BENCH / "cases.jsonl").open() if l.strip()})
    rng = __import__("random").Random(SEED)
    picked = sorted(rng.sample(asins, round(len(asins) * HOLDOUT_FRAC)))
    HOLDOUT_PATH.parent.mkdir(exist_ok=True)
    HOLDOUT_PATH.write_text(json.dumps(picked, indent=1))
    return set(picked)


def override(message: str) -> str | None:
    """The deterministic half of intent_of. None = the model decides."""
    if _EXPLORING_RE.search(message):
        return "browsing"
    if _COMPAT_RE.search(message) or model_code(message):
        return "buying"
    return None


def load(split: str = "dev") -> tuple[list[str], list[str], list[str]]:
    """(utterances, labels, asins). The asin is the CV group: cases.jsonl writes
    ~4 utterances per product, so a random split trains and tests on siblings of
    the same listing and reads several points high."""
    held = holdout_asins()
    cases = {json.loads(l)["case_id"]: json.loads(l) for l in (BENCH / "cases.jsonl").open() if l.strip()}
    X, y, g = [], [], []
    for line in (BENCH / "parses.jsonl").open():
        if not line.strip():
            continue
        r = json.loads(line)
        c = cases.get(r["case_id"])
        if c and (split == "all" or (c["asin"] in held) == (split == "test")):
            X.append(c["utterance"])
            y.append(intent_of(r["parse"], c["utterance"]))
            g.append(c["asin"])
    return X, y, g


def tfidf():
    return TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, strip_accents="unicode")


def model(kind: str = "tfidf") -> object:
    lr = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED)
    if kind == "tfidf":
        return make_pipeline(tfidf(), lr)
    if kind == "lex":
        return make_pipeline(LexFeatures(), StandardScaler(), lr)
    return make_pipeline(FeatureUnion([("t", tfidf()), ("l", make_pipeline(LexFeatures(), StandardScaler()))]), lr)


def predict(clf, messages: list[str]) -> list[str]:
    """Regex overrides win; the model fills the rest."""
    out = list(clf.predict(messages))
    return [override(m) or p for m, p in zip(messages, out)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--curve", action="store_true")
    ap.add_argument("--features", default="tfidf", choices=("tfidf", "lex", "both"))
    ap.add_argument("--test", action="store_true",
                    help="fit on dev, report ONCE on the frozen holdout. Do not tune against this.")
    args = ap.parse_args()

    X, y, groups = load("dev")
    n_buy = sum(1 for v in y if v == "buying")
    print(f"labelled cases {len(X)}   buying {n_buy}  browsing {len(y) - n_buy}")
    majority = max(n_buy, len(y) - n_buy) / len(y)
    print(f"majority-class baseline {majority:.3f}\n")

    for name, cv, kw in (("random 5-fold (LEAKY - siblings of the same asin span folds)",
                          StratifiedKFold(5, shuffle=True, random_state=SEED), {}),
                         ("grouped 5-fold by asin (honest)",
                          StratifiedGroupKFold(5, shuffle=True, random_state=SEED),
                          {"groups": groups})):
        raw = cross_val_predict(model(args.features), X, y, cv=cv, **kw)
        final = [override(m) or p for m, p in zip(X, raw)]
        print(f"--- {name}")
        print(classification_report(y, final, digits=3))
    print("confusion below is the grouped split")
    print("confusion (rows=rule label, cols=predicted), labels", sorted(set(y)))
    print(confusion_matrix(y, final, labels=sorted(set(y))))

    clf = model(args.features).fit(X, y)
    if args.test:
        Xt, yt, _ = load("test")
        pred = predict(clf, Xt)
        print(f"\n=== FROZEN HOLDOUT: {len(set(holdout_asins()))} products, {len(Xt)} cases ===")
        print(classification_report(yt, pred, digits=3))
    cache = Path(__file__).resolve().parent / ".cache" / "spec_probes.json"
    if cache.exists():
        probes = json.loads(cache.read_text())
        print("\nheld-out problem-statement probes (never trained on):")
        ok = 0
        for gold, msg in probes:
            pred = predict(clf, [msg])[0]
            ok += pred == gold
            print(f"  {'OK ' if pred == gold else 'MISS'} gold={gold:8} pred={pred:8}  {msg[:56]}")
        print(f"  -> {ok}/{len(probes)}")
    else:
        print(f"\n(no {cache} - skipping held-out probes)")

    if args.curve:
        print("\nlearning curve (5-fold CV accuracy at n training cases):")
        rng = np.random.default_rng(SEED)
        idx = rng.permutation(len(X))
        for frac in (0.25, 0.5, 0.75, 1.0):
            k = max(20, int(len(X) * frac))
            sub = idx[:k]
            Xs = [X[i] for i in sub]
            ys = [y[i] for i in sub]
            gs = [groups[i] for i in sub]
            if len(set(ys)) < 2:
                continue
            p = cross_val_predict(model(args.features), Xs, ys, groups=gs,
                                  cv=StratifiedGroupKFold(5, shuffle=True, random_state=SEED))
            p = [override(m) or v for m, v in zip(Xs, p)]
            print(f"  n={k:5}  acc={np.mean([a == b for a, b in zip(ys, p)]):.3f}")


if __name__ == "__main__":
    main()
