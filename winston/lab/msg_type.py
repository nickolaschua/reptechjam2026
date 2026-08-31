"""Message-type classifier: TF-IDF + logistic regression on the benchmark's free
`style` labels (each utterance was generated under one of 8 style instructions).

Held out by PRODUCT (GroupKFold) so no utterance about a test product is seen in
training, and by GENERATOR so the model is not just learning llama's dialect.
Rerun as cases.jsonl grows; ms-latency at inference, no LLM.

    python3 msg_type.py
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline

CASES = Path(__file__).resolve().parent / "bench" / "cases.jsonl"


def model():
    return make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1),
        LogisticRegression(max_iter=2000, C=3.0, class_weight="balanced"),
    )


def main() -> None:
    rows = [json.loads(l) for l in CASES.open() if l.strip()]
    X = np.array([r["utterance"] for r in rows]); y = np.array([r["style"] for r in rows])
    groups = np.array([r["asin"] for r in rows]); gen = np.array([r["generator"] for r in rows])
    print(f"{len(rows)} cases  labels={dict(Counter(y))}  generators={dict(Counter(gen))}")
    print(f"majority-class accuracy = {Counter(y).most_common(1)[0][1] / len(y):.3f}\n")

    # 1. grouped 5-fold by product
    pred = np.empty_like(y)
    for tr, te in GroupKFold(n_splits=min(5, len(set(groups)))).split(X, y, groups):
        pred[te] = model().fit(X[tr], y[tr]).predict(X[te])
    print(f"== held out by product ==  acc={np.mean(pred == y):.3f}  macro-F1={f1_score(y, pred, average='macro'):.3f}")
    print(classification_report(y, pred, zero_division=0))

    # 2. held out by generator
    for g in sorted(set(gen)):
        tr, te = gen != g, gen == g
        if tr.sum() < 20 or te.sum() < 10:
            continue
        p = model().fit(X[tr], y[tr]).predict(X[te])
        print(f"== train on others, test on {g} (n={te.sum()}) ==  acc={np.mean(p == y[te]):.3f}  "
              f"macro-F1={f1_score(y[te], p, average='macro'):.3f}")

    # 3. what it learned - the interpretable part of the story
    m = model().fit(X, y)
    vec, clf = m.steps[0][1], m.steps[1][1]
    names = np.array(vec.get_feature_names_out())
    print("\ntop features per class:")
    for i, c in enumerate(clf.classes_):
        top = names[np.argsort(clf.coef_[i])[-6:]][::-1]
        print(f"  {c:14s} {', '.join(top)}")


if __name__ == "__main__":
    main()
