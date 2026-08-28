# Experiment 4 — Constraint classification distribution

> **ORACLE DIAGNOSTIC.** This applies the evaluator's exact ordered classifier to reconstructed hidden constraints.

The classifier assigns 800 generated constraints across 6 emitted types. `other` can reveal all 800 constraints, while `category` and `brand` reveal none because the evaluator's classifier never emits those labels. There are 19 multi-rule strings resolved by precedence and 20 mismatches between the broader color regex used during card construction and the narrower ordered color classifier.
