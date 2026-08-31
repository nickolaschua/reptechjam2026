# Experiment 6 — Slate-width counterfactuals

> **AGENT-REALISTIC EVALUATION.** Policies use hybrid scores from information disclosed by each simulated turn. The target is used only for scoring.

Thresholds were selected on a fixed, scenario-stratified 60-session calibration split and evaluated once on 140 held-out sessions. The chosen adaptive policy scored **0.409286** versus **0.524651** for full Top-10. It gained 0 hits, lost 25, and delayed 10 conversions relative to Top-10.

The adaptive confidence combines a normalized RRF top score and the top-two margin; relative candidate inclusion and optional low-confidence abstention were tuned only on calibration sessions.
