# Experiment 9 — Adaptive hybrid architecture

> **AGENT-REALISTIC RANKING.** Every retrieval and clarification decision used only typed observable dialogue state and candidate metadata. Target and scenario fields were joined after each ranking was frozen.

Typed structured state reproduced all **2000 Experiment 7 turn rankings** bit-for-bit. Calibration selected clarification branch **deterministic_reranker_fixed_other**, then selected cumulative configuration **structured_state_identity_fixed_other** using TechnicalScore, rescues, regressions, MRR, and deterministic method-name ordering.

On its single 140-session held-out evaluation, the selected configuration scored **0.846109** TechnicalScore and **0.647506** MRR, compared with **0.846109** and **0.647506** for Experiment 7. It rescued **0** hard failures and caused **0** regressions. The preregistered decision is to **retain the Experiment 7 agent**.

The submission agent was not modified automatically.
