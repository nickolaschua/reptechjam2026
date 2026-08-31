# Experiment 10 — XTR/WARP retrieval

> **AGENT-REALISTIC RANKING, ORACLE-AFTER-FREEZE DIAGNOSTICS.** The 600 XTR/WARP queries contained only category plus active disclosed dialogue evidence. Targets, scenarios, sample IDs, hidden cards, future turns, and profiles were absent from Colab and joined locally only after Top-1000 rankings were frozen.

The returned archive passed all member checksums and provenance checks. It contains a reusable converted 4-bit WARP index built from 50,000 products with pinned `google/xtr-base-en` and `nprobe=32` CPU retrieval. Exact-only and Experiment 7 BM25-RRF reproduced all **2,000 Top-10 turn slates** and all **200 session outcomes each** bit-for-bit.

On the frozen 140-session held-out set, exact-only scored **0.831404**, Experiment 7 BM25-RRF scored **0.846109**, and XTR/WARP-RRF scored **0.846892**. Relative to exact-only, BM25 rescued **4** hard failures with **2** regressions; XTR/WARP rescued **2** with **0** regressions. Relative to BM25, XTR/WARP rescued **3** failures and introduced **3** regressions.

The preregistered decision is to **recommend `exact_stateful_xtr_warp_rrf`**. XTR/WARP is promoted only when it strictly beats BM25 held-out TechnicalScore and passes the existing exact-baseline rescue/regression gates. No starter-agent file was modified.
