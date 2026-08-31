# Experiment 8 — Intent-routed dense browsing

> **AGENT-REALISTIC RANKING.** The session-locked route and rankings were computed from the first message, category, and active observable constraints. Scenario labels and targets were joined only after rankings were frozen.

The unchanged submission agent first reproduced its frozen **0.840846** score and all **200 session outcomes** exactly. The same-parser research control then reproduced all frozen Experiment 7 turn slates. All **1100 buying-routed turns** were bit-for-bit identical in the routed treatment. Browsing used normalized float32 `sentence-transformers/all-MiniLM-L6-v2` embeddings and pure cosine Top-10; instrumentation recorded **zero exact and zero BM25 calls** on that route.

Intent routing accuracy was **100.0%**. On the frozen 140-session held-out set, the routed treatment scored **0.710690** TechnicalScore and **0.538013** MRR versus **0.846109** and **0.647506** for Experiment 7. It rescued **0** control failures and introduced **21** regressions.

The paraphrase suite is a deterministic browsing stress test, not a replacement for official scoring. The submission agent was not modified.
