# Experiment 7 — Residual failure analysis

> **AGENT-REALISTIC RANKING, ORACLE-AFTER-FREEZE DIAGNOSTICS.** Rankers received only category plus active disclosed dialogue evidence. Targets, scenarios, sample IDs, hidden cards, future turns, and user profiles were joined only after all rankings were frozen.

The exact baseline reproduced all 200 frozen Experiment 2 sessions exactly: Hit@10 **0.930**, MRR **0.623058**, MTTC **2.75**, Efficiency **0.825**, and TechnicalScore **0.816917**. It left **14 hard failures** and **34 weak successes** whose first hit ranked 6–10.

Calibration selected **exact_stateful_bm25_rrf**. On the frozen 140-session held-out set it scored **0.846109** versus **0.831404** for exact-only, rescued **4** hard failures, and introduced **2** regressions. The production decision is to **recommend `exact_stateful_bm25_rrf`** under the preregistered gates; held-out results were not used to switch to another cascade.

Failure-category and rescue-by-category counts are explicitly **non-exclusive**. The synthetic price representation is separated from title, features, details, description, categories, and store in every evidence-attribution record. User profiles appear only in the hard/weak audit artifacts and were never ranker inputs. No cascade was installed in the starter agent.
