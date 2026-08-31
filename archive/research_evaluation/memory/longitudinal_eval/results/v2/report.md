# Trustworthy Longitudinal Evaluation v2

## Outcome

- Mechanism: **HARMFUL**
- Evaluator: **EVALUATOR NOW TRUSTWORTHY**
- Frozen algorithm: alpha=0.30; gate=0.20; Buying=0.8/0.2; Browsing=0.2/0.8.
- No tuning, threshold recommendation, memory redesign, or retrieval change was performed.

## Fixture and determinism

The frozen fixture contains 40 isolated timelines: 10 LONGITUDINAL_POSITIVE, 10 MEMORY_IRRELEVANT, 10 CURRENT_OVERRIDE, and 10 BROWSING_PERSONALIZATION. Each has two setup sessions and one scored probe (80 setup records, 40 probes). Buying has 30 probes and Browsing exactly 10. All messages use the local parser grammar; response prose is stubbed identically. Both arms share canonical state, v1, catalogue order, product embeddings, price/negative masks, and eligibility. They differ only in v2, gate consequence, s3, and permissible rank outcomes.

## Relevant-set results

| Slice | n | M0 MRR | M3 MRR | RR delta | Help | Harm | Gate |
|---|---:|---:|---:|---:|---:|---:|---:|
| Overall | 40 | 0.021452 | 0.019327 | -0.002125 | 47.5% | 42.5% | 92.5% |
| LONGITUDINAL_POSITIVE | 10 | 0.003615 | 0.005686 | +0.002071 | 80.0% | 20.0% | 100.0% |
| MEMORY_IRRELEVANT | 10 | 0.032736 | 0.016062 | -0.016674 | 30.0% | 40.0% | 70.0% |
| CURRENT_OVERRIDE | 10 | 0.045841 | 0.053287 | +0.007446 | 40.0% | 50.0% | 100.0% |
| BROWSING_PERSONALIZATION | 10 | 0.003615 | 0.002271 | -0.001343 | 40.0% | 60.0% | 100.0% |
| Buying | 30 | 0.027397 | 0.025012 | -0.002385 | 50.0% | 36.7% | 90.0% |
| Browsing | 10 | 0.003615 | 0.002271 | -0.001343 | 40.0% | 60.0% | 100.0% |

Exact-target metrics are retained separately under `exact_target` in `metrics.json`; relevant-set metrics use the best-ranked member of each frozen 3–8 ASIN set. Nullable misses use `eligible_count + 1` only for penalized rank summaries.

## Gate diagnostics

| Relation | min | p10 | p25 | median | p75 | p90 | max | mean | sample SD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| RELEVANT | 0.253915 | 0.263070 | 0.290765 | 0.310738 | 0.348458 | 0.396054 | 0.437723 | 0.324941 | 0.055137 |
| IRRELEVANT | 0.156657 | 0.175017 | 0.194408 | 0.218595 | 0.235646 | 0.263100 | 0.287179 | 0.218645 | 0.038782 |
| CONFLICTING | 0.265765 | 0.275129 | 0.284814 | 0.324694 | 0.337449 | 0.346160 | 0.367530 | 0.314931 | 0.034079 |

RELEVANT-vs-IRRELEVANT rank AUC (ties count 0.5): **0.970000**. CONFLICTING is reported separately and is not included in the ROC/AUC diagnostic.

## Forensic examples

- **Helping — co_09_s2 (CURRENT_OVERRIDE):** relevant rank 21 → 8; RR delta +0.077381; gate=0.333191. M0 top: `B075MVQZLV` Soccer Socks,Fasoar Unisex Team Sports Football Long Tube Socks Pack of 2,6,10; M3 top: `B0BCKY5Q5M` Black White Cat Crew Novelty Socks Casual Funny Crazy Athletic Dress Socks Unisex.
- **Ignored/least changed — mi_00_s2 (MEMORY_IRRELEVANT):** relevant rank 73 → 73; RR delta +0.000000; gate=0.177057. M0 top: `B08JQDNMP4` FRACORA Women's Hiking Boots Lace up Ankle Boots Mid Outdoor Non Slip Backpacking Trekking Mountaineering Trail Shoes; M3 top: `B08JQDNMP4` FRACORA Women's Hiking Boots Lace up Ankle Boots Mid Outdoor Non Slip Backpacking Trekking Mountaineering Trail Shoes.
- **Harming — mi_04_s2 (MEMORY_IRRELEVANT):** relevant rank 4 → 18; RR delta -0.194444; gate=0.287179. M0 top: `B074QM7JZ5` Han Shi Watches, Women Fashion Casual Business Eiffel Tower Quartz Wristwatch Round Clock; M3 top: `B01LA7FTO8` Geneva Women Big Dial Leather Brand Bracelet Wrist Watch Wholesales 6 Pcs Fiiliip(Mixed Color).
- **Override safety — co_03_s2 (CURRENT_OVERRIDE):** relevant rank 55 → 91; RR delta -0.007193; gate=0.276170. M0 top: `B08GCF8QRT` Women's PU Leather Combat Boots Warm Fur Lined Black Ankle Booties Side Zipper White Boots Non-slip Winter Boots(White.US8); M3 top: `B08GCF8QRT` Women's PU Leather Combat Boots Warm Fur Lined Black Ankle Booties Side Zipper White Boots Non-slip Winter Boots(White.US8).

Every probe record contains full update lineage; canonical parsed state and disclosed slots; hard-mask counts; gate and weights; M0/M3 top results with ASIN/title/categories/s1/s2/s3/rank; raw and penalized target/relevant ranks; and every below-relevant M0 product that overtook the relevant set in M3.

## Verification

- fixture_validation: PASS
- paired_invariants: PASS
- artifact_hashes: PASS
- offline_reconstruction: PASS
- deterministic_rerun: PASS
- complete_test_suite: PASS

The bundle is self-contained: `vectors.npz` includes catalogue ASIN order, v1, pre-query v2, update vectors, post-update vectors, s1/s2/s3, and all masks. Array references record key, dtype, shape, L2 norm, and SHA-256. `manifest.json` fingerprints fixture, catalogue, embedding cache, sources, and artifacts.

## Bugs fixed and files

- Replaced obsolete item-like `visible_matches` serialization with truthful aggregate-vector descriptions and evaluator-owned update lineage.
- Restricted paired LLM tapes to parser calls and made response prose identical across arms.
- Added opt-in immutable forensic snapshots without exposing vectors in normal response/debug payloads.
- Added v2 fixture generation/validation, relevant-set metrics, gate/AUC diagnostics, atomic bundle writing, tamper detection, and offline reconstruction.
- Primary implementation: `agent.py`, `run_longitudinal_eval.py`, `run_longitudinal_eval_v2.py`, `longitudinal_eval/evaluator_v2.py`, `build_fixture_v2.py`, `users_40_v2.json`, and tests.

## Limitations

This is a diagnostic fixture, not an unbiased estimate of user utility. Relevant sets are catalogue-text heuristics frozen before M3, and repeated category/trait templates reduce ecological diversity. Aggregate vector memory has no fact-level deletion, so lineage is evaluator evidence rather than retrievable memory items. The pre-registered HARMFUL result is a diagnosis of this frozen configuration and fixture—not a recommendation to tune the threshold.
