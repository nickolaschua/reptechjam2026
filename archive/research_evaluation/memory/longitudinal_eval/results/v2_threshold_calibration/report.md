# Frozen v2 relevance-threshold calibration

**Required conclusion: GATE COSINE IS DISCRIMINATIVE BUT FIXED BLENDING REMAINS HARMFUL**

**Unimplemented follow-up experiment:** Frozen-vector blend-weight sweep.

## Frozen-input verification

All four v2 artifact hashes and the catalogue hash match the frozen manifest. Offline rank reconstruction passed. No embedding or LLM calls were made, and the source bundle was unchanged.

Thresholds: `[0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]`; comparator: `gate_cosine >= tau`; ranking: stable `(-score, ASIN)` over the persisted eligibility mask.

## Seven-slice M0/M3 baseline (tau 0.20)

| Threshold | Slice | n | M0 MRR | M3 MRR | Delta | M0 H@1/5/10 | M3 H@1/5/10 | M0 mean/median rank | M3 mean/median rank | Help/Harm/Unchanged |
|---:|---|---:|---:|---:|---:|---|---|---|---|---|
| 0.2 | OVERALL | 40 | 0.021452 | 0.019327 | -0.002125 | 0.000/0.050/0.050 | 0.000/0.025/0.050 | 482.5/251.0 | 505.2/246.5 | 0.475/0.425/0.100 |
| 0.2 | LONGITUDINAL_POSITIVE | 10 | 0.003615 | 0.005686 | +0.002071 | 0.000/0.000/0.000 | 0.000/0.000/0.000 | 619.1/434.5 | 514.1/280.5 | 0.800/0.200/0.000 |
| 0.2 | MEMORY_IRRELEVANT | 10 | 0.032736 | 0.016062 | -0.016674 | 0.000/0.100/0.100 | 0.000/0.000/0.000 | 509.4/189.5 | 544.9/177.5 | 0.300/0.400/0.300 |
| 0.2 | CURRENT_OVERRIDE | 10 | 0.045841 | 0.053287 | +0.007446 | 0.000/0.100/0.100 | 0.000/0.100/0.200 | 182.3/58.0 | 176.1/83.0 | 0.400/0.500/0.100 |
| 0.2 | BROWSING_PERSONALIZATION | 10 | 0.003615 | 0.002271 | -0.001343 | 0.000/0.000/0.000 | 0.000/0.000/0.000 | 619.1/434.5 | 785.7/609.0 | 0.400/0.600/0.000 |
| 0.2 | Buying | 30 | 0.027397 | 0.025012 | -0.002385 | 0.000/0.067/0.067 | 0.000/0.033/0.067 | 436.9/248.0 | 411.7/219.0 | 0.500/0.367/0.133 |
| 0.2 | Browsing | 10 | 0.003615 | 0.002271 | -0.001343 | 0.000/0.000/0.000 | 0.000/0.000/0.000 | 619.1/434.5 | 785.7/609.0 | 0.400/0.600/0.000 |

Help/harm/unchanged above use each probe's relevant-set reciprocal-rank delta. Hit rates and penalized-rank summaries are arm-specific.

## Gate distributions

Quantiles use linear interpolation; standard deviations are sample standard deviations (`ddof=1`).

| Scenario class | n | min | p10 | p25 | median | p75 | p90 | max | mean | sample sd |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| LONGITUDINAL_POSITIVE | 10 | 0.253915 | 0.263070 | 0.293239 | 0.310738 | 0.346569 | 0.396054 | 0.437723 | 0.324941 | 0.056648 |
| MEMORY_IRRELEVANT | 10 | 0.156657 | 0.175017 | 0.194408 | 0.218595 | 0.235646 | 0.263100 | 0.287179 | 0.218645 | 0.038782 |
| CURRENT_OVERRIDE | 10 | 0.265765 | 0.275129 | 0.284814 | 0.324694 | 0.337449 | 0.346160 | 0.367530 | 0.314931 | 0.034079 |
| BROWSING_PERSONALIZATION | 10 | 0.253915 | 0.263070 | 0.293239 | 0.310738 | 0.346569 | 0.396054 | 0.437723 | 0.324941 | 0.056648 |

### All probes by descending exact cosine

| # | Session | Scenario | Relation | Mode | Cosine | tau-0.20 pass |
|---:|---|---|---|---|---:|---|
| 1 | bp_02_s2 | BROWSING_PERSONALIZATION | RELEVANT | Browsing | 0.437722981 | True |
| 2 | lp_07_s2 | LONGITUDINAL_POSITIVE | RELEVANT | Buying | 0.437722981 | True |
| 3 | bp_01_s2 | BROWSING_PERSONALIZATION | RELEVANT | Browsing | 0.391424328 | True |
| 4 | lp_08_s2 | LONGITUDINAL_POSITIVE | RELEVANT | Buying | 0.391424328 | True |
| 5 | co_08_s2 | CURRENT_OVERRIDE | CONFLICTING | Buying | 0.367530257 | True |
| 6 | bp_06_s2 | BROWSING_PERSONALIZATION | RELEVANT | Browsing | 0.348458290 | True |
| 7 | lp_03_s2 | LONGITUDINAL_POSITIVE | RELEVANT | Buying | 0.348458290 | True |
| 8 | co_01_s2 | CURRENT_OVERRIDE | CONFLICTING | Buying | 0.343785226 | True |
| 9 | bp_04_s2 | BROWSING_PERSONALIZATION | RELEVANT | Browsing | 0.340903044 | True |
| 10 | lp_05_s2 | LONGITUDINAL_POSITIVE | RELEVANT | Buying | 0.340903044 | True |
| 11 | co_04_s2 | CURRENT_OVERRIDE | CONFLICTING | Buying | 0.338083744 | True |
| 12 | co_07_s2 | CURRENT_OVERRIDE | CONFLICTING | Buying | 0.335543215 | True |
| 13 | co_09_s2 | CURRENT_OVERRIDE | CONFLICTING | Buying | 0.333190680 | True |
| 14 | co_02_s2 | CURRENT_OVERRIDE | CONFLICTING | Buying | 0.316197604 | True |
| 15 | bp_03_s2 | BROWSING_PERSONALIZATION | RELEVANT | Browsing | 0.314230055 | True |
| 16 | lp_06_s2 | LONGITUDINAL_POSITIVE | RELEVANT | Buying | 0.314230055 | True |
| 17 | bp_07_s2 | BROWSING_PERSONALIZATION | RELEVANT | Browsing | 0.307246327 | True |
| 18 | lp_02_s2 | LONGITUDINAL_POSITIVE | RELEVANT | Buying | 0.307246327 | True |
| 19 | bp_08_s2 | BROWSING_PERSONALIZATION | RELEVANT | Browsing | 0.300662309 | True |
| 20 | lp_01_s2 | LONGITUDINAL_POSITIVE | RELEVANT | Buying | 0.300662309 | True |
| 21 | bp_00_s2 | BROWSING_PERSONALIZATION | RELEVANT | Browsing | 0.290764511 | True |
| 22 | lp_09_s2 | LONGITUDINAL_POSITIVE | RELEVANT | Buying | 0.290764511 | True |
| 23 | co_06_s2 | CURRENT_OVERRIDE | CONFLICTING | Buying | 0.289941192 | True |
| 24 | mi_04_s2 | MEMORY_IRRELEVANT | IRRELEVANT | Buying | 0.287179112 | True |
| 25 | co_00_s2 | CURRENT_OVERRIDE | CONFLICTING | Buying | 0.283105552 | True |
| 26 | co_03_s2 | CURRENT_OVERRIDE | CONFLICTING | Buying | 0.276169896 | True |
| 27 | co_05_s2 | CURRENT_OVERRIDE | CONFLICTING | Buying | 0.265765250 | True |
| 28 | bp_09_s2 | BROWSING_PERSONALIZATION | RELEVANT | Browsing | 0.264087558 | True |
| 29 | lp_00_s2 | LONGITUDINAL_POSITIVE | RELEVANT | Buying | 0.264087558 | True |
| 30 | mi_08_s2 | MEMORY_IRRELEVANT | IRRELEVANT | Buying | 0.260424376 | True |
| 31 | bp_05_s2 | BROWSING_PERSONALIZATION | RELEVANT | Browsing | 0.253914893 | True |
| 32 | lp_04_s2 | LONGITUDINAL_POSITIVE | RELEVANT | Buying | 0.253914893 | True |
| 33 | mi_05_s2 | MEMORY_IRRELEVANT | IRRELEVANT | Buying | 0.237142235 | True |
| 34 | mi_09_s2 | MEMORY_IRRELEVANT | IRRELEVANT | Buying | 0.231156975 | True |
| 35 | mi_03_s2 | MEMORY_IRRELEVANT | IRRELEVANT | Buying | 0.224919453 | True |
| 36 | mi_07_s2 | MEMORY_IRRELEVANT | IRRELEVANT | Buying | 0.212271273 | True |
| 37 | mi_06_s2 | MEMORY_IRRELEVANT | IRRELEVANT | Buying | 0.210650682 | True |
| 38 | mi_02_s2 | MEMORY_IRRELEVANT | IRRELEVANT | Buying | 0.188993260 | False |
| 39 | mi_00_s2 | MEMORY_IRRELEVANT | IRRELEVANT | Buying | 0.177056745 | False |
| 40 | mi_01_s2 | MEMORY_IRRELEVANT | IRRELEVANT | Buying | 0.156657159 | False |

## Classification diagnostics

RELEVANT-vs-IRRELEVANT ROC AUC: **0.970000**. PR AUC (average precision): **0.985283**. CONFLICTING probes are excluded.

At tau 0.20: TP=20, FP=7, TN=3, FN=0, TPR=1.000, FPR=0.700.

Highest-threshold Youden-J tie winner: tau=0.290764511, J=0.800, TPR=0.800, FPR=0.000.

All exact-score diagnostic cutoffs are in `classification_cutoffs.csv` and `results.json`.

## Threshold sweep

| Threshold | Slice | n | M0 MRR | M3 MRR | Delta | M0 H@1/5/10 | M3 H@1/5/10 | M0 mean/median rank | M3 mean/median rank | Help/Harm/Unchanged |
|---:|---|---:|---:|---:|---:|---|---|---|---|---|
| 0.0 | OVERALL | 40 | 0.021452 | 0.019322 | -0.002129 | 0.000/0.050/0.050 | 0.000/0.025/0.050 | 482.5/251.0 | 511.2/250.0 | 0.475/0.475/0.050 |
| 0.0 | LONGITUDINAL_POSITIVE | 10 | 0.003615 | 0.005686 | +0.002071 | 0.000/0.000/0.000 | 0.000/0.000/0.000 | 619.1/434.5 | 514.1/280.5 | 0.800/0.200/0.000 |
| 0.0 | MEMORY_IRRELEVANT | 10 | 0.032736 | 0.016045 | -0.016691 | 0.000/0.100/0.100 | 0.000/0.000/0.000 | 509.4/189.5 | 568.8/181.0 | 0.300/0.600/0.100 |
| 0.0 | CURRENT_OVERRIDE | 10 | 0.045841 | 0.053287 | +0.007446 | 0.000/0.100/0.100 | 0.000/0.100/0.200 | 182.3/58.0 | 176.1/83.0 | 0.400/0.500/0.100 |
| 0.0 | BROWSING_PERSONALIZATION | 10 | 0.003615 | 0.002271 | -0.001343 | 0.000/0.000/0.000 | 0.000/0.000/0.000 | 619.1/434.5 | 785.7/609.0 | 0.400/0.600/0.000 |
| 0.0 | Buying | 30 | 0.027397 | 0.025006 | -0.002391 | 0.000/0.067/0.067 | 0.000/0.033/0.067 | 436.9/248.0 | 419.7/219.0 | 0.500/0.433/0.067 |
| 0.0 | Browsing | 10 | 0.003615 | 0.002271 | -0.001343 | 0.000/0.000/0.000 | 0.000/0.000/0.000 | 619.1/434.5 | 785.7/609.0 | 0.400/0.600/0.000 |
| 0.05 | OVERALL | 40 | 0.021452 | 0.019322 | -0.002129 | 0.000/0.050/0.050 | 0.000/0.025/0.050 | 482.5/251.0 | 511.2/250.0 | 0.475/0.475/0.050 |
| 0.05 | LONGITUDINAL_POSITIVE | 10 | 0.003615 | 0.005686 | +0.002071 | 0.000/0.000/0.000 | 0.000/0.000/0.000 | 619.1/434.5 | 514.1/280.5 | 0.800/0.200/0.000 |
| 0.05 | MEMORY_IRRELEVANT | 10 | 0.032736 | 0.016045 | -0.016691 | 0.000/0.100/0.100 | 0.000/0.000/0.000 | 509.4/189.5 | 568.8/181.0 | 0.300/0.600/0.100 |
| 0.05 | CURRENT_OVERRIDE | 10 | 0.045841 | 0.053287 | +0.007446 | 0.000/0.100/0.100 | 0.000/0.100/0.200 | 182.3/58.0 | 176.1/83.0 | 0.400/0.500/0.100 |
| 0.05 | BROWSING_PERSONALIZATION | 10 | 0.003615 | 0.002271 | -0.001343 | 0.000/0.000/0.000 | 0.000/0.000/0.000 | 619.1/434.5 | 785.7/609.0 | 0.400/0.600/0.000 |
| 0.05 | Buying | 30 | 0.027397 | 0.025006 | -0.002391 | 0.000/0.067/0.067 | 0.000/0.033/0.067 | 436.9/248.0 | 419.7/219.0 | 0.500/0.433/0.067 |
| 0.05 | Browsing | 10 | 0.003615 | 0.002271 | -0.001343 | 0.000/0.000/0.000 | 0.000/0.000/0.000 | 619.1/434.5 | 785.7/609.0 | 0.400/0.600/0.000 |
| 0.1 | OVERALL | 40 | 0.021452 | 0.019322 | -0.002129 | 0.000/0.050/0.050 | 0.000/0.025/0.050 | 482.5/251.0 | 511.2/250.0 | 0.475/0.475/0.050 |
| 0.1 | LONGITUDINAL_POSITIVE | 10 | 0.003615 | 0.005686 | +0.002071 | 0.000/0.000/0.000 | 0.000/0.000/0.000 | 619.1/434.5 | 514.1/280.5 | 0.800/0.200/0.000 |
| 0.1 | MEMORY_IRRELEVANT | 10 | 0.032736 | 0.016045 | -0.016691 | 0.000/0.100/0.100 | 0.000/0.000/0.000 | 509.4/189.5 | 568.8/181.0 | 0.300/0.600/0.100 |
| 0.1 | CURRENT_OVERRIDE | 10 | 0.045841 | 0.053287 | +0.007446 | 0.000/0.100/0.100 | 0.000/0.100/0.200 | 182.3/58.0 | 176.1/83.0 | 0.400/0.500/0.100 |
| 0.1 | BROWSING_PERSONALIZATION | 10 | 0.003615 | 0.002271 | -0.001343 | 0.000/0.000/0.000 | 0.000/0.000/0.000 | 619.1/434.5 | 785.7/609.0 | 0.400/0.600/0.000 |
| 0.1 | Buying | 30 | 0.027397 | 0.025006 | -0.002391 | 0.000/0.067/0.067 | 0.000/0.033/0.067 | 436.9/248.0 | 419.7/219.0 | 0.500/0.433/0.067 |
| 0.1 | Browsing | 10 | 0.003615 | 0.002271 | -0.001343 | 0.000/0.000/0.000 | 0.000/0.000/0.000 | 619.1/434.5 | 785.7/609.0 | 0.400/0.600/0.000 |
| 0.15 | OVERALL | 40 | 0.021452 | 0.019322 | -0.002129 | 0.000/0.050/0.050 | 0.000/0.025/0.050 | 482.5/251.0 | 511.2/250.0 | 0.475/0.475/0.050 |
| 0.15 | LONGITUDINAL_POSITIVE | 10 | 0.003615 | 0.005686 | +0.002071 | 0.000/0.000/0.000 | 0.000/0.000/0.000 | 619.1/434.5 | 514.1/280.5 | 0.800/0.200/0.000 |
| 0.15 | MEMORY_IRRELEVANT | 10 | 0.032736 | 0.016045 | -0.016691 | 0.000/0.100/0.100 | 0.000/0.000/0.000 | 509.4/189.5 | 568.8/181.0 | 0.300/0.600/0.100 |
| 0.15 | CURRENT_OVERRIDE | 10 | 0.045841 | 0.053287 | +0.007446 | 0.000/0.100/0.100 | 0.000/0.100/0.200 | 182.3/58.0 | 176.1/83.0 | 0.400/0.500/0.100 |
| 0.15 | BROWSING_PERSONALIZATION | 10 | 0.003615 | 0.002271 | -0.001343 | 0.000/0.000/0.000 | 0.000/0.000/0.000 | 619.1/434.5 | 785.7/609.0 | 0.400/0.600/0.000 |
| 0.15 | Buying | 30 | 0.027397 | 0.025006 | -0.002391 | 0.000/0.067/0.067 | 0.000/0.033/0.067 | 436.9/248.0 | 419.7/219.0 | 0.500/0.433/0.067 |
| 0.15 | Browsing | 10 | 0.003615 | 0.002271 | -0.001343 | 0.000/0.000/0.000 | 0.000/0.000/0.000 | 619.1/434.5 | 785.7/609.0 | 0.400/0.600/0.000 |
| 0.2 | OVERALL | 40 | 0.021452 | 0.019327 | -0.002125 | 0.000/0.050/0.050 | 0.000/0.025/0.050 | 482.5/251.0 | 505.2/246.5 | 0.475/0.425/0.100 |
| 0.2 | LONGITUDINAL_POSITIVE | 10 | 0.003615 | 0.005686 | +0.002071 | 0.000/0.000/0.000 | 0.000/0.000/0.000 | 619.1/434.5 | 514.1/280.5 | 0.800/0.200/0.000 |
| 0.2 | MEMORY_IRRELEVANT | 10 | 0.032736 | 0.016062 | -0.016674 | 0.000/0.100/0.100 | 0.000/0.000/0.000 | 509.4/189.5 | 544.9/177.5 | 0.300/0.400/0.300 |
| 0.2 | CURRENT_OVERRIDE | 10 | 0.045841 | 0.053287 | +0.007446 | 0.000/0.100/0.100 | 0.000/0.100/0.200 | 182.3/58.0 | 176.1/83.0 | 0.400/0.500/0.100 |
| 0.2 | BROWSING_PERSONALIZATION | 10 | 0.003615 | 0.002271 | -0.001343 | 0.000/0.000/0.000 | 0.000/0.000/0.000 | 619.1/434.5 | 785.7/609.0 | 0.400/0.600/0.000 |
| 0.2 | Buying | 30 | 0.027397 | 0.025012 | -0.002385 | 0.000/0.067/0.067 | 0.000/0.033/0.067 | 436.9/248.0 | 411.7/219.0 | 0.500/0.367/0.133 |
| 0.2 | Browsing | 10 | 0.003615 | 0.002271 | -0.001343 | 0.000/0.000/0.000 | 0.000/0.000/0.000 | 619.1/434.5 | 785.7/609.0 | 0.400/0.600/0.000 |
| 0.25 | OVERALL | 40 | 0.021452 | 0.018679 | -0.002772 | 0.000/0.050/0.050 | 0.000/0.025/0.050 | 482.5/251.0 | 496.1/246.5 | 0.425/0.350/0.225 |
| 0.25 | LONGITUDINAL_POSITIVE | 10 | 0.003615 | 0.005686 | +0.002071 | 0.000/0.000/0.000 | 0.000/0.000/0.000 | 619.1/434.5 | 514.1/280.5 | 0.800/0.200/0.000 |
| 0.25 | MEMORY_IRRELEVANT | 10 | 0.032736 | 0.013472 | -0.019264 | 0.000/0.100/0.100 | 0.000/0.000/0.000 | 509.4/189.5 | 508.4/177.5 | 0.100/0.100/0.800 |
| 0.25 | CURRENT_OVERRIDE | 10 | 0.045841 | 0.053287 | +0.007446 | 0.000/0.100/0.100 | 0.000/0.100/0.200 | 182.3/58.0 | 176.1/83.0 | 0.400/0.500/0.100 |
| 0.25 | BROWSING_PERSONALIZATION | 10 | 0.003615 | 0.002271 | -0.001343 | 0.000/0.000/0.000 | 0.000/0.000/0.000 | 619.1/434.5 | 785.7/609.0 | 0.400/0.600/0.000 |
| 0.25 | Buying | 30 | 0.027397 | 0.024148 | -0.003249 | 0.000/0.067/0.067 | 0.000/0.033/0.067 | 436.9/248.0 | 399.5/219.0 | 0.433/0.267/0.300 |
| 0.25 | Browsing | 10 | 0.003615 | 0.002271 | -0.001343 | 0.000/0.000/0.000 | 0.000/0.000/0.000 | 619.1/434.5 | 785.7/609.0 | 0.400/0.600/0.000 |
| 0.3 | OVERALL | 40 | 0.021452 | 0.023703 | +0.002251 | 0.000/0.050/0.050 | 0.000/0.050/0.075 | 482.5/251.0 | 520.3/248.0 | 0.250/0.225/0.525 |
| 0.3 | LONGITUDINAL_POSITIVE | 10 | 0.003615 | 0.005438 | +0.001823 | 0.000/0.000/0.000 | 0.000/0.000/0.000 | 619.1/434.5 | 570.8/347.5 | 0.500/0.200/0.300 |
| 0.3 | MEMORY_IRRELEVANT | 10 | 0.032736 | 0.032736 | +0.000000 | 0.000/0.100/0.100 | 0.000/0.100/0.100 | 509.4/189.5 | 509.4/189.5 | 0.000/0.000/1.000 |
| 0.3 | CURRENT_OVERRIDE | 10 | 0.045841 | 0.054273 | +0.008432 | 0.000/0.100/0.100 | 0.000/0.100/0.200 | 182.3/58.0 | 192.1/58.0 | 0.300/0.200/0.500 |
| 0.3 | BROWSING_PERSONALIZATION | 10 | 0.003615 | 0.002364 | -0.001251 | 0.000/0.000/0.000 | 0.000/0.000/0.000 | 619.1/434.5 | 808.9/669.5 | 0.200/0.500/0.300 |
| 0.3 | Buying | 30 | 0.027397 | 0.030816 | +0.003418 | 0.000/0.067/0.067 | 0.000/0.067/0.100 | 436.9/248.0 | 424.1/225.0 | 0.267/0.133/0.600 |
| 0.3 | Browsing | 10 | 0.003615 | 0.002364 | -0.001251 | 0.000/0.000/0.000 | 0.000/0.000/0.000 | 619.1/434.5 | 808.9/669.5 | 0.200/0.500/0.300 |
| 0.35 | OVERALL | 40 | 0.021452 | 0.021765 | +0.000313 | 0.000/0.050/0.050 | 0.000/0.050/0.050 | 482.5/251.0 | 500.7/251.0 | 0.050/0.075/0.875 |
| 0.35 | LONGITUDINAL_POSITIVE | 10 | 0.003615 | 0.005152 | +0.001538 | 0.000/0.000/0.000 | 0.000/0.000/0.000 | 619.1/434.5 | 622.8/469.0 | 0.100/0.100/0.800 |
| 0.35 | MEMORY_IRRELEVANT | 10 | 0.032736 | 0.032736 | +0.000000 | 0.000/0.100/0.100 | 0.000/0.100/0.100 | 509.4/189.5 | 509.4/189.5 | 0.000/0.000/1.000 |
| 0.35 | CURRENT_OVERRIDE | 10 | 0.045841 | 0.046734 | +0.000893 | 0.000/0.100/0.100 | 0.000/0.100/0.100 | 182.3/58.0 | 182.1/58.0 | 0.100/0.000/0.900 |
| 0.35 | BROWSING_PERSONALIZATION | 10 | 0.003615 | 0.002437 | -0.001177 | 0.000/0.000/0.000 | 0.000/0.000/0.000 | 619.1/434.5 | 688.4/505.5 | 0.000/0.200/0.800 |
| 0.35 | Buying | 30 | 0.027397 | 0.028207 | +0.000810 | 0.000/0.067/0.067 | 0.000/0.067/0.067 | 436.9/248.0 | 438.1/248.0 | 0.067/0.033/0.900 |
| 0.35 | Browsing | 10 | 0.003615 | 0.002437 | -0.001177 | 0.000/0.000/0.000 | 0.000/0.000/0.000 | 619.1/434.5 | 688.4/505.5 | 0.000/0.200/0.800 |
| 0.4 | OVERALL | 40 | 0.021452 | 0.021418 | -0.000033 | 0.000/0.050/0.050 | 0.000/0.050/0.050 | 482.5/251.0 | 498.1/251.0 | 0.000/0.050/0.950 |
| 0.4 | LONGITUDINAL_POSITIVE | 10 | 0.003615 | 0.003590 | -0.000025 | 0.000/0.000/0.000 | 0.000/0.000/0.000 | 619.1/434.5 | 626.0/469.0 | 0.000/0.100/0.900 |
| 0.4 | MEMORY_IRRELEVANT | 10 | 0.032736 | 0.032736 | +0.000000 | 0.000/0.100/0.100 | 0.000/0.100/0.100 | 509.4/189.5 | 509.4/189.5 | 0.000/0.000/1.000 |
| 0.4 | CURRENT_OVERRIDE | 10 | 0.045841 | 0.045841 | +0.000000 | 0.000/0.100/0.100 | 0.000/0.100/0.100 | 182.3/58.0 | 182.3/58.0 | 0.000/0.000/1.000 |
| 0.4 | BROWSING_PERSONALIZATION | 10 | 0.003615 | 0.003507 | -0.000107 | 0.000/0.000/0.000 | 0.000/0.000/0.000 | 619.1/434.5 | 674.5/505.5 | 0.000/0.100/0.900 |
| 0.4 | Buying | 30 | 0.027397 | 0.027389 | -0.000008 | 0.000/0.067/0.067 | 0.000/0.067/0.067 | 436.9/248.0 | 439.2/248.0 | 0.000/0.033/0.967 |
| 0.4 | Browsing | 10 | 0.003615 | 0.003507 | -0.000107 | 0.000/0.000/0.000 | 0.000/0.000/0.000 | 619.1/434.5 | 674.5/505.5 | 0.000/0.100/0.900 |
| 0.45 | OVERALL | 40 | 0.021452 | 0.021452 | +0.000000 | 0.000/0.050/0.050 | 0.000/0.050/0.050 | 482.5/251.0 | 482.5/251.0 | 0.000/0.000/1.000 |
| 0.45 | LONGITUDINAL_POSITIVE | 10 | 0.003615 | 0.003615 | +0.000000 | 0.000/0.000/0.000 | 0.000/0.000/0.000 | 619.1/434.5 | 619.1/434.5 | 0.000/0.000/1.000 |
| 0.45 | MEMORY_IRRELEVANT | 10 | 0.032736 | 0.032736 | +0.000000 | 0.000/0.100/0.100 | 0.000/0.100/0.100 | 509.4/189.5 | 509.4/189.5 | 0.000/0.000/1.000 |
| 0.45 | CURRENT_OVERRIDE | 10 | 0.045841 | 0.045841 | +0.000000 | 0.000/0.100/0.100 | 0.000/0.100/0.100 | 182.3/58.0 | 182.3/58.0 | 0.000/0.000/1.000 |
| 0.45 | BROWSING_PERSONALIZATION | 10 | 0.003615 | 0.003615 | +0.000000 | 0.000/0.000/0.000 | 0.000/0.000/0.000 | 619.1/434.5 | 619.1/434.5 | 0.000/0.000/1.000 |
| 0.45 | Buying | 30 | 0.027397 | 0.027397 | +0.000000 | 0.000/0.067/0.067 | 0.000/0.067/0.067 | 436.9/248.0 | 436.9/248.0 | 0.000/0.000/1.000 |
| 0.45 | Browsing | 10 | 0.003615 | 0.003615 | +0.000000 | 0.000/0.000/0.000 | 0.000/0.000/0.000 | 619.1/434.5 | 619.1/434.5 | 0.000/0.000/1.000 |
| 0.5 | OVERALL | 40 | 0.021452 | 0.021452 | +0.000000 | 0.000/0.050/0.050 | 0.000/0.050/0.050 | 482.5/251.0 | 482.5/251.0 | 0.000/0.000/1.000 |
| 0.5 | LONGITUDINAL_POSITIVE | 10 | 0.003615 | 0.003615 | +0.000000 | 0.000/0.000/0.000 | 0.000/0.000/0.000 | 619.1/434.5 | 619.1/434.5 | 0.000/0.000/1.000 |
| 0.5 | MEMORY_IRRELEVANT | 10 | 0.032736 | 0.032736 | +0.000000 | 0.000/0.100/0.100 | 0.000/0.100/0.100 | 509.4/189.5 | 509.4/189.5 | 0.000/0.000/1.000 |
| 0.5 | CURRENT_OVERRIDE | 10 | 0.045841 | 0.045841 | +0.000000 | 0.000/0.100/0.100 | 0.000/0.100/0.100 | 182.3/58.0 | 182.3/58.0 | 0.000/0.000/1.000 |
| 0.5 | BROWSING_PERSONALIZATION | 10 | 0.003615 | 0.003615 | +0.000000 | 0.000/0.000/0.000 | 0.000/0.000/0.000 | 619.1/434.5 | 619.1/434.5 | 0.000/0.000/1.000 |
| 0.5 | Buying | 30 | 0.027397 | 0.027397 | +0.000000 | 0.000/0.067/0.067 | 0.000/0.067/0.067 | 436.9/248.0 | 436.9/248.0 | 0.000/0.000/1.000 |
| 0.5 | Browsing | 10 | 0.003615 | 0.003615 | +0.000000 | 0.000/0.000/0.000 | 0.000/0.000/0.000 | 619.1/434.5 | 619.1/434.5 | 0.000/0.000/1.000 |
| 0.55 | OVERALL | 40 | 0.021452 | 0.021452 | +0.000000 | 0.000/0.050/0.050 | 0.000/0.050/0.050 | 482.5/251.0 | 482.5/251.0 | 0.000/0.000/1.000 |
| 0.55 | LONGITUDINAL_POSITIVE | 10 | 0.003615 | 0.003615 | +0.000000 | 0.000/0.000/0.000 | 0.000/0.000/0.000 | 619.1/434.5 | 619.1/434.5 | 0.000/0.000/1.000 |
| 0.55 | MEMORY_IRRELEVANT | 10 | 0.032736 | 0.032736 | +0.000000 | 0.000/0.100/0.100 | 0.000/0.100/0.100 | 509.4/189.5 | 509.4/189.5 | 0.000/0.000/1.000 |
| 0.55 | CURRENT_OVERRIDE | 10 | 0.045841 | 0.045841 | +0.000000 | 0.000/0.100/0.100 | 0.000/0.100/0.100 | 182.3/58.0 | 182.3/58.0 | 0.000/0.000/1.000 |
| 0.55 | BROWSING_PERSONALIZATION | 10 | 0.003615 | 0.003615 | +0.000000 | 0.000/0.000/0.000 | 0.000/0.000/0.000 | 619.1/434.5 | 619.1/434.5 | 0.000/0.000/1.000 |
| 0.55 | Buying | 30 | 0.027397 | 0.027397 | +0.000000 | 0.000/0.067/0.067 | 0.000/0.067/0.067 | 436.9/248.0 | 436.9/248.0 | 0.000/0.000/1.000 |
| 0.55 | Browsing | 10 | 0.003615 | 0.003615 | +0.000000 | 0.000/0.000/0.000 | 0.000/0.000/0.000 | 619.1/434.5 | 619.1/434.5 | 0.000/0.000/1.000 |
| 0.6 | OVERALL | 40 | 0.021452 | 0.021452 | +0.000000 | 0.000/0.050/0.050 | 0.000/0.050/0.050 | 482.5/251.0 | 482.5/251.0 | 0.000/0.000/1.000 |
| 0.6 | LONGITUDINAL_POSITIVE | 10 | 0.003615 | 0.003615 | +0.000000 | 0.000/0.000/0.000 | 0.000/0.000/0.000 | 619.1/434.5 | 619.1/434.5 | 0.000/0.000/1.000 |
| 0.6 | MEMORY_IRRELEVANT | 10 | 0.032736 | 0.032736 | +0.000000 | 0.000/0.100/0.100 | 0.000/0.100/0.100 | 509.4/189.5 | 509.4/189.5 | 0.000/0.000/1.000 |
| 0.6 | CURRENT_OVERRIDE | 10 | 0.045841 | 0.045841 | +0.000000 | 0.000/0.100/0.100 | 0.000/0.100/0.100 | 182.3/58.0 | 182.3/58.0 | 0.000/0.000/1.000 |
| 0.6 | BROWSING_PERSONALIZATION | 10 | 0.003615 | 0.003615 | +0.000000 | 0.000/0.000/0.000 | 0.000/0.000/0.000 | 619.1/434.5 | 619.1/434.5 | 0.000/0.000/1.000 |
| 0.6 | Buying | 30 | 0.027397 | 0.027397 | +0.000000 | 0.000/0.067/0.067 | 0.000/0.067/0.067 | 436.9/248.0 | 436.9/248.0 | 0.000/0.000/1.000 |
| 0.6 | Browsing | 10 | 0.003615 | 0.003615 | +0.000000 | 0.000/0.000/0.000 | 0.000/0.000/0.000 | 619.1/434.5 | 619.1/434.5 | 0.000/0.000/1.000 |
| 0.65 | OVERALL | 40 | 0.021452 | 0.021452 | +0.000000 | 0.000/0.050/0.050 | 0.000/0.050/0.050 | 482.5/251.0 | 482.5/251.0 | 0.000/0.000/1.000 |
| 0.65 | LONGITUDINAL_POSITIVE | 10 | 0.003615 | 0.003615 | +0.000000 | 0.000/0.000/0.000 | 0.000/0.000/0.000 | 619.1/434.5 | 619.1/434.5 | 0.000/0.000/1.000 |
| 0.65 | MEMORY_IRRELEVANT | 10 | 0.032736 | 0.032736 | +0.000000 | 0.000/0.100/0.100 | 0.000/0.100/0.100 | 509.4/189.5 | 509.4/189.5 | 0.000/0.000/1.000 |
| 0.65 | CURRENT_OVERRIDE | 10 | 0.045841 | 0.045841 | +0.000000 | 0.000/0.100/0.100 | 0.000/0.100/0.100 | 182.3/58.0 | 182.3/58.0 | 0.000/0.000/1.000 |
| 0.65 | BROWSING_PERSONALIZATION | 10 | 0.003615 | 0.003615 | +0.000000 | 0.000/0.000/0.000 | 0.000/0.000/0.000 | 619.1/434.5 | 619.1/434.5 | 0.000/0.000/1.000 |
| 0.65 | Buying | 30 | 0.027397 | 0.027397 | +0.000000 | 0.000/0.067/0.067 | 0.000/0.067/0.067 | 436.9/248.0 | 436.9/248.0 | 0.000/0.000/1.000 |
| 0.65 | Browsing | 10 | 0.003615 | 0.003615 | +0.000000 | 0.000/0.000/0.000 | 0.000/0.000/0.000 | 619.1/434.5 | 619.1/434.5 | 0.000/0.000/1.000 |
| 0.7 | OVERALL | 40 | 0.021452 | 0.021452 | +0.000000 | 0.000/0.050/0.050 | 0.000/0.050/0.050 | 482.5/251.0 | 482.5/251.0 | 0.000/0.000/1.000 |
| 0.7 | LONGITUDINAL_POSITIVE | 10 | 0.003615 | 0.003615 | +0.000000 | 0.000/0.000/0.000 | 0.000/0.000/0.000 | 619.1/434.5 | 619.1/434.5 | 0.000/0.000/1.000 |
| 0.7 | MEMORY_IRRELEVANT | 10 | 0.032736 | 0.032736 | +0.000000 | 0.000/0.100/0.100 | 0.000/0.100/0.100 | 509.4/189.5 | 509.4/189.5 | 0.000/0.000/1.000 |
| 0.7 | CURRENT_OVERRIDE | 10 | 0.045841 | 0.045841 | +0.000000 | 0.000/0.100/0.100 | 0.000/0.100/0.100 | 182.3/58.0 | 182.3/58.0 | 0.000/0.000/1.000 |
| 0.7 | BROWSING_PERSONALIZATION | 10 | 0.003615 | 0.003615 | +0.000000 | 0.000/0.000/0.000 | 0.000/0.000/0.000 | 619.1/434.5 | 619.1/434.5 | 0.000/0.000/1.000 |
| 0.7 | Buying | 30 | 0.027397 | 0.027397 | +0.000000 | 0.000/0.067/0.067 | 0.000/0.067/0.067 | 436.9/248.0 | 436.9/248.0 | 0.000/0.000/1.000 |
| 0.7 | Browsing | 10 | 0.003615 | 0.003615 | +0.000000 | 0.000/0.000/0.000 | 0.000/0.000/0.000 | 619.1/434.5 | 619.1/434.5 | 0.000/0.000/1.000 |
| 0.75 | OVERALL | 40 | 0.021452 | 0.021452 | +0.000000 | 0.000/0.050/0.050 | 0.000/0.050/0.050 | 482.5/251.0 | 482.5/251.0 | 0.000/0.000/1.000 |
| 0.75 | LONGITUDINAL_POSITIVE | 10 | 0.003615 | 0.003615 | +0.000000 | 0.000/0.000/0.000 | 0.000/0.000/0.000 | 619.1/434.5 | 619.1/434.5 | 0.000/0.000/1.000 |
| 0.75 | MEMORY_IRRELEVANT | 10 | 0.032736 | 0.032736 | +0.000000 | 0.000/0.100/0.100 | 0.000/0.100/0.100 | 509.4/189.5 | 509.4/189.5 | 0.000/0.000/1.000 |
| 0.75 | CURRENT_OVERRIDE | 10 | 0.045841 | 0.045841 | +0.000000 | 0.000/0.100/0.100 | 0.000/0.100/0.100 | 182.3/58.0 | 182.3/58.0 | 0.000/0.000/1.000 |
| 0.75 | BROWSING_PERSONALIZATION | 10 | 0.003615 | 0.003615 | +0.000000 | 0.000/0.000/0.000 | 0.000/0.000/0.000 | 619.1/434.5 | 619.1/434.5 | 0.000/0.000/1.000 |
| 0.75 | Buying | 30 | 0.027397 | 0.027397 | +0.000000 | 0.000/0.067/0.067 | 0.000/0.067/0.067 | 436.9/248.0 | 436.9/248.0 | 0.000/0.000/1.000 |
| 0.75 | Browsing | 10 | 0.003615 | 0.003615 | +0.000000 | 0.000/0.000/0.000 | 0.000/0.000/0.000 | 619.1/434.5 | 619.1/434.5 | 0.000/0.000/1.000 |
| 0.8 | OVERALL | 40 | 0.021452 | 0.021452 | +0.000000 | 0.000/0.050/0.050 | 0.000/0.050/0.050 | 482.5/251.0 | 482.5/251.0 | 0.000/0.000/1.000 |
| 0.8 | LONGITUDINAL_POSITIVE | 10 | 0.003615 | 0.003615 | +0.000000 | 0.000/0.000/0.000 | 0.000/0.000/0.000 | 619.1/434.5 | 619.1/434.5 | 0.000/0.000/1.000 |
| 0.8 | MEMORY_IRRELEVANT | 10 | 0.032736 | 0.032736 | +0.000000 | 0.000/0.100/0.100 | 0.000/0.100/0.100 | 509.4/189.5 | 509.4/189.5 | 0.000/0.000/1.000 |
| 0.8 | CURRENT_OVERRIDE | 10 | 0.045841 | 0.045841 | +0.000000 | 0.000/0.100/0.100 | 0.000/0.100/0.100 | 182.3/58.0 | 182.3/58.0 | 0.000/0.000/1.000 |
| 0.8 | BROWSING_PERSONALIZATION | 10 | 0.003615 | 0.003615 | +0.000000 | 0.000/0.000/0.000 | 0.000/0.000/0.000 | 619.1/434.5 | 619.1/434.5 | 0.000/0.000/1.000 |
| 0.8 | Buying | 30 | 0.027397 | 0.027397 | +0.000000 | 0.000/0.067/0.067 | 0.000/0.067/0.067 | 436.9/248.0 | 436.9/248.0 | 0.000/0.000/1.000 |
| 0.8 | Browsing | 10 | 0.003615 | 0.003615 | +0.000000 | 0.000/0.000/0.000 | 0.000/0.000/0.000 | 619.1/434.5 | 619.1/434.5 | 0.000/0.000/1.000 |

### Activation and operating-region test

| tau | Relevant activation (n=20) | Irrelevant activation (n=10) | Conflicting activation (n=10) | LP delta | BP delta | MI delta | CO delta | Overall delta | Qualifies |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0.00 | 1.000 | 1.000 | 1.000 | +0.002071 | -0.001343 | -0.016691 | +0.007446 | -0.002129 | False |
| 0.05 | 1.000 | 1.000 | 1.000 | +0.002071 | -0.001343 | -0.016691 | +0.007446 | -0.002129 | False |
| 0.10 | 1.000 | 1.000 | 1.000 | +0.002071 | -0.001343 | -0.016691 | +0.007446 | -0.002129 | False |
| 0.15 | 1.000 | 1.000 | 1.000 | +0.002071 | -0.001343 | -0.016691 | +0.007446 | -0.002129 | False |
| 0.20 | 1.000 | 0.700 | 1.000 | +0.002071 | -0.001343 | -0.016674 | +0.007446 | -0.002125 | False |
| 0.25 | 1.000 | 0.200 | 1.000 | +0.002071 | -0.001343 | -0.019264 | +0.007446 | -0.002772 | False |
| 0.30 | 0.700 | 0.000 | 0.600 | +0.001823 | -0.001251 | +0.000000 | +0.008432 | +0.002251 | False |
| 0.35 | 0.200 | 0.000 | 0.100 | +0.001538 | -0.001177 | +0.000000 | +0.000893 | +0.000313 | False |
| 0.40 | 0.100 | 0.000 | 0.000 | -0.000025 | -0.000107 | +0.000000 | +0.000000 | -0.000033 | False |
| 0.45 | 0.000 | 0.000 | 0.000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 | False |
| 0.50 | 0.000 | 0.000 | 0.000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 | False |
| 0.55 | 0.000 | 0.000 | 0.000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 | False |
| 0.60 | 0.000 | 0.000 | 0.000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 | False |
| 0.65 | 0.000 | 0.000 | 0.000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 | False |
| 0.70 | 0.000 | 0.000 | 0.000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 | False |
| 0.75 | 0.000 | 0.000 | 0.000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 | False |
| 0.80 | 0.000 | 0.000 | 0.000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 | False |

Qualifying thresholds (reported without maximum-MRR selection): **none**.

## Tau-0.20 MEMORY_IRRELEVANT passers

### mi_03_s2 — class A

- Query: `["I'm looking for rain jackets."]`
- Gate cosine: `0.224919453`
- Relevant rank/RR: 31 (0.032258065) → 16 (0.062500000); delta +0.030241935
- Classification basis: Embedded update text contains no probe-domain or temporary/session-specific query content; the cosine and rank movement arise from cross-domain trait semantics and promoted catalogue products.
- Target: `B001CM8L5G` — Kidorable Red Ladybug Natural Rubber Rain Boots With A Pull On Heel Tab
- Relevant ASINs: `['B001CM8L5G', 'B004HZYFFA', 'B004IFAMXS', 'B0055N6S8K', 'B006SVLS0G']`
- Persisted overtakers: 0

Textual/vector lineage, complete target/relevant catalogue metadata, parsed query state, and every hash-joined overtaker with s1/s2/s3 are in `forensics.json` and `forensics.csv`.

### mi_04_s2 — class A

- Query: `["I'm looking for watches."]`
- Gate cosine: `0.287179112`
- Relevant rank/RR: 4 (0.250000000) → 18 (0.055555556); delta -0.194444444
- Classification basis: Embedded update text contains no probe-domain or temporary/session-specific query content; the cosine and rank movement arise from cross-domain trait semantics and promoted catalogue products.
- Target: `9999666671` — Artist Unknown Urban Ladies Silver Diamante Large Dial Red PU Leather Strap Watch Analog Japanese Quartz Extra Battery
- Relevant ASINs: `['9999666671', 'B00009EIVN', 'B0000V9NBG', 'B00016W998', 'B0001V67WI']`
- Persisted overtakers: 14

Textual/vector lineage, complete target/relevant catalogue metadata, parsed query state, and every hash-joined overtaker with s1/s2/s3 are in `forensics.json` and `forensics.csv`.

### mi_05_s2 — class A

- Query: `["I'm looking for earrings."]`
- Gate cosine: `0.237142235`
- Relevant rank/RR: 582 (0.001718213) → 520 (0.001923077); delta +0.000204864
- Classification basis: Embedded update text contains no probe-domain or temporary/session-specific query content; the cosine and rank movement arise from cross-domain trait semantics and promoted catalogue products.
- Target: `9479290707` — Dyexces Sweaters for Women Cable Knit Sweater Women V Neck Long Sleeve Pullover Short Sweater Dresses
- Relevant ASINs: `['9479290707', 'B0000APUDV', 'B0000ATC4O', 'B0000C4E4Y', 'B0000EVVP4']`
- Persisted overtakers: 35

Textual/vector lineage, complete target/relevant catalogue metadata, parsed query state, and every hash-joined overtaker with s1/s2/s3 are in `forensics.json` and `forensics.csv`.

### mi_06_s2 — class A

- Query: `["I'm looking for work boots."]`
- Gate cosine: `0.210650682`
- Relevant rank/RR: 78 (0.012820513) → 101 (0.009900990); delta -0.002919523
- Classification basis: Embedded update text contains no probe-domain or temporary/session-specific query content; the cosine and rank movement arise from cross-domain trait semantics and promoted catalogue products.
- Target: `B0002PUNXM` — Carolina Men's 8 Inch Sarge Hi Light Brown
- Relevant ASINs: `['B0002PUNXM', 'B000M0EMES', 'B000MF6M8M', 'B000TAIJ9A', 'B000Y06SXO']`
- Persisted overtakers: 27

Textual/vector lineage, complete target/relevant catalogue metadata, parsed query state, and every hash-joined overtaker with s1/s2/s3 are in `forensics.json` and `forensics.csv`.

### mi_07_s2 — class A

- Query: `["I'm looking for necklaces."]`
- Gate cosine: `0.212271273`
- Relevant rank/RR: 251 (0.003984064) → 413 (0.002421308); delta -0.001562756
- Classification basis: Embedded update text contains no probe-domain or temporary/session-specific query content; the cosine and rank movement arise from cross-domain trait semantics and promoted catalogue products.
- Target: `7750000348` — Vatican Holy Pray Rosary Roses Petal Odor Beads Silver Plated In Plastic Round Box
- Relevant ASINs: `['7750000348', 'B0000APUDV', 'B0000ATC4O', 'B0000EVVP4', 'B0007PQO3U']`
- Persisted overtakers: 162

Textual/vector lineage, complete target/relevant catalogue metadata, parsed query state, and every hash-joined overtaker with s1/s2/s3 are in `forensics.json` and `forensics.csv`.

### mi_08_s2 — class A

- Query: `["I'm looking for backpacks."]`
- Gate cosine: `0.260424376`
- Relevant rank/RR: 128 (0.007812500) → 104 (0.009615385); delta +0.001802885
- Classification basis: Embedded update text contains no probe-domain or temporary/session-specific query content; the cosine and rank movement arise from cross-domain trait semantics and promoted catalogue products.
- Target: `B000NSTV0O` — Samsonite Wander Verb Backpack-Black/Dark Orange
- Relevant ASINs: `['B000NSTV0O', 'B000NZKTFI', 'B000OZC6Z8', 'B000UCDGXG', 'B0010ZMXWM']`
- Persisted overtakers: 7

Textual/vector lineage, complete target/relevant catalogue metadata, parsed query state, and every hash-joined overtaker with s1/s2/s3 are in `forensics.json` and `forensics.csv`.

### mi_09_s2 — class A

- Query: `["I'm looking for sandals."]`
- Gate cosine: `0.231156975`
- Relevant rank/RR: 1920 (0.000520833) → 2177 (0.000459348); delta -0.000061486
- Classification basis: Embedded update text contains no probe-domain or temporary/session-specific query content; the cosine and rank movement arise from cross-domain trait semantics and promoted catalogue products.
- Target: `B0002M6JB0` — Merrell Men's Tideriser Luna Slide LTR Sandal
- Relevant ASINs: `['B0002M6JB0', 'B0007D4UD8', 'B0007DHELS', 'B0009F1A4G', 'B000BHNHYS']`
- Persisted overtakers: 311

Textual/vector lineage, complete target/relevant catalogue metadata, parsed query state, and every hash-joined overtaker with s1/s2/s3 are in `forensics.json` and `forensics.csv`.

## Interpretation

An exact diagnostic cutoff reaches at least 90% relevant recall and at most 10% irrelevant FPR, but the strict operating region is empty because activated memory still harms benefit/overall slices.

The conclusion follows the required precedence rule; no threshold was selected by maximizing MRR.
