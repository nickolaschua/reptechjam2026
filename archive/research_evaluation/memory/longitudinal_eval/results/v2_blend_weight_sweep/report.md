# Frozen v2 staged mode-conditioned blend-weight sweep

## Experimental controls

- Fixed gate: `gate_cosine >= 0.2` (the immutable frozen production threshold).
- Buying and Browsing use independent weights. No common weight and no two-dimensional grid were evaluated.
- Sequence: Buying sweep → Buying lock → Browsing sweep → Browsing lock → final evaluation.
- Memory vectors, EWMA, embeddings, fixture, catalogue, eligibility masks, and retrieval behavior are frozen.
- Broad stability requires at least 3 adjacent strict-pass sweep points, ensuring a tested interior lock candidate.

## Stage 1 — Buying

| b | a | LP delta | MI delta | CO delta | Buying delta | Help | Harm | Unchanged | Rel pass/total | Irrel pass/total | Strict |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0.000 | 1.000 | +0.000000000 | +0.000000000 | +0.000000000 | +0.000000000 | 0.000 | 0.000 | 1.000 | 10/10 | 7/10 | False |
| 0.025 | 0.975 | +0.000310822 | -0.000184573 | +0.000485180 | +0.000203810 | 0.433 | 0.367 | 0.200 | 10/10 | 7/10 | False |
| 0.050 | 0.950 | +0.000315560 | +0.000139780 | +0.001755023 | +0.000736788 | 0.433 | 0.400 | 0.167 | 10/10 | 7/10 | True |
| 0.075 | 0.925 | +0.000597980 | -0.008189826 | -0.001649074 | -0.003080307 | 0.433 | 0.467 | 0.100 | 10/10 | 7/10 | False |
| 0.100 | 0.900 | +0.000851596 | -0.014329072 | +0.006960185 | -0.002172430 | 0.467 | 0.400 | 0.133 | 10/10 | 7/10 | False |
| 0.125 | 0.875 | +0.001097500 | -0.015088955 | +0.001694034 | -0.004099140 | 0.467 | 0.433 | 0.100 | 10/10 | 7/10 | False |
| 0.150 | 0.850 | +0.001414223 | -0.017364682 | +0.001604959 | -0.004781834 | 0.467 | 0.400 | 0.133 | 10/10 | 7/10 | False |
| 0.175 | 0.825 | +0.001719804 | -0.016924080 | +0.008233528 | -0.002323583 | 0.500 | 0.367 | 0.133 | 10/10 | 7/10 | False |
| 0.200 | 0.800 | +0.002071415 | -0.016673853 | +0.007446440 | -0.002385333 | 0.500 | 0.367 | 0.133 | 10/10 | 7/10 | False |
| 0.250 | 0.750 | +0.002787456 | -0.018315654 | +0.004467197 | -0.003687000 | 0.433 | 0.467 | 0.100 | 10/10 | 7/10 | False |
| 0.300 | 0.700 | +0.003565079 | -0.017627715 | +0.008376680 | -0.001895318 | 0.433 | 0.467 | 0.100 | 10/10 | 7/10 | False |
| 0.400 | 0.600 | +0.002703629 | -0.015269961 | +0.001845544 | -0.003573596 | 0.367 | 0.533 | 0.100 | 10/10 | 7/10 | False |
| 0.500 | 0.500 | +0.001918493 | -0.003748248 | -0.014473233 | -0.005434330 | 0.367 | 0.533 | 0.100 | 10/10 | 7/10 | False |

**Stage 1 outcome: STOP — NO_BROAD_STABLE_BUYING_REGION**

Strict-pass Buying weights were [0.05], but none formed at least 3 adjacent sweep points. The result is too narrow to lock without selecting an edge or an isolated numerical pocket.

Stages 2 and 3 were not run, as required by the staged protocol.
