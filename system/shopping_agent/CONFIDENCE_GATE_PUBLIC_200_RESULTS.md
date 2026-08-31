# Confidence-gate public-set evaluation

Run date: 2026-08-31

This report accompanies [`confidence_gate_public_200.json`](confidence_gate_public_200.json), which contains the aggregate metrics, all 200 per-session outcomes for both arms, turn-level confidence decisions, target metadata, and aligned session comparisons.

## Method

The released 200-session public set was evaluated twice with the active shopping agent and the same route-specific ranking pipeline:

1. `pre_gate`: the confidence threshold was disabled, so every member of the fixed top-10 unseen pool survived.
2. `confidence_gate_0_40`: the fixed top-10 unseen pool retained only products with current-query similarity `s1 >= 0.40`. Lower-ranked products did not backfill rejected positions.

The harness used the configured OpenAI embedding backend and cached 50,000-row catalogue matrix. Response wording was deterministic so this run measures retrieval and dialogue routing without chat-generation variance.

## Results

| Metric | Pre-gate | Confidence gate 0.40 | Delta |
|---|---:|---:|---:|
| Sample count | 200 | 200 | 0 |
| Hit Rate@10 | 0.980000 | 0.980000 | 0.000000 |
| MRR | 0.556563 | 0.611563 | **+0.055000** |
| MTTC | 2.130000 | 2.455000 | +0.325000 |
| Efficiency | 0.887000 | 0.854500 | -0.032500 |
| Recommended technical score | 0.834369 | 0.844369 | **+0.010000** |

The active gate evaluated 4,870 products over 487 turns. It retained 3,669, rejected 1,201 (24.66%), and produced 35 empty-result turns (7.19%). Both arms hit 196 of 200 targets.

## Why MRR improved

MRR measures the reciprocal of the target's rank when a session first hits; it does not penalize a later hit. The confidence gate preserves upstream order but deletes low-`s1` products. If a target survives at original pool rank `r` and `k` rejected products were ahead of it, its returned rank becomes `r-k`, increasing its reciprocal-rank contribution.

Across the 200 sessions:

- 35 sessions improved rank, contributing `+11.825` total reciprocal rank.
- 4 sessions worsened rank, contributing `-0.825`.
- 161 sessions were unchanged.
- Therefore the mean change was `(11.825 - 0.825) / 200 = +0.055`.
- Of the 35 rank improvements, 13 hit on the same turn and 22 hit later. None hit earlier.

Large examples include `public_0060` and `public_0191`, both moving from rank 8 to rank 1 (`+0.875` each), and `public_0151`, moving from rank 7 to rank 1 (`+0.857143`). This explains the apparently contradictory metrics: confidence filtering improved the rank of eventual hits while extra clarification increased MTTC by 0.325 turns. In the technical score, the weighted MRR gain contributed `+0.0165`, the efficiency loss contributed `-0.0065`, and the net change was `+0.0100`.

## The four missed sessions

The same four sessions missed in both arms. The confidence gate did not create these failures. In every turn of every miss, the target was outside the fixed upstream top-10 pool before confidence filtering, so it could neither pass nor be returned. Because the gate intentionally has no backfill, rejecting weak top-10 rows also could not pull these targets into consideration.

### `public_0046` — intent override

Target: `B0B42PVX1F`, a women's wool plus-size thigh-high sock/leg-warmer product.

The dialogue began with the weak old preference `No Closure closure`, then overrode it with `wool` and disclosed the exact fiber composition. All ten turns used keyword-state ranking. The retrieved pages contained generic wool, crew, ankle, and knee-high socks, but the target never entered the evaluated top 10. Confidence was permissive here (10/10 survivors through turn 8 and 9/10 on turns 9–10), confirming an upstream crowded-category/ranking miss rather than a confidence rejection.

### `public_0053` — buying

Target: `B07TZK3GZK`, a rose-gold floral passport cover.

The generated intent combined `passport covers`, `leather`, and `color: black`, despite the title describing rose gold. Results drifted from passport wallets into generic black leather wallets and eventually belts. The target was outside the top 10 on all turns in both arms. The confidence gate progressively reduced the pool from 9 survivors to only 1 by turns 9–10, but no-backfill meant the target below rank 10 was never reconsidered. This session exposes contradictory target evidence plus insufficient category specificity upstream.

### `public_0078` — intent override

Target: `B0C5RLJDSF`, Hanes women's crew socks.

The evaluator introduced it as `Socks No Show & Liner Socks`, while the product title says crew socks. The dialogue later overrode `Pull On closure` with `cotton` and disclosed fiber and origin details. Keyword ranking consequently favored no-show socks and many generic cotton socks. The target remained outside the top 10 for all ten turns; confidence retained between 4 and 10 rows, so it was never directly rejected. This is primarily a catalogue-category/title conflict amplified by a dense commodity category.

### `public_0139` — browsing

Target: `B09SGYPW3M`, an OFEEFAN women's ruffle short-sleeve V-neck top.

The user eventually disclosed polyester and a highly relevant ruffle-sleeve phrase. The target still never entered the top 10 because the category contains many near-duplicate women's T-shirts and ruffle tops; by turn 10, a different ruffle short-sleeve V-neck product occupied rank 1. Confidence retained 8–10 rows on every turn, so this was another upstream keyword-ranking collision, not a confidence failure.

## Conclusion

The confidence gate improved ranking quality among products already present in the fixed top-10 pool, but it cannot repair upstream recall. The four residual failures all require better candidate ranking/category evidence or a deliberate backfill design change; lowering the confidence threshold would not surface them because none reached the evaluated pool.
