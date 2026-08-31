# Phase 6 longitudinal benchmark design

This is a research design, not a result report. Every session uses an existing
public buying row and real catalogue ASIN. Constant profiles omit tested facts.
The shopper is stateless across sessions; only earlier scheduled disclosures
are re-injected for consistency, and probe prompts forbid volunteering them.
The fixed price tendency is **under approximately $120**, used only with
numeric-price targets.

## U1 — stable learner

Constant persona: practical, selective, occasional shopper. Latent facts:
breathable/natural materials, neutral colours, and the $120 tendency.

| S | Public row | ASIN | Role / event |
|---|---|---|---|
| 1 | public_0018 | B07H3T5YGH | cold start; none |
| 2 | public_0199 | B089M57PSQ | establish P1 |
| 3 | public_0024 | B076X3JXMW | reinforce P1 in jumpsuits |
| 4 | public_0029 | B01IAKCZEK | establish P2 |
| 5 | public_0156 | B0C3KZXV4B | reinforce P2 in bags |
| 6 | public_0026 | B093R14VP1 | establish P3 |
| 7 | public_0163 | B0834T68X3 | combine P1/P3 |
| 8 | public_0108 | B01I21CI7G | combine P1/P2 |
| 9 | public_0088 | B07Z6J5N6Y | mature history; no dump |
| 10 | public_0178 | B01FWQ8NH8 | probe; no restatement |

B0 should be mechanism-identical with/without history. Future B1 may help but
over-aggregate; B2 should select material/colour/price memories; B3 will test
catalogue-local projected directions. No outcome is claimed.

## U2 — query override

Constant persona: decisive regular shopper. Latent facts: dark/neutral colours,
understated/minimal style, and the same price tendency.

| S | Public row | ASIN | Role / event |
|---|---|---|---|
| 1 | public_0149 | B07CBYYHTL | establish dark colour |
| 2 | public_0160 | B01AAANF2Y | reinforce dark colour |
| 3 | public_0028 | B0B9ZYDDZ1 | establish minimal style |
| 4 | public_0054 | B08PP1ZJQ5 | reinforce basic style |
| 5 | public_0042 | B01LWOGORL | establish price tendency |
| 6 | public_0028 | B0B9ZYDDZ1 | strong dark/minimal |
| 7 | public_0152 | B000EQU0NW | dark/classic, new category |
| 8 | public_0156 | B0C3KZXV4B | price-compatible reinforcement |
| 9 | public_0160 | B01AAANF2Y | strong recent bias |
| 10 | public_0145 | B00IJZZWGA | explicit bright/sporty override |

B0 ignores prior bias. Future B1 may expose harm. B2 should suppress P1/P2 and
retain portable P3. B3 tests the same claim with projection; none is active now.

## U3 — distractor history

Constant persona: occasional shopper with varied activity contexts. Latent
facts: breathable materials and the $120 tendency.

| S | Public row | ASIN | Role / event |
|---|---|---|---|
| 1 | public_0136 | B091F54MWM | useful P1 |
| 2 | public_0083 | B0BPMCJ1RD | one-off office collar/button-down |
| 3 | public_0095 | B09N78FT2W | one-off gym compression/pockets |
| 4 | public_0093 | B07PYB8F1G | useful P2 |
| 5 | public_0058 | B08L83YQTZ | one-off waterproof hood |
| 6 | public_0114 | B07H34Z5V6 | reinforce useful P1 |
| 7 | public_0066 | B0BFLFSB2Y | one formal dinner |
| 8 | public_0101 | B07QMS8TX8 | one trip's cargo storage |
| 9 | public_0005 | B074G1JP8Z | recent snow/waterproof distractor |
| 10 | public_0194 | B09WR1NZ48 | probe; no P1/P2 restatement |

The exact S10 runs at H0/H1/H3/H5/H9. B0 ignores each prefix. Future B1 may
degrade with noise; B2 should recover useful facts; B3 tests projected support.

## U4 — negative preference

Constant persona: careful regular shopper. Latent facts: no wool, no
polyester-heavy products, and no very bright/neon colours.

| S | Public row | ASIN | Role / event |
|---|---|---|---|
| 1 | public_0032 | B0834HZQZF | establish N1 |
| 2 | public_0088 | B07Z6J5N6Y | reinforce N1 |
| 3 | public_0132 | B08X2X83DW | neutral task |
| 4 | public_0199 | B089M57PSQ | establish N2 |
| 5 | public_0185 | B0BCW4QKV5 | reinforce N2 |
| 6 | public_0118 | B09M72C8PG | neutral task |
| 7 | public_0028 | B0B9ZYDDZ1 | establish N3 |
| 8 | public_0188 | B0B5ZS2J2W | reinforce N1 |
| 9 | public_0152 | B000EQU0NW | mature N2/N3 history |
| 10 | public_0178 | B01FWQ8NH8 | probe; no restatement |

B0 stores but cannot act on negatives. Existing QLMP B1/B2 helpers exclude
negative-polarity items, so the benchmark honestly exposes that limitation.
B3 negative handling remains a research question.

## Replay and interpretation

Every S10 has one fixture object and SHA-256 fingerprint. `NO_HISTORY` uses an
empty store; `FULL_HISTORY` imports only that user's S1–S9 snapshot. U3 prefixes
use arbitrary sequence-index filters, leaving room for selected-subset ablation.

Shopper generations may differ across replays. Logs retain provider/model,
every turn, recommendations and target ranks, leakage flags, disclosure chain,
and fingerprint. Phase 6 reports no memory lift or harm rate while
`historical_memory_applied` is false.
