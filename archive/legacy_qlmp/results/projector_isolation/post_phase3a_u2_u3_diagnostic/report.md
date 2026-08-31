# A. Current-state audit

Current branch/HEAD: `nickolas` at `956f0599accc2c40038632e94392770ff0466f03` (`updated agent, documentation, and UI`, 2026-08-30T15:36:51+08:00). The worktree was already dirty before this diagnostic; those pre-existing changes were preserved.

Authoritative input: `preregistered_full_k500_r16_definitive`, run-manifest SHA-256 `1504db2b0ee5b5d98ca1bda2a92e7ad80cc5210cb4da77ebc7f29d6fb274a151`. The fixture SHA-256 is `8b782aee8ed7e23504f5c3420388e6b2003c16918fa0108f28dfdd7c630aa2bf`; the vector snapshot SHA-256 is `4763a536c9a16c25a7aa99eb062aec95778299ab5d2ea6d92decb337423b975a`; the catalogue fingerprint is `da979b05a68af864cb0dcf9ee6a81c010c7e66a57978ad286c7a2e005fc69a67`; and the product-text fingerprint is `f219fbfbc2a99b87e638241c2206c421eb69760037f9aa3a2e8da87c36a01da0`.

All M0, QLMP Phase-1/2, and projector-harness hashes frozen in that manifest match the current files. The replay used `m0_full_catalogue`, K=500, rank=16, the 50,000-row `text-embedding-3-large` cache, exact persisted q vectors, and persisted memory vectors. Top-500 IDs/scores, all singular values, and every U2/U3 projected memory norm replay exactly. External calls: 0 LLM, 0 OpenAI.

# B. Files changed

- `nickolas/shopping_agent/longitudinal_eval/projector_failure_diagnostic.py`: offline diagnostic only.
- `nickolas/shopping_agent/tests/test_projector_failure_diagnostic.py`: deterministic diagnostic tests only.
- This run directory: `summary.json`, `attribute_coverage.csv`, `component_extremes.csv`, `memory_components.csv`, and this report.

No pre-existing modified file was edited by this task.

# C. Product embedding representation

The exact current builder is:

```python
title = p.get("title") or ""
cats = ", ".join(p.get("categories") or [])
feats = "; ".join((p.get("features") or [])[:3])
text = f"Product: {title}. Categories: {cats}. Features: {feats}.".strip()
```

Thus title, every category string, and only the first three feature strings are embedded. Price, details, description, store, ratings, and features after index 2 are excluded.

- DIRECTLY REPRESENTED: product type/category; material when present early (U3 material coverage 86.4%); casual (58.6%); general colour (43.4%); bright/colourful (12.8%).
- INDIRECTLY / WEAKLY REPRESENTED: dark/black (4.8%), sporty/athletic (3.0%), minimal/understated proxies (5.8%), breathability (4.6%), formal/dressy (4.2%), cargo/storage (6.6%), hooded (0.2%), and winter/insulation (0.2%). Much more evidence exists in later features/details/descriptions than reaches the embedded string.
- NOT REPRESENTED: structured price/budget. In the U3 local neighbourhood, compression and rain/waterproof are also absent from both embedded and raw audited text.

Budget is structurally unavailable to the projector as currently defined. The few embedded `$`/`price`/`premium` strings are uncontrolled lexical mentions, not the catalogue price field.

# D. U2 Top-500 composition

Query (no target leakage):

- Raw: `hey, i'm looking for a pair of sporty ankle socks for women, ideally made of cotton. usually i go for dark and simple styles, but today i want something bright and colorful, and i want to stay under about $120.`
- Effective q text: `clothing cotton bright colorful women`

| Attribute | Embedded text | Raw catalogue | Raw only | Distinct embedded forms |
|---|---:|---:|---:|---|
| Price (field/proxy) | 12/500 (2.4%) lexical proxies | 74/500 (14.8%) valid prices | 73 | 4 proxy forms |
| Colour | 217 (43.4%) | 334 (66.8%) | 117 | 19 |
| Dark/black | 24 (4.8%) | 85 (17.0%) | 61 | black, charcoal, dark, navy |
| Bright/colourful | 64 (12.8%) | 107 (21.4%) | 43 | 7 |
| Sporty/athletic | 15 (3.0%) | 72 (14.4%) | 57 | 5 |
| Minimal/understated | 29 (5.8%) | 120 (24.0%) | 91 | basic, classic, plain, simple |

Only 74 products have a valid price: min $4.41, p25 $12.59, median $16.99, p75 $26.37, max $139.68; 426 have missing price. Query-score/price Spearman correlation is 0.0247. The affordability/premium proxy is too sparse to encode price (2 affordable and 6 premium proxy hits; only one proxy-hit item has a known price).

The neighbourhood is broad clothing rather than socks: only 8 sock leaf-category products. The largest leaves are T-shirts 70, Casual dresses 62, Westlake 48, blouses/button-downs 41, dresses 38, tunics 35, and tanks/camis 22. Women is the second-level category for 398/500.

# E. U2 local geometry

`sigma_1..sigma_16`:

```text
12.6716, 4.0264, 3.1280, 2.5144, 2.4778, 2.3248, 2.1567, 2.0731,
2.0006, 1.9729, 1.9024, 1.8346, 1.7704, 1.7676, 1.6701, 1.6456
```

`sigma_1/sigma_16 = 7.7004`. Cumulative total tangent-matrix energy: C1 40.15%, C1-2 44.20%, C1-4 48.23%, C1-8 53.36%, C1-16 60.02%. C1 is a large generic query-versus-broad-clothing direction: its negative extremes are generic tees/dresses, while its positive extremes include bright athletic ankle socks, bright sneakers, and a bright tote. C2 is approximately tee/basic-top versus boho/maxi-dress variation. C5 separates patterned/Westlake leggings and prints from cotton dresses/lingerie; black has a large C5 coefficient.

| Memory | Cosine | Tangent norm | rho | Projected norm | Dominant selected components (energy share) |
|---|---:|---:|---:|---:|---|
| budget: at most 120 (redundant) | .1805 | .9836 | .0169 | .1277 | C1 32.8%, C2 25.8%, C11 15.3% |
| color: black | .2926 | .9563 | .1104 | .3177 | C1 38.7%, C5 31.6%, C3 9.4% |
| style: minimal | .2437 | .9699 | .0578 | .2331 | C1 53.6%, C7 18.5%, C5 9.7% |
| style: understated | .2485 | .9686 | .0654 | .2478 | C1 51.9%, C7 14.8%, C10 7.5% |

Within the full numerical local row span, rank-16 retains 4.6% of budget's local-span energy, versus 21.1% for black, 12.6% for minimal, and 13.3% for understated. Budget is absent as structured embedding information; black and simple/style variation are real catalogue axes. This jointly supports representation and selected-variance bottlenecks for U2. Budget remains REDUNDANT for this query.

# F. U2 hard-condition contamination

Current M0 has one genuine hard condition here: `price_max = 120`. M0 maps missing/unparseable price to 9999 and therefore excludes it. Category and department are soft boosts, not hard masks.

```text
Top-500 total:          500
hard-compliant:          73
hard-incompatible:      427
percentage incompatible: 85.4%
```

Of the 427 incompatible rows, 426 have missing price and one is $139.68. Incompatibles occur throughout the list: 46 in the top 50, 90 in the top 100, rank median 239.

# G. U3 Top-500 composition

Query (no target leakage):

- Raw: `hey, i'm looking for a women's short sleeve tee made of rayon, like 95% rayon and a bit of spandex, ideally with a pull-on style.`
- Effective q text: `clothing short sleeve rayon spandex pull-on women`

| Attribute | Embedded text | Raw catalogue | Raw only | Distinct embedded forms |
|---|---:|---:|---:|---|
| Breathability | 23 (4.6%) | 103 (20.6%) | 80 | breathability, breathable, mesh, quick dry |
| Materials | 432 (86.4%) | 438 (87.6%) | 6 | 13 forms |
| Formal/dressy | 21 (4.2%) | 158 (31.6%) | 137 | business, cocktail, dressy, formal, office |
| Casual | 293 (58.6%) | 379 (75.8%) | 86 | 6 forms |
| Compression | 0 | 0 | 0 | none |
| Hooded | 1 (0.2%) | 2 (0.4%) | 1 | hoodie |
| Rain/waterproof | 0 | 0 | 0 | none |
| Cargo/storage | 33 (6.6%) | 59 (11.8%) | 26 | pocket, pockets |
| Winter/insulation | 1 (0.2%) | 67 (13.4%) | 66 | warmth, winter |
| Price (field/proxy) | 9 (1.8%) lexical proxies | 88 (17.6%) valid prices | 88 | 3 proxy forms |

Known prices span $5.99-$69.59 (p25 $16.99, median $21.99, p75 $25.99); 412/500 are missing. No affordable/premium proxy product has a known price, so a price-proxy correlation is not estimable. Query-score/price Spearman is -0.2008, which is incidental similarity, not price representation.

The neighbourhood is tee-oriented but still broad: T-shirts 115, tunics 100, blouses/button-downs 48, Casual dresses 44, Westlake 32, dresses 28, tanks/camis 26. Women is the second-level category for 453/500.

# H. U3 local geometry

`sigma_1..sigma_16`:

```text
12.0491, 3.6029, 2.8751, 2.6592, 2.2728, 2.2014, 2.0861, 2.0217,
2.0113, 1.8875, 1.8043, 1.7121, 1.6745, 1.6658, 1.6143, 1.5663
```

`sigma_1/sigma_16 = 7.6928`. Cumulative total tangent-matrix energy: C1 41.01%, C1-2 44.68%, C1-4 49.01%, C1-8 54.22%, C1-16 61.12%.

| Memory | Cosine | Tangent norm | rho | Projected norm | Dominant selected components (energy share) |
|---|---:|---:|---:|---:|---|
| budget: at most 120 | .1440 | .9896 | .0165 | .1270 | C2 30.6%, C1 28.1%, C8 14.0% |
| material: breathable | .3004 | .9538 | .0901 | .2864 | C1 47.8%, C8 15.1%, C6 12.9% |
| fit: compression | .3596 | .9331 | .0854 | .2726 | C5 19.7%, C4 16.6%, C1 14.6% |
| style: hooded | .3707 | .9287 | .0818 | .2657 | C1 44.4%, C5 27.9%, C11 6.2% |
| style: dressy | .4120 | .9112 | .1524 | .3557 | C2 41.1%, C1 27.6%, C6 15.3% |
| pockets: lots | .2599 | .9656 | .0564 | .2293 | C1 21.9%, C12 16.8%, C9 11.8% |
| insulation: insulated | .1815 | .9834 | .0233 | .1500 | C16 27.0%, C5 20.8%, C8 11.6% |

The formal result is geometrically interpretable. C2 positive extremes are rayon/maxi/party/cocktail dresses; negative extremes are basic tees/tunics. C6 positive extremes are long-sleeve, business, office, and button-down blouses; negative extremes are basic short-sleeve tees. `dressy` places 28.2% of its full-local-span energy inside rank-16, compared with 17.6% for breathable and 4.6% for budget. Formal therefore projects strongly because formal/casual and dress/tee variation is genuinely present and selected by the local SVD.

This is the decisive portability evidence: the geometry correctly recognizes a valid product axis but cannot know the historical formal requirement was episodic and conflicts with the current casual tee intent.

# I. U3 hard-condition contamination

Under current M0 semantics, this persisted state has no genuine deterministic hard condition. `women`, category, rayon/spandex, sleeve, and pull-on contribute to query/state and soft scoring; they are not M0 hard masks. The historical budget is not a current-state condition.

```text
Top-500 total:          500
hard-compliant:         500
hard-incompatible:        0
percentage incompatible:  0.0%
```

# J. Embedded-text vs raw-catalogue gap

| Preference axis | Raw catalogue representation | Embedded product text | Diagnostic consequence |
|---|---|---|---|
| Budget/price | Structured field; U2 74 known, U3 88 known | Price field excluded; only sparse accidental proxies | Projector cannot recover under-$120 semantics |
| General colour | U2 334/500 | 217/500 | Available but incomplete |
| Bright/colourful | U2 107/500 | 64/500 | Available, moderately sparse |
| Dark/black | U2 85/500 | 24/500 | Weak local representation |
| Sporty/athletic | U2 72/500 | 15/500 | Mostly outside embedded first-three-feature view |
| Minimal/understated | U2 120/500 | 29/500 | Mostly weak/proxy representation |
| Breathability | U3 103/500 | 23/500 | Useful axis exists but most evidence is hidden from projector |
| Material | U3 438/500 | 432/500 | Strongly represented |
| Formal/dressy | U3 158/500 | 21/500 | Sparse text still defines a strong selected dress/tee axis |
| Casual | U3 379/500 | 293/500 | Strongly represented |
| Compression | U3 0/500 | 0/500 | Absent from this neighbourhood |
| Hooded | U3 2/500 | 1/500 | Effectively absent; projection comes from correlated generic axes |
| Rain/waterproof | U3 0/500 | 0/500 | Absent from this neighbourhood |
| Cargo/storage | U3 59/500 | 33/500 | Weak/moderate proxy via pockets |
| Winter/insulation | U3 67/500 | 1/500 | Almost entirely outside embedded text |

# K. Candidate-universe counterfactual

Only U2 permits a nontrivial `diagnostic_post_hard_filter`: same q, same 50,000-row OpenAI matrix, same dot-product ordering, K=500, rank=16, restricted to M0 price-compliant rows. There are 9,794 eligible catalogue rows.

The filtered Top-500 overlaps the original Top-500 by only 73 products (14.6%). Socks rise from 8 to 12, Westlake falls from 48 to 9, sporty embedded coverage rises 3.0% to 5.0%, bright/colourful falls 12.8% to 4.0%, general colour falls 43.4% to 27.2%, and minimal/understated rises 5.8% to 6.2%.

Spectrum summary changes are modest: C1 energy 40.15% to 37.01%, C1-16 energy 60.02% to 57.04%, and `sigma_1/sigma_16` 7.70 to 7.34. But the rank-16 orientation changes materially: mean squared principal cosine 0.6116, chordal distance 2.4929/4, with the final three principal angles 70.9°, 79.3°, and 80.8°.

No U3 filtered counterfactual was constructed because current M0 supplies no hard condition; inventing gender/category/material hard masks would mix architectures.

# L. Failure classification

- REPRESENTATION: Outcome A is strong for budget and partial for breathability, sporty, formal, hooded/rain, and insulation. The projector cannot use structured fields it never embeds.
- VARIANCE: Outcome B is supported. Rank-16 over-selects broad product-type/style variation: U2 conflict axes have 12.6%-21.1% of local-span energy in rank-16 versus budget's 4.6%; U3 formal has 28.2% versus breathable 17.6% and budget 4.6%.
- PORTABILITY: Outcome C is supported most strongly by U3 formal. Formal/casual is real, local, and captured; none of that indicates whether a prior formal requirement should steer this session.
- CANDIDATE-UNIVERSE: plausible for U2 because the genuine M0 price constraint changes 85.4% of rows and materially rotates the subspace. It cannot explain U3, which has no current M0 hard constraint.

Catalogue support is not memory portability. QLMP asks whether a memory direction is expressible by local products; the desired controller must decide whether history should influence current intent. U3 demonstrates those questions are not equivalent.

# M. Required decisions

## Representation

`REPRESENTATION BOTTLENECK PARTIAL`

## Variance

`VARIANCE BOTTLENECK SUPPORTED`

## Portability

`PORTABILITY FAILURE SUPPORTED`

## Candidate universe

`FILTERED-UNIVERSE FOLLOW-UP JUSTIFIED`

# N. Original scientific verdict

`PROJECTOR STOP`

This diagnostic does not relabel, replace, or reopen the Phase-3A result.

# O. Smallest next experiment

Run exactly one separately preregistered filtered-universe projector-isolation experiment, limited to fixtures with persisted deterministic current-M0 hard constraints. Freeze the same q vectors, cache, K=500, rank=16, labels, and decision gates; use the current M0 hard semantics without importing `experiment_1`. This tests the diagnosed U2 candidate-universe failure mode only. It is not B3 and cannot reverse `PROJECTOR STOP` without new preregistered evidence.

# P. Tests

- QLMP core: 79 collected, 79 passed, 0 failed, 0 skipped.
- Targeted diagnostic + M0/projector/dense/state contracts: 65 collected, 65 passed, 0 failed, 0 skipped.
- Total: 144 collected, 144 passed, 0 failed, 0 skipped.
- Diagnostic replay: exact Top-K IDs/scores, singular spectra, memory projected norms, fixture/cache/catalogue/source hashes; 0 LLM and 0 OpenAI calls.

# Q. Scope audit

Confirmed: QLMP geometry unchanged; QLMP B1/B2 unchanged; M0 ranking unchanged; M0 routing unchanged; dense scorer unchanged; product embedding text unchanged; embedding model unchanged; `experiment_1` unchanged; official evaluator unchanged; Graphify not run; no commit. B3 was not implemented; q-star was not constructed; K/rank/model/text/geometry were not tuned or changed.
