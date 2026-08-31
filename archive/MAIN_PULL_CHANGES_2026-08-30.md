# Main Pull Changes — 2026-08-30

## Pull result

- Command run: `git pull --no-rebase origin main`
- Result: **Already up to date**
- Current branch: `nickolas`
- Current `HEAD` / `origin/main`: `956f0599accc2c40038632e94392770ff0466f03`
- The latest actual fast-forward from `main` occurred at **2026-08-30 16:17:14 +08:00**, moving this branch from `c840a8b` to `956f059`.
- This report therefore describes that latest meaningful pull range: `c840a8b..956f059`.

> Note: the worktree already contained many uncommitted changes before this check. They were preserved and are not counted as pulled changes in this report.

## Net changes introduced by the pull

The effective diff contains **7 files changed, 51,345 insertions, and 264 deletions**.

| File | Change | Insertions | Deletions | Summary |
|---|---:|---:|---:|---|
| `experiment_1/agent_doc.md` | Modified | 173 | 38 | Expanded the architecture documentation for lexical matching, categorical filtering, vector search, and entropy-based clarification. |
| `experiment_1/asin_images.json` | Added | 49,946 | 0 | Added a large ASIN-to-product-image lookup used by the visualizer. |
| `experiment_1/run_eval_v2.py` | Modified | 23 | 2 | Made the evaluator maintain and apply explicit shopper intent state across override turns. |
| `experiment_1/shop_agent.py` | Modified | 758 | 172 | Added richer hard filters, canonical department handling, entropy-based question selection, and related retrieval/state changes. |
| `experiment_1/shopper_agent.py` | Modified | 129 | 10 | Added constraint provenance and revocation so overridden preferences do not leak into later shopper turns. |
| `experiment_1/visualizer/index.html` | Modified | 220 | 24 | Expanded product cards/inspection UI and added real or category-fallback product images. |
| `experiment_1/visualizer/server.py` | Modified | 96 | 18 | Added target/product metadata and image mapping to manual and streaming visualizer responses. |

## Commit history represented in the pull

The range includes merged ancestry as well as first-parent commits. Individual commit stats overlap; they must not be added together. The table above is the authoritative net diff.

| Commit | Author | Date (+08:00) | Purpose |
|---|---|---|---|
| `4735c4d` | gsharsh | 2026-08-29 18:38 | Fixed intent-override state transitions in `shop_agent.py`. |
| `96a3e55` | gsharsh | 2026-08-29 18:48 | Added shopper-side tracking for intent overrides and connected it to evaluation. |
| `f70747c` | Nickolas Chua | 2026-08-29 21:19 | Merged pull request #2 from `nickolas`; this is merged ancestry whose Nickolas-side content was already present at the old branch tip. |
| `43c9bd6` | gsharsh | 2026-08-29 22:22 | Made an override a real state transition across the agent, shopper simulator, and evaluator. |
| `dc814f2` | Nickolas Chua | 2026-08-30 15:02 | Merged Nickolas's updates into `main`; this brought the prior Nickolas tip into the first-parent history. |
| `956f059` | **YangXu624** | 2026-08-30 15:36 | Updated the agent, technical documentation, product-image data, and visualizer UI/server. |

## Intent-override changes

The gsharsh commits make intent overrides persistent state changes rather than one-off prompt text:

- Added `ShopperIntentState`, which records each constraint's value, source turn, source type, and active/revoked status.
- When an override occurs, prior initial or explicit preferences are revoked, the new preference becomes active, and the search epoch increments.
- Rebuilds the shopper system prompt after an override.
- Removes revoked values from hard constraints and soft preferences supplied to the simulated shopper.
- Explicitly instructs the shopper not to mention or reinforce revoked preferences.
- Wires this stateful behavior into both interactive execution and `run_eval_v2.py` evaluation runs.
- Updates `shop_agent.py` so clearing or changing an attribute also resets the corresponding hard-condition state.

## Yangxu's changes — `956f059`

Yangxu's commit is the final commit in the pull and changes **5 files: 50,981 insertions and 154 deletions**. Most insertions come from the 49,946-line image mapping.

### 1. Agent and retrieval pipeline

File: `experiment_1/shop_agent.py` — **546 insertions and 74 deletions**.

Key changes:

- Added `standardize_department()` to normalize raw catalog department values into consistent demographic buckets such as women, men, girls, boys, unisex, baby/toddler, multi-demographic, and unspecified.
- Precomputes catalog arrays for canonical department, average rating, rating count, and brand/store so categorical filters can be applied efficiently.
- Expanded conversation/search state with hard conditions for:
  - target department;
  - minimum average star rating;
  - minimum review count;
  - store/brand;
  - existing maximum-price filtering.
- Parses these constraints from both local text handling and structured LLM output.
- Applies vectorized masks before lexical/vector ranking. Missing rating data is allowed as a safe fallback, while explicit brand matching is exact and case-normalized.
- Slices the embedding matrix to the hard-filtered catalog before semantic maximum-inner-product search.
- Added `_select_best_attributes_to_ask()`, which chooses up to two clarification attributes using:
  - candidate-space entropy;
  - expected conditional entropy;
  - information gain;
  - C4.5 gain ratio;
  - coverage/sparsity adjustment;
  - a minimum-usefulness safeguard.
- Avoids asking for attributes that are already satisfied by hard conditions and records the selected clarification attributes in state.
- Expanded debug output to expose department, rating, review-count, and store constraints.

### 2. Technical documentation

File: `experiment_1/agent_doc.md` — **211 lines changed**.

The documentation now describes a layered retrieval design:

1. Lexical keyword matching, including verification/routing, regex parsing, token rebuilding, fallback cascades, post-scoring, and diversification.
2. Vectorized categorical filtering with standardized demographic buckets and hard-mask slicing.
3. Shannon-entropy attribute selection, including the formulas and selection rule for information gain, gain ratio, and coverage adjustment.
4. Semantic vector search using fine-tuned BGE embeddings and sliced MIPS retrieval.
5. The full unified cascade from state ingestion through filtering, lexical routing, semantic fallback, reranking, and clarification selection.

### 3. Product image data

File: `experiment_1/asin_images.json` — **new, 49,946 lines**.

- Maps catalog ASINs to image URLs.
- Is loaded once by the visualizer server at startup.
- Supplies recommendation-card imagery without changing the main catalog format.

### 4. Visualizer server

File: `experiment_1/visualizer/server.py` — **114 lines changed**.

- Retains the full product catalog in `GLOBAL_PRODUCTS` for metadata lookup.
- Loads `asin_images.json` into `GLOBAL_IMAGE_MAPPING`, with startup logging and graceful failure handling.
- Sends richer target metadata: title, brand, features, description, details, rating, review count, price, and categories.
- Adds `target_asin`, target title/brand, and target metadata to manual-session and SSE payloads.
- Adds image URLs and `is_target` flags to recommendations.
- Uses the target ASIN consistently for hit detection and rank calculation.
- Skips samples whose target ASIN is absent from the catalog instead of failing later.

### 5. Visualizer UI

File: `experiment_1/visualizer/index.html` — **244 lines changed**.

- Displays mapped product images when available.
- Provides category-based fallback images when a mapping is missing.
- Expands recommendation details, including features and structured product metadata.
- Adds a selected-product inspector for viewing richer information about a recommendation.
- Improves target-product presentation and passes full product objects through the UI for inspection.

## Practical impact

- Intent overrides should now reliably remove obsolete shopper preferences from later turns.
- Retrieval can enforce more real-world hard constraints before ranking, reducing irrelevant candidates.
- Clarification questions are selected from the current candidate distribution rather than a fixed attribute order.
- The visualizer can show recognizable products, expose why a result matched, and identify the hidden target during evaluation/debugging.
- The documentation now matches the expanded three-layer retrieval and entropy-selection design.

## Verification references

- Old branch tip: `c840a8b` (`document memory milestones and baseline evidence`)
- New branch tip: `956f059` (`updated agent, documentation, and UI`)
- Net comparison: `git diff c840a8b..956f059`
- Yangxu-only comparison: `git show 956f059`
