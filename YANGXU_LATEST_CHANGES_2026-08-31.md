# Yangxu's Latest Main-Branch Changes — 2026-08-31

## Scope

- Previous `main` commit: `956f0599accc2c40038632e94392770ff0466f03` — `updated agent, documentation, and UI`
- New `origin/main` commit: `1404ee1419af579822c4dd867112de92595e35c0` — `updated agent and UI`
- Author and committer: **YangXu624 `<yangxu624@gmail.com>`**
- Commit time: **2026-08-31 00:24:05 +08:00**
- Comparison used: `956f059..1404ee1`

The new commit was fetched and inspected directly from `origin/main`. It was not checked out over the current worktree because the worktree already marks the affected tracked `experiment_1` files as deleted. Those local changes were left untouched.

## Change summary

Yangxu's new push changes **4 files**, with **2,049 insertions and 1,235 deletions**.

| File | Change | Insertions | Deletions | Main purpose |
|---|---:|---:|---:|---|
| `experiment_1/shop_agent.py` | Modified | 157 | 63 | Adds buying-versus-browsing intent modes and profile-gated semantic reranking. |
| `experiment_1/visualizer/conversation.html` | Added | 1,330 | 0 | Moves the conversational simulation interface into its own page. |
| `experiment_1/visualizer/index.html` | Reworked | 458 | 1,172 | Replaces the old conversation-first landing page with a searchable product catalog. |
| `experiment_1/visualizer/server.py` | Modified | 104 | 0 | Adds routes for the split UI and a paginated catalog-search API. |

## 1. Intent-aware retrieval changes

File: `experiment_1/shop_agent.py`

### Buying and browsing modes

The agent now stores an `intent_mode` for every session. The allowed values are:

- `buying`: the shopper has specific requirements or appears close to a purchase decision;
- `browsing`: the shopper is exploring, vague, or open to suggestions.

The dialogue-state LLM is instructed to re-evaluate this mode on every turn. It should move from browsing to buying as requirements become concrete and return to browsing only after an explicit reset.

The mode is also exposed in agent debug telemetry.

### Intent-specific retrieval thresholds

The retrieval cascade now changes its fallback thresholds by intent:

| Intent | Minimum strict-AND results before widening | Minimum results before vector fallback | Effect |
|---|---:|---:|---|
| Buying | 15 | 10 | More precision-oriented; retains strict lexical matches longer. |
| Browsing | 30 | 15 | More recall-oriented; widens to OR matching and vector retrieval sooner. |

This makes open-ended browsing produce a broader candidate pool while preserving tighter filtering for high-intent purchase requests.

### Intent-specific clarification priorities

`_select_best_attributes_to_ask()` now accepts the live intent mode and uses an intent-specific priority list as an entropy-score tiebreaker:

- Buying: material → brand → color → size → style → use case → budget.
- Browsing: use case → style → brand → material → color → size → budget.

If there are no candidates yet, these priority orders directly determine which unasked attributes are selected. When candidates exist, Shannon-entropy/information-gain scoring remains primary and intent priority resolves ties or low-information cases.

## 2. Long-term profile-gated reranking

File: `experiment_1/shop_agent.py`

### Profile representation

At session reset, the agent builds a profile string from:

- `preference_tags`;
- profile `summary`;
- `purchase_frequency`.

It encodes and normalizes this text once, then stores the resulting embedding as `profile_emb` in session state. Encoding failure is treated as non-fatal and simply disables the profile signal.

### Gate behavior

The agent now computes a normalized query embedding on every custom-agent turn and compares it with the stored profile embedding.

- Gate threshold: cosine similarity of **0.25**.
- Below the threshold: ranking remains based on the current conversational state.
- At or above the threshold: product/profile semantic similarity is blended into candidate scores.

### Intent-dependent score blending

| Intent | Current-state score | Long-term profile score |
|---|---:|---:|
| Buying | 85% | 15% |
| Browsing | 40% | 60% |

The rationale encoded in the implementation is that a shopper with a concrete buying request should be driven mainly by the current requirements, while open-ended browsing can lean more heavily on persistent taste.

Candidate profile similarities are calculated in a vectorized matrix operation. An ASIN-to-catalog-index dictionary was added so candidate embeddings can be located in constant time.

New debug fields report:

- query/profile similarity;
- whether the profile gate opened;
- whether profile reranking ran;
- detected intent mode.

## 3. UI split: catalog and conversation

Files:

- `experiment_1/visualizer/index.html`
- `experiment_1/visualizer/conversation.html`

The visualizer is now split into two distinct experiences.

### Product catalog landing page

`index.html` has been substantially rewritten and is now titled **ASTRA — Product Catalog**. It provides:

- free-text product search;
- department tabs for All, Clothing, Shoes, and Jewelry;
- maximum-price filtering;
- minimum-rating filtering at 3, 4, or 4.5 stars;
- 24-product pages with pagination;
- product cards showing image/fallback artwork, title, brand, price, rating, review count, categories, and ASIN;
- loading, empty-result, and connection-error states;
- a navigation link to `/conversation`.

Search input is debounced, while department, price, rating, and page changes trigger immediate refreshes.

### Dedicated conversation page

The new `conversation.html` contains the conversational-search and evaluation interface that previously dominated `index.html`. It includes:

- session search and scenario filtering;
- automated and manual simulation modes;
- Server-Sent Events for automated conversations;
- manual shopper-message submission;
- recommendation cards with images, rank, ASIN, and target markers;
- hidden-target metadata and constraint inspection;
- selected-product inspection;
- simulation status and success reporting.

This separation makes `/` a catalog-browsing surface and `/conversation` the agent-testing/debugging surface.

## 4. New server routes and catalog API

File: `experiment_1/visualizer/server.py`

### Routes

- `GET /conversation` serves `conversation.html` explicitly with no-cache headers.
- `GET /catalog_search` performs server-side filtering and returns JSON.
- Existing simulation, session-list, manual-step, and SSE routes remain available.

### Catalog-search parameters

| Parameter | Meaning |
|---|---|
| `q` | Case-insensitive substring match against title, brand, or categories. |
| `dept` | Department grouping: `all`, `clothing`, `shoes`, or `jewelry`. |
| `max_price` | Optional maximum price. |
| `min_rating` | Optional minimum average rating. |
| `page` | One-based result page. |

The endpoint returns up to 24 products per page with total count, current page, and product metadata. Empty searches are sorted by review count so popular products appear first. Product images come from the existing ASIN-image mapping.

## Practical impact

- Search behavior can now adapt to whether a shopper is exploring or ready to buy.
- Long-term profile memory affects rankings only when the current query is sufficiently related to the profile.
- Profile influence is deliberately stronger during browsing and weaker during focused buying.
- The front end now supports both ordinary catalog exploration and the existing conversation/evaluation workflow.
- The server exposes a lightweight catalog-search API that can be reused independently of the conversation simulator.

## Review notes and watch items

These are implementation observations, not confirmed runtime failures:

1. **Query embeddings are now computed every custom-agent turn.** This happens even when the session has no usable profile, because the same embedding is shared with vector fallback. It may add latency to turns that previously completed entirely through lexical retrieval.
2. **Catalog filtering stops after collecting 2,000 matches.** On an empty query, those first 2,000 products are then sorted by review count. The landing page therefore may not show the true most-reviewed products across a catalog larger than 2,000 matching entries.
3. **Missing price/rating values pass numeric filters.** A zero or unavailable price is not rejected by `max_price`, and a zero or unavailable rating is not rejected by `min_rating`. This follows the agent's existing “benefit of the doubt” behavior but may surprise catalog users.
4. **The catalog category tabs omit watches.** The agent supports a watches root department, but the new catalog UI and server keyword map expose only clothing, shoes, and jewelry.
5. **The new conversation page contains trailing whitespace.** `git diff --check` reports formatting warnings in `conversation.html`; these do not affect browser behavior.

## Validation performed

- Confirmed `origin/main` advanced by exactly one commit from `956f059` to `1404ee1`.
- Confirmed the commit author and committer are YangXu624.
- Verified the exact four-file diff and insertion/deletion counts.
- Compiled the fetched `shop_agent.py` and `visualizer/server.py` source successfully with Python's parser.
- Ran `git diff --check`; only trailing-whitespace warnings were reported in the new HTML file.
- No local tracked files were overwritten or restored during this inspection.

## Reference commands

```text
git show 1404ee1419af579822c4dd867112de92595e35c0
git diff 956f0599accc2c40038632e94392770ff0466f03..1404ee1419af579822c4dd867112de92595e35c0
```

