# Long-Term Memory System Flow

So, when considering both the user's current intent and their long-term memory, both are represented as L2-normalized vector embeddings, `v1` and `v2`.

1. First, we construct `v1` by embedding the current active session state—not just the latest prompt. This includes the current category, department, and positive preferences disclosed during the conversation. We load `v2`, the user's stored long-term preference vector, at the start of the session and keep it frozen throughout that session.

2. Next, we determine whether the long-term memory is relevant to the current intent. We calculate the cosine similarity between `v1` and `v2`. This is the **relevance gate**. If the similarity is below the active threshold of `0.30`, we ignore the long-term memory by setting `a = 1` and `b = 0`. If the user has no existing long-term memory, we use the same weights.

3. For every product in the 50,000-product catalogue, which is already represented as a normalized embedding `p_i`, we calculate its similarity to the current intent:

   $$
   s_{1i} = p_i^T v_1
   $$

   We also calculate its similarity to the long-term memory:

   $$
   s_{2i} = p_i^T v_2
   $$

   Because all the vectors are L2-normalized, these dot products are equivalent to cosine similarity.

4. For every product, we calculate the combined score:

   $$
   s_{3i} = a s_{1i} + b s_{2i}
   $$

   The values of `a` and `b` depend on whether the relevance gate passed and whether the user is buying or browsing:

   - If there is no long-term memory or the relevance gate fails: `a = 1.0`, `b = 0.0`
   - If the gate passes and the user is buying: `a = 0.8`, `b = 0.2`
   - If the gate passes and the user is browsing: `a = 0.2`, `b = 0.8`

   This lets the system consider both the current shopping intent and the user's long-term preferences. Buying emphasizes the current request, while browsing gives long-term preferences more influence.

5. We then apply hard constraints, such as the maximum price and explicitly negated terms. If enough eligible keyword matches exist, those products are ranked with the root agent's handcrafted state-evidence score, without long-term-memory reranking. Otherwise, we use the top 150 eligible products ordered by descending `s3` and ASIN as a vector fallback. Seen removal, the fixed top-10 `s1 >= 0.40` confidence gate, rank-preserving diversity, and `top_k` then run in that order.

6. At the end of the session, we extract only the positive, reusable preferences disclosed during the conversation. We do not store the transcript, negative preferences, budget, category, department, purchases, outcomes, or other session-specific information. These reusable preferences are embedded to produce a new vector, `v_new`.

7. Finally, we update the long-term-memory vector. For a returning user, we first measure how repetitive the evidence is:

   $$
   c = \operatorname{clip}(v_2^T v_{new}, 0, 1), \qquad \alpha = 0.30(1-c)
   $$

   $$
   v_3 = \operatorname{normalize}((1-\alpha)v_2 + \alpha v_{new})
   $$

   For a new user, we store `v_new` directly as `v3`. If the session contains no positive reusable preferences, the existing vector remains unchanged. Repetitive evidence moves the centroid less; novel or negatively aligned evidence receives at most the `0.30` update cap. This slows directional drift, but one centroid is not a true multi-interest memory. A fixed `alpha=0.30` policy remains available as an experimental control. The resulting `v3` is persisted and becomes the user's long-term-memory vector `v2` in their next session.
