# Vector Similarity & Distance Scoring Methods Reference Sheet

This document serves as a reference guide for similarity and distance scoring formulations for catalog search, retrieval, and re-ranking.

---

## 1. Retrieval Stages: DP1 vs. DP2

In conversational search, scoring is split into two distinct stages:

* **DP1 (Retrieval / Query-to-Catalog)**: Maps dynamic user intents/states onto the 50k-catalog embedding space. Mismatches in query length, semantic density, and token frequency make standard cosine similarity struggle at this stage.
* **DP2 (Neural Re-ranking / Candidate Scoring)**: Produces the final precision score to isolate the target product from close neighbors. Determines whether the agent should return recommendations or prompt the user for clarification.

---

## 2. Comparison of Similarity Formulations

| Method | Formula | Rationale / Why It Outperforms Cosine | Primary Use Case |
| :--- | :--- | :--- | :--- |
| **Max-Inner Product Search (MIPS)** | $S(q, d) = \langle q, d \rangle$ *(unnormalized)* | Preserves vector norms, which naturally encode catalog popularity, priors, and review density. | **DP1 (Candidate Retrieval)** |
| **Bilinear Transform** | $S(q, d) = q^T W d$ | Uses a learnable matrix $W$ to align subjective user phrasing with formal catalog schema attributes. | **DP1 & DP2** |
| **Late Interaction (MaxSim)** | $S(Q, D) = \sum_{i} \max_{j} \langle q_i, d_j \rangle$ | Operates on token-level vectors to prevent single-vector average collapse on multi-attribute queries. | **DP2 (Neural Re-ranking)** |
| **Softmax Cosine / Bregman** | $S(q, d) = \exp(\tau \cdot \cos(q, d))$ | Exponentiation and temperature scaling ($\tau$) widen score margins between top candidates. | **DP2 (Confidence Scoring)** |

### Detailed Analysis of Methods

#### 1. Maximum Inner Product Search (MIPS)
Unlike standard Cosine Similarity—which normalizes vector lengths to unit length ($||u|| = 1$)—MIPS preserves the raw magnitude of the vectors. In e-commerce, the magnitude of a product embedding ($||d||$) is not noise; it is a vital prior that correlates directly with the product's overall review volume, click-through rate, sales popularity, and catalog confidence. Stripping this norm away via cosine normalization forces retrieval to ignore these popularity signals, leading to cold-start problems and recommending obscure long-tail items.

#### 2. Bilinear Transform ($q^T W d$)
Traditional dot-product similarity assumes the query and catalog document reside in the exact same semantic vector space with a shared orientation. However, conversational customer queries (subjective, conversational, descriptive) and product catalog fields (structured keywords, size dimensions, brands) represent two different domains. Introducing a learnable metric tensor (interaction matrix $W$) allows the model to align subjective user phrasings (e.g., "comfy winter boots") with formal catalog attributes (e.g., "insulated lining", "rubber sole") without having to project both into a shared domain.

#### 3. Late Interaction (ColBERT / MaxSim)
Single-vector embedding models (Bi-encoders) compress an entire text sequence (e.g., a search with multiple slots: *"black waterproof running shoes size 10"*) into a single averaged vector. This compression suffers from information loss and the "hubness" problem: a strong match on one dominant keyword ("shoes") can completely drown out a critical mismatch on another ("waterproof"). Late interaction models keep token-level embeddings and compute token-by-token maximum similarities (MaxSim). Each query term is matched against all document terms individually, ensuring negation and specific constraints are explicitly checked.

#### 4. Softmax Cosine & Temperature Scaling
Raw cosine similarity scores often suffer from compression (e.g., rank #1 having $0.87$ similarity and rank #5 having $0.85$). To compute margins for confidence scoring, these values must be mapped to a probability distribution. Applying exponentiation and a scaling temperature factor ($\tau \approx 10\text{--}20$) sharpens this output distribution, magnifying tiny margin differences between close catalog items to maximize Mean Reciprocal Rank (MRR).

---

## 3. Implementation & Optimization Strategies

### In-Memory MIPS Acceleration
* Augment embeddings with a popularity coordinate to transform MIPS into standard Euclidean search, then query using fast libraries like `usearch` or `faiss.IndexFlatIP`. 
* Executes in under **$1\text{ ms}$** for a 50k catalog in Python, removing the need for external vector database infrastructure.

### Token-Level Late Interaction
Avoid heavy cross-encoder latency by computing token-wise matrix products in PyTorch:
```python
# Q: (B, num_q_tokens, dim)
# D: (B, num_d_tokens, dim)
sim_matrix = torch.bmm(Q, D.transpose(1, 2))  # Shape: (B, num_q_tokens, num_d_tokens)
score = sim_matrix.max(dim=-1).values.sum(dim=-1)
```

### Early Stopping Calibration
At the **DP2** stage, compute the temperature-scaled softmax margin between the top two recommendations. If $\Delta S = S_1 - S_2 \ge \tau$, immediately output the Top-10 recommendations to terminate the session early, minimizing the Mean Turns to Conversion (MTTC) metric.
