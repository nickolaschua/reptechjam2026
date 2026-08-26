# NLP & Embedder Selection, Evaluation, and Fine-Tuning Guide

This document outlines the candidate models, evaluation metrics, and fine-tuning strategies for the **NLP & Embedder** workstream of the Shopping Copilot project.

---

## 1. Candidate Embedder & NLP Models

To understand conversational queries and match them to clothing catalog items, we focus on compact, transformer-based models that run efficiently on CPU.

### A. Dense Embedder Models (Bi-Encoders)
* **`sentence-transformers/all-MiniLM-L6-v2`**
  * *Overview*: 384-dimensional embeddings. Small footprint (~80 MB) and extremely fast inference.
  * *Pros*: Ideal for in-memory CPU cosine/MIPS similarity. Very low latency.
* **`BAAI/bge-small-en-v1.5`**
  * *Overview*: 384-dimensional embeddings. Top performer on the MTEB leaderboard for retrieval tasks.
  * *Pros*: Strong zero-shot retrieval accuracy. Includes instructions for query prefixing (e.g., adding `"Represent this sentence for searching relevant passages:"` to query vectors).
* **`sentence-transformers/all-mpnet-base-v2`**
  * *Overview*: 768-dimensional embeddings (~420 MB).
  * *Pros*: High representation accuracy; captures fine-grained attributes better than MiniLM, at the cost of slightly higher search latency.

### B. NLP Slot Fillers & Cross-Encoders
* **`microsoft/deberta-v3-small` / `deberta-v3-base`**
  * *Overview*: Excellent for classification and token-level classification (NER/Slot-filling).
  * *Pros*: Can be fine-tuned to extract product slots (e.g., color, size) or classify session intents (Buying vs. Browsing).
* **`BAAI/bge-reranker-base`**
  * *Overview*: A Cross-Encoder model.
  * *Pros*: Computes token-level joint attention over the query and candidate document. Too slow for retrieving 50k items (DP1), but perfect for re-ranking the Top-50 candidates (DP2).

---

## 2. Evaluation Metrics

To assess which embedder works best, evaluate candidates on the 200 public sessions using these metrics:

* **Retrieval Recall / Coverage (Hit Rate @ K)**:
  * Measures whether the correct target product falls within the top $K$ retrieved candidates (e.g., Hit Rate@50 or Hit Rate@100). Higher recall ensures the re-ranker stage has the target product in its pool.
* **Mean Reciprocal Rank (MRR @ 10)**:
  * Measures how close to the top position the target product is placed in the final list.
* **Latency (Inference Speed)**:
  * Milliseconds taken to encode a single query text (critical to keep the interactive multi-turn loop under latency bounds).
* **Cosine Separation Margin ($\Delta S$)**:
  * The average similarity gap between the true target product and the next closest negative product. A larger gap indicates a more robust embedding space.

---

## 3. Fine-Tuning Strategies for the Clothing Domain

Since the generic pre-trained models do not know specific fashion styles, brands, or clothing-specific terms, fine-tuning them is crucial.

### Strategy A: Unsupervised Domain Adaptation (TSDAE)
* **Goal**: Teach the embedder the fashion vocabulary and brand catalog names without using labeled sessions.
* **How It Works**: 
  * Use **TSDAE (Transformer-based Sequential Denoising Auto-Encoder)** from the `sentence-transformers` library.
  * Train the model to reconstruct catalog product texts (`catalog.jsonl` titles, categories, and descriptions) from corrupted versions.
  * *Result*: The embedder learns clothing-specific terminology (e.g., "breathable mesh", "memory foam sole").

### Strategy B: Contrastive Learning on Labeled Sessions (MNRL)
* **Goal**: Teach the embedder to align conversational user queries with catalog product descriptions.
* **How It Works**:
  * Extract pairs from [public_set.jsonl](file:///Users/yangxu/code/reptechjam2026/techjam-conversational-search/data/public_set.jsonl). Pair the initial customer message (e.g., *"I'm looking for a cotton black shirt"*) with the matching target product's catalog details.
  * Train using **MultipleNegativesRankingLoss (MNRL)**. The loss treats the correct query-product pair as a positive instance, and all other products in the training batch as negative instances.

### Strategy C: Generative Pseudo-Labeling (GPL)
* **Goal**: Scale up positive training pairs if the 200 public sessions are too few.
* **How It Works**:
  * Take product titles from the catalog and use a generative model (like a local LLM or T5) to write synthetic customer queries (e.g., *"Show me some leather boots under $80"*).
  * Pair these synthetic queries with their source products to generate tens of thousands of positive pairs for contrastive fine-tuning.

---

## 4. Standalone Retrieval Harness (Testing Independently)

To evaluate your embedder's retrieval capabilities before integrating it with other components, you can use the script below. It builds an in-memory index of the 50k catalog and checks if the correct target product is successfully retrieved in the Top-$K$ results for each public session.

Create a test script (e.g., `yangxu/test_retrieval.py`) containing:

```python
import json
import time
import numpy as np
from sentence_transformers import SentenceTransformer

# 1. Load Embedder Model (Runs 100% Offline)
model_name = "sentence-transformers/all-MiniLM-L6-v2"
print(f"Loading {model_name}...")
model = SentenceTransformer(model_name)

# 2. Load Catalog
catalog_ids = []
catalog_texts = []
catalog_path = "techjam-conversational-search/data/catalog.jsonl"

print(f"Loading catalog from {catalog_path}...")
with open(catalog_path, "r", encoding="utf-8") as f:
    for line in f:
        p = json.loads(line)
        catalog_ids.append(str(p["parent_asin"]))
        # Combine key metadata fields for indexing
        text = f"{p.get('title', '')} {' '.join(p.get('categories', []))} {' '.join(p.get('features', []))}"
        catalog_texts.append(text)

# 3. Generate Catalog Embeddings (Magnitude Preserved)
print("Encoding 50,000 product descriptions (this may take a few minutes on CPU)...")
start_time = time.time()
catalog_embeddings = model.encode(catalog_texts, batch_size=256, show_progress_bar=True, convert_to_numpy=True)
print(f"Catalog encoded in {time.time() - start_time:.2f}s")

# Normalize for standard cosine search OR keep raw for MIPS
norms = np.linalg.norm(catalog_embeddings, axis=1, keepdims=True)
catalog_embeddings_normalized = catalog_embeddings / np.maximum(norms, 1e-12)

# 4. Load Public Sessions
queries = []
targets = []
dataset_path = "techjam-conversational-search/data/public_set.jsonl"

with open(dataset_path, "r", encoding="utf-8") as f:
    for line in f:
        s = json.loads(line)
        targets.append(str(s["ground_truth"]["parent_asin"]))
        # Extract initial search query signals from the user profile summary
        queries.append(s["user_profile"]["summary"])

print(f"Encoding {len(queries)} evaluation queries...")
query_embeddings = model.encode(queries, convert_to_numpy=True)
query_norms = np.linalg.norm(query_embeddings, axis=1, keepdims=True)
query_embeddings_normalized = query_embeddings / np.maximum(query_norms, 1e-12)

# 5. Evaluate Retrieval Recall@K (MIPS vs. Cosine)
print("\n--- Running Evaluation ---")
for search_type, cat_embs, q_embs in [
    ("Cosine Similarity", catalog_embeddings_normalized, query_embeddings_normalized),
    ("Maximum Inner Product (MIPS)", catalog_embeddings, query_embeddings)
]:
    hits_at_k = {10: 0, 50: 0, 100: 0}
    start_time = time.time()
    
    for idx, q_emb in enumerate(q_embs):
        target = targets[idx]
        # Direct matrix multiplication over the 50k items
        scores = np.dot(cat_embs, q_emb)
        top_indices = np.argsort(scores)[::-1][:100]
        top_asins = [catalog_ids[i] for i in top_indices]
        
        for k in hits_at_k:
            if target in top_asins[:k]:
                hits_at_k[k] += 1
                
    latency_ms = ((time.time() - start_time) / len(queries)) * 1000
    print(f"\n[{search_type}] (Avg Latency: {latency_ms:.2f}ms/query)")
    for k, hit_count in hits_at_k.items():
        print(f"  Recall@{k} (Hit Rate@{k}): {hit_count / len(targets):.4f}")
```

---

## 5. Grading Constraints & System Alignment

When optimizing your embedder, keep the following constraints from [problem_statement.md](file:///Users/yangxu/code/reptechjam2026/problem_statement.md) in mind:

1. **Strict Offline Execution**: The final scoring sandbox may have internet access completely disabled. You **cannot** use external API-based embedders (like OpenAI or Cohere) for retrieval. All transformer weights must be loaded locally.
   * *Strategy*: Pre-download your SentenceTransformer weights (e.g., saving model directories to `src/models/` in your submission bundle) and point your script to the local directories rather than Hugging Face Hub.
2. **In-Memory Constraints**: Large external vector database clusters (like Milvus or Qdrant) are out of scope. 
   * *Strategy*: Use `numpy` matrix calculations or lightweight local vectors like `faiss-cpu` / `usearch` which easily load inside your Python Agent container memory.
3. **Turn Limits & MTTC**: You have a maximum of 10 interaction turns. 
   * *Strategy*: Standalone Recall@K measures how early the target is placed in the candidate list. Higher Recall@10 on raw retrieval translates to fewer turns (MTTC) and higher MRR scores.