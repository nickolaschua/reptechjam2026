# Shopping Copilot: System Architecture & Workstream Proposal

> **Archived proposal:** Preserved for design history; this is not the current architecture.

This document outlines the system architecture and workstream breakdown for a 5-person team developing the Shopping Copilot agent.

---

## 1. Core Architecture Modules

The agent architecture is structured into five core functional modules:

### I. NLP & Embedder (Input Understanding)
* **Goal**: Analyze the user's incoming natural language message to extract search intent and structure query signals.
* **Key Features**:
  * **Structured Slot Parser**: Uses lightweight NLP rules or an in-memory small model to parse messages into attribute slots (e.g., `color`, `material`, `budget`, `style`).
  * **Query Embedder**: Generates dense semantic vector embeddings for search queries using a lightweight, local model (e.g., SentenceTransformers).
  * **Intent Classifier**: Identifies whether the user is in "Buying" mode (seeking specific filters/constraints) or "Browsing" mode (open-ended description).

### II. Dialogue Manager & LLM API Engine (Conversational Interface)
* **Goal**: Manage dialogue flow and call LLM APIs to generate clarifying questions and select target attributes.
* **Key Features**:
  * **Structured JSON Generation**: Prompts the LLM (e.g., GPT-4o or a local LLM) to return a structured JSON response matching the competition contract, including `"message"` and `"ask_attribute"`.
  * **Entropy-Based Attribute Elicitation**: Calculates information gain across the candidate product pool to steer the LLM toward asking for the attribute that bisects the remaining candidate space most efficiently.
  * **Offline Fallback Track**: Detects API key absence or network failures to fall back onto a local, rule-based dialogue strategy (such as standard BM25 + deterministic question prompts) to ensure the system passes final judging under offline environments.

### III. Session Memory & State Tracker (Context Persistence)
* **Goal**: Track session-level state, dialogue history, and accumulated slot constraints across turns.
* **Key Features**:
  * **Persistent Memory Layer**: Implements a structured store (reset per `session_id`) to track user message histories and accumulated preferences.
  * **Intent Override Engine**: Detects explicit topic shifts (e.g., *"Actually, forget the leather boots, show me red canvas sneakers instead"*) to erase contradictory memory slots while persisting invariant parameters (like budget limits).
  * **Personalized Context Distillation**: Incorporates the safe, anonymized aggregate `user_profile` into the active memory to personalize recommendations from Turn 1.

### IV. Retrieval Engine (Hybrid Sparse & Dense Catalog Search)
* **Goal**: Query the frozen 50,000 catalog (`catalog.jsonl`) to return a high-recall candidate pool.
* **Key Features**:
  * **Sparse Search (BM25)**: SQLite FTS5 index for fast exact matching on category, store, title, and key features.
  * **Dense Search (MIPS)**: Maximum Inner Product Search (MIPS) on unnormalized / magnitude-scaled vector embeddings in-memory to preserve popularity and review authority signals.
  * **Hybrid Fusion**: Combines sparse and dense candidate pools using reciprocal rank fusion (RRF) or custom weights.

### V. Ranking Engine (Top-10 Sorting & Re-ranking)
* **Goal**: Fine-tune the sorted order of the retrieved candidate pool to place the exact target product in the Top 1.
* **Key Features**:
  * **Late Interaction Ranking**: Token-level vector similarity (e.g., MaxSim) to compute fine-grained matches between search terms and catalog titles/descriptions.
  * **Early Termination Rule**: Computes the confidence gap between Top-1 and Top-2 recommendations; triggers an early termination recommendation list when confidence exceeds threshold $\tau$, avoiding unnecessary dialogue turns and reducing MTTC.

---

## 2. Evaluation Matrix & Objective Mapping

| Competition Metric | System Submodule Responsible | Implementation Strategy |
| :--- | :--- | :--- |
| **Hit Rate@10** | IV. Retrieval Engine | Combined BM25 and dense MIPS candidate pooling. |
| **MRR (Mean Reciprocal Rank)** | V. Ranking Engine | Fine-grained token interaction and confidence-based sorting. |
| **MTTC (Mean Turns to Conversion)** | II. Dialogue Manager & III. Memory System | Entropy-driven `ask_attribute` selection, intent override handling, and early stopping. |

---

## 3. Workstream Breakdown

* **Workstream 1: NLP and Embedder**
  * Parses incoming user messages into structured slots and generates dense query embeddings to understand user input.
* **Workstream 2: LLM API Conversations**
  * Manages the conversational flow with the user, prompting the LLM for natural-language clarifying questions and structured attributes.
* **Workstream 3: Memory System**
  * Tracks persistent session histories and slot configurations to remember recent user prompts and handle intent overrides.
* **Workstream 4: Retrieval**
  * Interfaces with the frozen catalog to execute hybrid sparse and dense candidate searches.
* **Workstream 5: Ranking**
  * Re-ranks retrieval candidates to sort target products into the Top-10 and executes early-stopping criteria.

---

## 4. Feasibility & Implementation Considerations

When building out this architecture, keep the following competition constraints and workarounds in mind:

* **Offline Score Environment**: Because the organizer may run final evaluation scripts under network-restricted, headless environments with no live API credentials, Module II *must* include an offline fallback tracker (such as local heuristics, deterministic queries, or small local model fallback) so that the agent doesn't throw exceptions and fail evaluation.
* **Dialogue Feedback Mapping**: The deterministic simulator relies solely on the structured `"ask_attribute"` field returned in your response to decide what details to reveal next. It completely ignores the natural-language `"message"`. Ensure your LLM prompts enforce strict JSON formats returning a valid attribute enum value along with user-facing text.
* **Latency & Token Efficiency**: Evaluating hundreds of multi-turn sessions can lead to thousands of LLM API requests, causing timeouts or high model token costs. Optimize the system by utilizing local classification to route traffic, and apply early-stopping thresholds when score separation in the Ranker is high.
