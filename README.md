# TechJam Conversational E-Commerce Search Challenge 2026

Welcome to the development workspace for the TechJam Conversational E-Commerce Search Challenge. This workspace is organized as follows:

## Repository Layout

```text
reptechjam2026/
├── README.md (Root workspace guide)
├── problem_statement.md (Challenge guidelines)
├── SHA256SUMS (Data checksums)
│
├── techjam-conversational-search/              (Active development directory)
│   ├── DATA_ATTRIBUTION.md
│   ├── README.md
│   ├── data/
│   │   ├── README.md
│   │   ├── catalog.jsonl                       (50,000 product frozen catalog)
│   │   └── public_set.jsonl                    (200 labeled public sessions)
│   ├── docs/
│   │   ├── agent_api_contract.json             (JSON validation schemas)
│   │   ├── baseline_results.json               (Weak starter metrics)
│   │   ├── competition_specification.md        (Core rules & metrics)
│   │   ├── evaluation_config.json              (Scoring configurations)
│   │   └── submission_rules.md                 (Submission criteria)
│   ├── evaluator/
│   │   ├── __init__.py
│   │   └── local_evaluator.py                  (Session simulator and scorer)
│   └── starter/
│       ├── __init__.py
│       └── agent.py                            (Editable weak agent baseline)
│
└── techjam-conversational-search-participant-kit/  (Participant kit template)
    ├── DATA_ATTRIBUTION.md
    ├── README.md
    ├── data/
    │   ├── README.md
    │   └── public_set.jsonl
    ├── docs/ (Same as main folder)
    ├── evaluator/ (Same as main folder)
    ├── starter/ (Same as main folder)
    └── tests/
        ├── __init__.py
        └── test_evaluator.py                   (Evaluator unit tests)
```

---

## Technical Overview

### 1. Core API Contract
Any custom agent must export the `Agent` class containing:

* **`Agent.reset`**: Starts a new session and loads the anonymized `user_profile`.
* **`Agent.respond`**: Receives a turn's `user_message` and outputs a response dictionary containing:
  * `"message"`: A natural-language response or clarifying question.
  * `"ask_attribute"`: The attribute being queried (one of `category`, `material`, `color`, `size`, `style`, `brand`, `budget`, `feature`, `use_case`, `other`, or `null`).
  * `"recommendations"`: An ordered list of up to 10 product identifiers (`parent_asin`).

### 2. Metrics and Scoring
The deterministic simulator evaluates the Agent through:
* **Hit Rate @ 10**: Fraction of sessions finding the target within 10 turns.
* **MRR (Mean Reciprocal Rank)**: Position-based ranking quality.
* **MTTC (Mean Turn To Conversion)**: Turn counts to hit target (misses count as 11).
* **Technical Score**: `0.50 * HitRate@10 + 0.30 * MRR + 0.20 * (11 - MTTC) / 10`.

### 3. How the Local Evaluator Works
The local evaluator is a deterministic, rule-based simulator. It does *not* use an LLM to generate customer replies dynamically, nor does it look at the natural language `"message"` returned by the agent. 

Instead, the evaluator simulates customer replies deterministically using the agent's `"ask_attribute"` output and the target product's metadata constraints:

* **Initial Step**: The customer sends an initial prompt revealing either the product category or the first constraint depending on the scenario type (`buying`, `browsing`, etc.).
* **Turn Loop**: Your agent processes the prompt and returns a response containing `"ask_attribute"` and `"recommendations"`.
* **Hit Evaluation**: The evaluator checks if the target product's `parent_asin` is in your recommendations. If so, the session ends in success (a Hit). If not, the evaluator determines the next user reply:
  * **No Valid Attribute**: If `"ask_attribute"` is null, empty, or not a string, the customer replies: *"Those options are not quite right yet. Ask me about one specific attribute."*
  * **Valid Attribute**: The evaluator looks for up to two undisclosed constraints matching that attribute type (e.g., mapping `"cotton"` to `"material"`). 
    * If matches exist: *"For that, what matters is: <constraint1>; <constraint2>."*
    * If no matches exist: *"I don't have an additional preference for <ask_attribute>."*
* **Scenario Exceptions**:
  * **Intent Override**: On turn 3 or 4, the customer ignores your query and injects: *"Actually, ignore my earlier preference. What I need is: <new_constraint>."*
  * **Boundary**: If the customer has no preferences for a requested attribute, they reply: *"I don't have a preference for <ask_attribute>; please use your judgment."*
