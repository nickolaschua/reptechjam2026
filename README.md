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
