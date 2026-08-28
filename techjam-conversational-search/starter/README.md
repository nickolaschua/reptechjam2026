# Current competition agent

This is the production packaging of the Experiment 7 recommendation: a stateful exact-phrase ranker with a conditional generic-BM25 reciprocal-rank-fusion fallback.

## Behavior

- preserves category and disclosed constraints for each session;
- asks for `other`, the evaluator's allowed catch-all attribute, to reveal remaining requirements;
- removes the initial preference when an intent-override message arrives;
- ranks exact phrase coverage first;
- activates equal-weight exact + BM25 RRF when evidence is absent, ambiguous, or has no complete catalog match;
- uses no target identifiers, public labels, network calls, API keys, or user-profile data for ranking.

If NumPy, SciPy, or scikit-learn is unavailable, the agent automatically uses a smaller stateful SQLite FTS5 fallback.

## Install

```powershell
python -m pip install -r starter/requirements.txt
```

## Evaluate

From `techjam-conversational-search`:

```powershell
python -m evaluator.local_evaluator --output results.json
```
