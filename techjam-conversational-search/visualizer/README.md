# Agent flow visualizer

This tool compares the organizer-provided agent with the current agent over the public dataset. It writes every session and every turn to ordinary JSON, then presents the result in a searchable side-by-side dashboard.

## Generate the full comparison

From the workspace root:

```powershell
python techjam-conversational-search/visualizer/trace_agents.py
```

The default inputs are:

- Provided baseline: `techjam-conversational-search-participant-kit/starter/agent.py`
- Current system: `techjam-conversational-search/starter/agent.py`
- Dataset: `techjam-conversational-search/data/public_set.jsonl`
- Output: `techjam-conversational-search/visualizer/comparison.json`

The generator also writes `comparison.example.json`, a small standalone version of the automatically selected illustrative session.

The command accepts `--baseline-agent`, `--candidate-agent`, `--catalog`, `--dataset`, and `--output` paths. Use `--sample-id public_0001` for one example or `--limit 10` for a quick run.

## Open the dashboard

Start a local static server:

```powershell
python -m http.server 8000 --directory techjam-conversational-search/visualizer
```

Then open <http://localhost:8000>. The dashboard can also load another generated JSON file through **Open another trace**.

## What the JSON contains

- dataset and catalog counts;
- aggregate metrics and deltas for both agents;
- all public sessions and target product previews;
- the deterministic simulator's intent card and scenario behavior;
- every customer message, agent response, requested attribute, Top-10 slate, target rank, error, and token report;
- a per-session comparison status and a selected example session.

The trace is deliberately testing-only: it exposes hidden simulator truth and target products that an agent must never use during a real evaluation.
