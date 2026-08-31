# Resolver pass on ubuntu-makers

Nothing beyond Python 3.10+ stdlib is needed on the server. `USER` below is your
server username. Run every command from the **laptop**, repo root.

```bash
# 1. Tailscale (school login in the browser), then confirm the node is visible
tailscale up && tailscale status | grep ubuntu-makers

# 2. Push code + data + prebuilt index + current partial results (~260 MB, once)
rsync -avz --relative \
  winston/ \
  techjam-conversational-search/data/catalog.jsonl \
  techjam-conversational-search/evaluator/ \
  techjam-conversational-search/starter/ \
  nickolas/experiments/__init__.py nickolas/experiments/experiment_11_candidate_agent.py \
  USER@ubuntu-makers:/scratch/USER/reptechjam2026/

# 3. Submit (first run also installs Ollama and pulls qwen ~4.7 GB into /scratch)
ssh USER@ubuntu-makers 'mkdir -p /scratch/$USER/logs && sbatch /scratch/$USER/reptechjam2026/winston/lab/bench/server/run_resolver.sbatch'

# 4. Watch
ssh USER@ubuntu-makers 'squeue -u $USER; tail -n 5 /scratch/$USER/logs/*.out'

# 5. Pull results back, then report locally
rsync -avz USER@ubuntu-makers:/scratch/USER/reptechjam2026/winston/lab/bench/{results.jsonl,parses.jsonl} winston/lab/bench/
cd winston/lab/bench && python3 report.py
```

Expected on a 4090: ~2-4 s per parse, so ~1,500 remaining parses in 1-2 h; the
job asks for 3 h. Re-submitting resumes.

Alternative if you'd rather keep the scripts on the laptop: run the same job with
`OLLAMA_HOST=0.0.0.0:11434` for `serve` and no `score.py`, then locally
`OLLAMA_HOST=ubuntu-makers:11434 python3 score.py`. Only works if the server's
port 11434 is reachable over Tailscale; the batch route above needs no open port.
