#!/bin/bash
#SBATCH --job-name=bench-resolver
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/%u/logs/%j.out
#SBATCH --error=/scratch/%u/logs/%j.err
# Resolver pass on ubuntu-makers (RTX 4090). Ollama runs inside the job, no root,
# models in /scratch. score.py is resumable: rsync results.jsonl/parses.jsonl up
# first and it continues from there. Extra args are passed to score.py.
set -euo pipefail
module load cuda/13.1 2>/dev/null || true

REPO=/scratch/$USER/reptechjam2026
OLL=/scratch/$USER/ollama
mkdir -p "$OLL/models" "/scratch/$USER/logs"

if [ ! -x "$OLL/bin/ollama" ]; then
  echo "installing ollama into $OLL"
  curl -fsSL https://ollama.com/download/ollama-linux-amd64.tgz | tar -xz -C "$OLL"
fi

export OLLAMA_MODELS="$OLL/models"
export OLLAMA_HOST=127.0.0.1:11434          # serve binds here; score.py reads the same var
"$OLL/bin/ollama" serve > "/scratch/$USER/logs/ollama-$SLURM_JOB_ID.log" 2>&1 &
OLLAMA_PID=$!
trap 'kill $OLLAMA_PID 2>/dev/null || true' EXIT
until curl -s localhost:11434/api/tags >/dev/null 2>&1; do sleep 1; done
"$OLL/bin/ollama" pull qwen2.5:7b-instruct

cd "$REPO/winston/lab/bench"
python3 --version
python3 -u score.py "$@"
echo "done: $(wc -l < results.jsonl) results, $(wc -l < parses.jsonl) parses"
