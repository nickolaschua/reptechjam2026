#!/bin/bash
# Ship winston/ to the server (no commit to the shared branch needed) and submit.
#   bash ship.sh USERNAME
set -euo pipefail
U=${1:?usage: bash ship.sh USERNAME}
W="$(cd "$(dirname "$0")/../.." && pwd)"
rsync -az --exclude '.cache' --exclude '__pycache__' --exclude 'experiments/.cache' \
      --exclude 'lab/bench/.cache' "$W/" "$U@ubuntu-makers:/scratch/$U/winston/"
ssh "$U@ubuntu-makers" "cd /scratch/$U/winston/lab/lora && mkdir -p /scratch/$U/logs && sbatch lora.sbatch && squeue -u $U"
echo "watch:  ssh $U@ubuntu-makers tail -f /scratch/$U/logs/<jobid>.out"
