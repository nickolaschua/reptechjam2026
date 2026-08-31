#!/bin/bash
# On ubuntu-makers, from /scratch/$USER/winston/lab/lora:   sbatch lora.sbatch
# Conventions from compute_server_guide.md: module load, venv + logs under /scratch/$USER.
#SBATCH --job-name=lora-parser
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/%u/logs/%j.out
#SBATCH --error=/scratch/%u/logs/%j.err
set -euo pipefail
mkdir -p /scratch/$USER/logs /scratch/$USER/hf
export HF_HOME=/scratch/$USER/hf            # model weights off the home quota
module load cuda/13.1 || true
VENV=/scratch/$USER/lora-venv
[ -x $VENV/bin/python ] || python3 -m venv $VENV
source $VENV/bin/activate
pip install -q --upgrade pip
pip install -q torch transformers peft accelerate pyyaml
cd "$(dirname "$0")"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python build_sft.py
echo "== zero-shot floor (no adapter) =="
python eval_lora.py  --model Qwen/Qwen2.5-1.5B-Instruct --limit 30
echo "== train =="
python train_lora.py --model Qwen/Qwen2.5-1.5B-Instruct --out adapter-1.5b
echo "== fine-tuned =="
python eval_lora.py  --model Qwen/Qwen2.5-1.5B-Instruct --adapter adapter-1.5b
