#!/bin/bash

#SBATCH --job-name=diag_merlin
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=80G
#SBATCH --cpus-per-task=6
#SBATCH --time=01:00:00
#SBATCH --exclude=hendrixgpu05fl,hendrixgpu06fl,hendrixgpu09fl,hendrixgpu10fl,hendrixgpu11fl,hendrixgpu12fl,hendrixgpu13fl,hendrixgpu17fl,hendrixgpu18fl,hendrixgpu19fl,hendrixgpu20fl

echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -n 1)"
echo "=== diag_merlin  job=${SLURM_JOB_ID}  node=$(hostname) ==="
date

PROJECT_DIR="${SLURM_SUBMIT_DIR}"
source "${PROJECT_DIR}/venv/bin/activate"
python --version

TARGET_SHAPE="${TARGET_SHAPE:-}"

cd "${PROJECT_DIR}"
python -u scripts/diag_merlin_memory.py ${TARGET_SHAPE:+--target_shape $TARGET_SHAPE}

echo "=== done ===" && date
