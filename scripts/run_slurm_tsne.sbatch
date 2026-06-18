#!/bin/bash

#SBATCH --job-name=tsne
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --exclude=hendrixgpu05fl,hendrixgpu06fl,hendrixgpu09fl,hendrixgpu10fl,hendrixgpu11fl,hendrixgpu12fl,hendrixgpu13fl,hendrixgpu17fl,hendrixgpu18fl,hendrixgpu19fl,hendrixgpu20fl

GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1)
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)  (${GPU_MEM} MB)"

# ── Configurable via env vars ─────────────────────────────────────────────────
#
# Example:
#   RUN_DIR=runs/qit_cifar10_vit_b_16 N_PER_CLASS=200 sbatch scripts/run_slurm_tsne.sbatch
#
# ─────────────────────────────────────────────────────────────────────────────

RUN_DIR="${RUN_DIR:-runs/qit_cifar10_vit_b_16}"
CONFIGS="${CONFIGS:-fp w4a4}"
N_PER_CLASS="${N_PER_CLASS:-200}"
METHOD="${METHOD:-tsne}"

PROJECT_DIR="${SLURM_SUBMIT_DIR}"
VENV="${PROJECT_DIR}/venv"

echo "=== ${METHOD}_comparison  job=${SLURM_JOB_ID}  node=$(hostname) ==="
echo "run_dir=${RUN_DIR}  configs=${CONFIGS}  n_per_class=${N_PER_CLASS}  method=${METHOD}"
date

source "${VENV}/bin/activate"
cd "${PROJECT_DIR}"

python scripts/plot_tsne_comparison.py "${RUN_DIR}" \
    --configs ${CONFIGS} \
    --n_per_class "${N_PER_CLASS}" \
    --method "${METHOD}"

echo "=== done ===" && date
