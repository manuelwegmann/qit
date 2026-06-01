#!/bin/bash

#SBATCH --job-name=qit_merlin
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=80G
#SBATCH --cpus-per-task=10
#SBATCH --time=24:00:00
#SBATCH --exclude=hendrixgpu05fl,hendrixgpu06fl,hendrixgpu09fl,hendrixgpu10fl,hendrixgpu11fl,hendrixgpu12fl,hendrixgpu13fl,hendrixgpu17fl,hendrixgpu18fl,hendrixgpu19fl,hendrixgpu20fl

GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1)
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)  (${GPU_MEM} MB)"

# ── Configurable via env vars ─────────────────────────────────────────────────
#
# Default run (all data, W2-8 A4-8, 20 epochs):
#   sbatch scripts/run_slurm_qit_merlin.sbatch
#
# Longer run with gradient accumulation (effective batch = 4):
#   EPOCHS=30 GRAD_ACCUM=4 OUTPUT_DIR=runs/qit_merlin_long sbatch scripts/run_slurm_qit_merlin.sbatch
#
# ─────────────────────────────────────────────────────────────────────────────

DATA_DIR="${DATA_DIR:-${SLURM_SUBMIT_DIR}/../CT-CLIP/data/merlin_data}"
REPORTS="${REPORTS:-${SLURM_SUBMIT_DIR}/../CT-CLIP/data/reports_final.xlsx}"
LABELS="${LABELS:-${SLURM_SUBMIT_DIR}/../CT-CLIP/data/zero_shot_findings_disease_cls.csv}"
METADATA="${METADATA:-${SLURM_SUBMIT_DIR}/../CT-CLIP/data/metadata.csv}"

EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
LR="${LR:-1e-5}"
W_BITS_MIN="${W_BITS_MIN:-2}"
W_BITS_MAX="${W_BITS_MAX:-8}"
A_BITS_MIN="${A_BITS_MIN:-4}"
A_BITS_MAX="${A_BITS_MAX:-8}"
VAL_EVERY="${VAL_EVERY:-1}"
NUM_WORKERS="${NUM_WORKERS:-8}"
AMP_DTYPE="${AMP_DTYPE:-fp16}"
CONDITIONS="${CONDITIONS:-atelectasis pleural_effusion renal_cyst}"
N_TRAIN="${N_TRAIN:-2000}"
N_VAL="${N_VAL:-500}"
PATIENCE="${PATIENCE:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/qit_merlin_w${W_BITS_MIN}-${W_BITS_MAX}a${A_BITS_MIN}-${A_BITS_MAX}}"

PROJECT_DIR="${SLURM_SUBMIT_DIR}"
VENV="${PROJECT_DIR}/venv"

echo "=== qit_merlin  job=${SLURM_JOB_ID}  node=$(hostname) ==="
echo "data_dir=${DATA_DIR}"
echo "epochs=${EPOCHS}  batch_size=${BATCH_SIZE}  grad_accum=${GRAD_ACCUM}  lr=${LR}  amp=${AMP_DTYPE}"
echo "w_bits=${W_BITS_MIN}-${W_BITS_MAX}  a_bits=${A_BITS_MIN}-${A_BITS_MAX}"
echo "output=${OUTPUT_DIR}"
date

source "${VENV}/bin/activate"
python --version
python -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available())"

cd "${PROJECT_DIR}"

python scripts/run_qit_merlin.py \
    --data_dir    "${DATA_DIR}"    \
    --reports     "${REPORTS}"     \
    --labels      "${LABELS}"      \
    --metadata    "${METADATA}"    \
    --epochs      "${EPOCHS}"      \
    --batch_size  "${BATCH_SIZE}"  \
    --grad_accum  "${GRAD_ACCUM}"  \
    --lr          "${LR}"          \
    --w_bits_min  "${W_BITS_MIN}"  \
    --w_bits_max  "${W_BITS_MAX}"  \
    --a_bits_min  "${A_BITS_MIN}"  \
    --a_bits_max  "${A_BITS_MAX}"  \
    --val_every   "${VAL_EVERY}"   \
    --num_workers "${NUM_WORKERS}" \
    --amp_dtype   "${AMP_DTYPE}"   \
    --conditions  ${CONDITIONS}    \
    --n_train     "${N_TRAIN}"     \
    --n_val       "${N_VAL}"       \
    --patience    "${PATIENCE}"    \
    --output_dir  "${OUTPUT_DIR}"

echo "=== done ===" && date
