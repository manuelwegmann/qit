#!/bin/bash

#SBATCH --job-name=qat
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=10
#SBATCH --time=02:00:00
#SBATCH --exclude=hendrixgpu05fl,hendrixgpu06fl,hendrixgpu09fl,hendrixgpu10fl,hendrixgpu11fl,hendrixgpu12fl,hendrixgpu13fl,hendrixgpu17fl,hendrixgpu18fl,hendrixgpu19fl,hendrixgpu20fl

GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1)
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)  (${GPU_MEM} MB)"
if [ "${GPU_MEM}" -lt 38000 ]; then
    echo "ERROR: GPU has only ${GPU_MEM} MB — need A100 (40 GB). Exiting."
    exit 1
fi

# ── Configurable via env vars ─────────────────────────────────────────────────
#
# Run pretrained baseline:
#   INIT=pretrained OUTPUT_DIR=runs/qat_cifar10_resnet18_pretrained_w4a4 sbatch scripts/run_slurm_qat.sbatch
#
# Run QIT-initialised:
#   INIT=runs/qit_cifar10_resnet18/checkpoint_best.pt OUTPUT_DIR=runs/qat_cifar10_resnet18_qit_w4a4 sbatch scripts/run_slurm_qat.sbatch
#
# ─────────────────────────────────────────────────────────────────────────────

BACKBONE="${BACKBONE:-resnet18}"
DATASET="${DATASET:-cifar10}"
DATA_DIR="${DATA_DIR:-}"
INIT="${INIT:-pretrained}"
W_BITS="${W_BITS:-4}"
A_BITS="${A_BITS:-4}"
EPOCHS="${EPOCHS:-30}"
PATIENCE="${PATIENCE:-0}"
LR="${LR:-1e-4}"
LR_SCHEDULE="${LR_SCHEDULE:-cosine}"
BATCH_SIZE="${BATCH_SIZE:-64}"
WEIGHT_GRAN="${WEIGHT_GRAN:-per_channel}"
NUM_WORKERS="${NUM_WORKERS:-8}"
OUTPUT_DIR="${OUTPUT_DIR:-}"

PROJECT_DIR="${SLURM_SUBMIT_DIR}"
VENV="${PROJECT_DIR}/venv"

echo "=== qat  job=${SLURM_JOB_ID}  node=$(hostname) ==="
echo "backbone=${BACKBONE}  dataset=${DATASET}  init=${INIT}"
echo "w_bits=${W_BITS}  a_bits=${A_BITS}  epochs=${EPOCHS}  patience=${PATIENCE}  lr=${LR} (${LR_SCHEDULE})"
echo "batch_size=${BATCH_SIZE}  weight_gran=${WEIGHT_GRAN}"
date

source "${VENV}/bin/activate"
python --version
python -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available())"

cd "${PROJECT_DIR}"

DATA_DIR_FLAG=""
if [ -n "${DATA_DIR}" ]; then
    DATA_DIR_FLAG="--data_dir ${DATA_DIR}"
fi
OUTPUT_DIR_FLAG=""
if [ -n "${OUTPUT_DIR}" ]; then
    OUTPUT_DIR_FLAG="--output_dir ${OUTPUT_DIR}"
fi

python scripts/run_qat.py \
    --backbone            "${BACKBONE}"    \
    --dataset             "${DATASET}"     \
    --init                "${INIT}"        \
    --w_bits              "${W_BITS}"      \
    --a_bits              "${A_BITS}"      \
    --epochs              "${EPOCHS}"      \
    --patience            "${PATIENCE}"   \
    --lr                  "${LR}"          \
    --lr_schedule         "${LR_SCHEDULE}" \
    --batch_size          "${BATCH_SIZE}"  \
    --weight_granularity  "${WEIGHT_GRAN}" \
    --num_workers         "${NUM_WORKERS}" \
    ${DATA_DIR_FLAG}                       \
    ${OUTPUT_DIR_FLAG}

echo "=== done ===" && date
