#!/bin/bash

#SBATCH --job-name=cross_eval
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=6
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
# Example (STL-10 QIT student probed on CIFAR-10):
#   BACKBONE=vit_b_16 SOURCE_DATASET=stl10 TARGET_DATASET=cifar10 \
#   sbatch scripts/run_slurm_cross_dataset_eval.sbatch
#
# Example (CNN with BN recalibration on the target domain):
#   BACKBONE=resnet18 SOURCE_DATASET=stl10 TARGET_DATASET=cifar10 RECALIBRATE_BN=1 \
#   OUTPUT_DIR=runs/cross_stl10_to_cifar10_resnet18_bnrecal \
#   sbatch scripts/run_slurm_cross_dataset_eval.sbatch
#
# ─────────────────────────────────────────────────────────────────────────────

BACKBONE="${BACKBONE:-resnet18}"
SOURCE_DATASET="${SOURCE_DATASET:-stl10}"
TARGET_DATASET="${TARGET_DATASET:-cifar10}"
CHECKPOINT="${CHECKPOINT:-}"   # empty = runs/qit_<source_dataset>_<backbone>/checkpoint_best.pt
DATA_DIR="${DATA_DIR:-}"
WEIGHT_GRAN="${WEIGHT_GRAN:-per_tensor}"
CONFIGS="${CONFIGS:-}"         # space-separated subset, e.g. "fp w8a8 w4a4"; empty = all
RECALIBRATE_BN="${RECALIBRATE_BN:-}"   # set to 1 to add --recalibrate_bn
BATCH_SIZE="${BATCH_SIZE:-16}"
N_PROBE_TRAIN="${N_PROBE_TRAIN:--1}"
N_PROBE_TEST="${N_PROBE_TEST:--1}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/cross_${SOURCE_DATASET}_to_${TARGET_DATASET}_${BACKBONE}}"

PROJECT_DIR="${SLURM_SUBMIT_DIR}"
VENV="${PROJECT_DIR}/venv"

echo "=== cross_dataset_eval  job=${SLURM_JOB_ID}  node=$(hostname) ==="
echo "backbone=${BACKBONE}  source=${SOURCE_DATASET}  target=${TARGET_DATASET}"
echo "checkpoint=${CHECKPOINT:-<default>}  weight_gran=${WEIGHT_GRAN}  configs=${CONFIGS:-all}  recalibrate_bn=${RECALIBRATE_BN:-0}"
date

source "${VENV}/bin/activate"
python --version
python -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available())"

cd "${PROJECT_DIR}"

CHECKPOINT_FLAG=""
if [ -n "${CHECKPOINT}" ]; then
    CHECKPOINT_FLAG="--checkpoint ${CHECKPOINT}"
fi

DATA_DIR_FLAG=""
if [ -n "${DATA_DIR}" ]; then
    DATA_DIR_FLAG="--data_dir ${DATA_DIR}"
fi

CONFIGS_FLAG=""
if [ -n "${CONFIGS}" ]; then
    CONFIGS_FLAG="--configs ${CONFIGS}"
fi

python scripts/run_cross_dataset_eval.py \
    --backbone            "${BACKBONE}"        \
    --source_dataset      "${SOURCE_DATASET}"  \
    --target_dataset      "${TARGET_DATASET}"  \
    --weight_granularity  "${WEIGHT_GRAN}"     \
    --batch_size          "${BATCH_SIZE}"      \
    --n_probe_train       "${N_PROBE_TRAIN}"   \
    --n_probe_test        "${N_PROBE_TEST}"    \
    --output_dir          "${OUTPUT_DIR}"      \
    ${CHECKPOINT_FLAG}                         \
    ${DATA_DIR_FLAG}                           \
    ${CONFIGS_FLAG}                            \
    ${RECALIBRATE_BN:+--recalibrate_bn}

echo "=== done ===" && date
