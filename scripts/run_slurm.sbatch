#!/bin/bash

#SBATCH --job-name=qft_qit_cifar
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=10
#SBATCH --time=04:00:00
#SBATCH --exclude=hendrixgpu05fl,hendrixgpu06fl,hendrixgpu09fl,hendrixgpu10fl,hendrixgpu11fl,hendrixgpu12fl,hendrixgpu13fl,hendrixgpu17fl,hendrixgpu18fl,hendrixgpu19fl,hendrixgpu20fl

GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1)
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)  (${GPU_MEM} MB)"
if [ "${GPU_MEM}" -lt 38000 ]; then
    echo "ERROR: GPU has only ${GPU_MEM} MB — need A100 (40 GB). Exiting."
    exit 1
fi

BACKBONE="${BACKBONE:-resnet18}"
N_TRAIN="${N_TRAIN:-7000}"
EPOCHS="${EPOCHS:-50}"
PATIENCE="${PATIENCE:-10}"
VAL_WINDOW="${VAL_WINDOW:-3}"
LR="${LR:-1e-4}"
LR_SCHEDULE="${LR_SCHEDULE:-cosine}"
LOSS="${LOSS:-cosine}"
WEIGHT_GRAN="${WEIGHT_GRAN:-per_channel}"
FP_LOSS_WEIGHT="${FP_LOSS_WEIGHT:-0.0}"
REG_WEIGHT="${REG_WEIGHT:-0.0}"
BATCH_SIZE="${BATCH_SIZE:-16}"
W_BITS_MIN="${W_BITS_MIN:-2}"
W_BITS_MAX="${W_BITS_MAX:-8}"
A_BITS_MIN="${A_BITS_MIN:-4}"
A_BITS_MAX="${A_BITS_MAX:-8}"
NUM_WORKERS="${NUM_WORKERS:-8}"
DATASET="${DATASET:-stl10}"
DATA_DIR="${DATA_DIR:-}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/qit_${DATASET}_${BACKBONE}}"

PROJECT_DIR="${SLURM_SUBMIT_DIR}"
VENV="${PROJECT_DIR}/venv"

echo "=== qft_qit_cifar  job=${SLURM_JOB_ID}  node=$(hostname) ==="
echo "backbone=${BACKBONE}  dataset=${DATASET}  n_train=${N_TRAIN}  epochs=${EPOCHS}  patience=${PATIENCE}  lr=${LR} (${LR_SCHEDULE})  loss=${LOSS}  weight_gran=${WEIGHT_GRAN}"
echo "fp_loss_weight=${FP_LOSS_WEIGHT}  reg_weight=${REG_WEIGHT}  bs=${BATCH_SIZE}  w=[${W_BITS_MIN},${W_BITS_MAX}]  a=[${A_BITS_MIN},${A_BITS_MAX}]  val_window=${VAL_WINDOW}"
date

source "${VENV}/bin/activate"
python --version
python -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available())"

cd "${PROJECT_DIR}"

# Build optional data_dir flag
DATA_DIR_FLAG=""
if [ -n "${DATA_DIR}" ]; then
    DATA_DIR_FLAG="--data_dir ${DATA_DIR}"
fi

python scripts/run_qit.py \
    --backbone            "${BACKBONE}"        \
    --dataset             "${DATASET}"         \
    --n_train             "${N_TRAIN}"         \
    --epochs              "${EPOCHS}"          \
    --patience            "${PATIENCE}"        \
    --val_window          "${VAL_WINDOW}"      \
    --lr                  "${LR}"              \
    --lr_schedule         "${LR_SCHEDULE}"     \
    --loss                "${LOSS}"            \
    --weight_granularity  "${WEIGHT_GRAN}"     \
    --fp_loss_weight      "${FP_LOSS_WEIGHT}"  \
    --reg_weight          "${REG_WEIGHT}"      \
    --batch_size          "${BATCH_SIZE}"      \
    --w_bits_min          "${W_BITS_MIN}"      \
    --w_bits_max          "${W_BITS_MAX}"      \
    --a_bits_min          "${A_BITS_MIN}"      \
    --a_bits_max          "${A_BITS_MAX}"      \
    --num_workers         "${NUM_WORKERS}"     \
    --output_dir          "${OUTPUT_DIR}"      \
    ${DATA_DIR_FLAG}


echo "=== comparison plot ===" && date
python scripts/plot_qit_comparison.py --results "${OUTPUT_DIR}/results.json"

echo "=== done ===" && date
