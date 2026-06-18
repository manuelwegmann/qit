#!/bin/bash

#SBATCH --job-name=qat_merlin
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=80G
#SBATCH --cpus-per-task=14
#SBATCH --time=16:00:00
#SBATCH --exclude=hendrixgpu05fl,hendrixgpu06fl,hendrixgpu09fl,hendrixgpu10fl,hendrixgpu11fl,hendrixgpu12fl,hendrixgpu13fl,hendrixgpu17fl,hendrixgpu18fl,hendrixgpu19fl,hendrixgpu20fl

GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1)
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)  (${GPU_MEM} MB)"

# ── Configurable via env vars ─────────────────────────────────────────────────
#
# Pretrained-init run:
#   CONDITION=renal_cyst INIT=pretrained OUTPUT_DIR=runs/qat_merlin_renal_cyst_pretrained_w4a4 \
#     sbatch scripts/run_slurm_qat_merlin.sbatch
#
# QIT-init run:
#   CONDITION=renal_cyst INIT=runs/qit_merlin_fulldata/checkpoint_final.pt \
#     OUTPUT_DIR=runs/qat_merlin_renal_cyst_qit_w4a4 \
#     sbatch scripts/run_slurm_qat_merlin.sbatch
#
# ─────────────────────────────────────────────────────────────────────────────

DATA_DIR="${DATA_DIR:-${SLURM_SUBMIT_DIR}/../CT-CLIP/data/merlin_data}"
REPORTS="${REPORTS:-${SLURM_SUBMIT_DIR}/../CT-CLIP/data/reports_final.xlsx}"
LABELS="${LABELS:-${SLURM_SUBMIT_DIR}/../CT-CLIP/data/zero_shot_findings_disease_cls.csv}"
METADATA="${METADATA:-${SLURM_SUBMIT_DIR}/../CT-CLIP/data/metadata.csv}"
CONDITION="${CONDITION:-renal_cyst}"
INIT="${INIT:-pretrained}"
W_BITS="${W_BITS:-4}"
A_BITS="${A_BITS:-4}"
EPOCHS="${EPOCHS:-30}"
PATIENCE="${PATIENCE:-0}"
LR="${LR:-1e-5}"
LR_SCHEDULE="${LR_SCHEDULE:-cosine}"
BATCH_SIZE="${BATCH_SIZE:-16}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
WEIGHT_GRAN="${WEIGHT_GRAN:-per_channel}"
NUM_WORKERS="${NUM_WORKERS:-14}"
OUTPUT_DIR="${OUTPUT_DIR:-}"

PROJECT_DIR="${SLURM_SUBMIT_DIR}"
VENV="${PROJECT_DIR}/venv"

echo "=== qat_merlin  job=${SLURM_JOB_ID}  node=$(hostname) ==="
echo "condition=${CONDITION}  init=${INIT}"
echo "w_bits=${W_BITS}  a_bits=${A_BITS}  epochs=${EPOCHS}  patience=${PATIENCE}  lr=${LR} (${LR_SCHEDULE})"
echo "batch_size=${BATCH_SIZE}  grad_accum=${GRAD_ACCUM}  weight_gran=${WEIGHT_GRAN}"
date

source "${VENV}/bin/activate"
python --version
python -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available())"

cd "${PROJECT_DIR}"

OUTPUT_DIR_FLAG=""
if [ -n "${OUTPUT_DIR}" ]; then
    OUTPUT_DIR_FLAG="--output_dir ${OUTPUT_DIR}"
fi

python -u scripts/run_qat_merlin.py \
    --data_dir           "${DATA_DIR}"    \
    --reports            "${REPORTS}"     \
    --labels             "${LABELS}"      \
    --metadata           "${METADATA}"    \
    --condition          "${CONDITION}"   \
    --init               "${INIT}"        \
    --w_bits             "${W_BITS}"      \
    --a_bits             "${A_BITS}"      \
    --epochs             "${EPOCHS}"      \
    --patience           "${PATIENCE}"    \
    --lr                 "${LR}"          \
    --lr_schedule        "${LR_SCHEDULE}" \
    --batch_size         "${BATCH_SIZE}"  \
    --grad_accum         "${GRAD_ACCUM}"  \
    --weight_granularity "${WEIGHT_GRAN}" \
    --num_workers        "${NUM_WORKERS}" \
    ${OUTPUT_DIR_FLAG}

echo "=== done ===" && date
