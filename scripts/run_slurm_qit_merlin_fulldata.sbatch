#!/bin/bash

#SBATCH --job-name=qit_merlin_fd
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=80G
#SBATCH --cpus-per-task=14
#SBATCH --time=24:00:00
#SBATCH --exclude=hendrixgpu05fl,hendrixgpu06fl,hendrixgpu09fl,hendrixgpu10fl,hendrixgpu11fl,hendrixgpu12fl,hendrixgpu13fl,hendrixgpu17fl,hendrixgpu18fl,hendrixgpu19fl,hendrixgpu20fl

GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1)
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)  (${GPU_MEM} MB)"

# ── Configurable via env vars ─────────────────────────────────────────────────
#
# Full-data QIT run (combined train+val, no validation / early stopping):
#   sbatch scripts/run_slurm_qit_merlin_fulldata.sbatch
#
# This calls run_qit_merlin.py --full_data. The flags below reproduce the old
# standalone fulldata recipe (bf16, grad_accum=4, 160x224x224 resize).
# Probing is no longer done here — run cache_merlin_features.py + run_probe_merlin.py
# on the resulting checkpoint afterwards.
#
# Wider bit-width sampling + more epochs (override --time/--job-name via sbatch flags,
# since #SBATCH directives can't be set from env vars):
#   EPOCHS=10 W_BITS_MIN=2 A_BITS_MIN=2 OUTPUT_DIR=runs/qit_merlin_fulldata_w2a8 \
#     sbatch --time=48:00:00 --job-name=qit_merlin_fd_w2a8 scripts/run_slurm_qit_merlin_fulldata.sbatch
#
# ─────────────────────────────────────────────────────────────────────────────

DATA_DIR="${DATA_DIR:-${SLURM_SUBMIT_DIR}/../CT-CLIP/data/merlin_data}"
REPORTS="${REPORTS:-${SLURM_SUBMIT_DIR}/../CT-CLIP/data/reports_final.xlsx}"
LABELS="${LABELS:-${SLURM_SUBMIT_DIR}/../CT-CLIP/data/zero_shot_findings_disease_cls.csv}"
METADATA="${METADATA:-${SLURM_SUBMIT_DIR}/../CT-CLIP/data/metadata.csv}"
CONDITIONS="${CONDITIONS:-atelectasis pleural_effusion renal_cyst}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/qit_merlin_fulldata}"
EPOCHS="${EPOCHS:-5}"
W_BITS_MIN="${W_BITS_MIN:-4}"
W_BITS_MAX="${W_BITS_MAX:-8}"
A_BITS_MIN="${A_BITS_MIN:-4}"
A_BITS_MAX="${A_BITS_MAX:-8}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
NUM_WORKERS="${NUM_WORKERS:-12}"
AMP_DTYPE="${AMP_DTYPE:-bf16}"
TARGET_SHAPE="${TARGET_SHAPE:-160 224 224}"

PROJECT_DIR="${SLURM_SUBMIT_DIR}"
VENV="${PROJECT_DIR}/venv"

echo "=== qit_merlin_fulldata  job=${SLURM_JOB_ID}  node=$(hostname) ==="
echo "output=${OUTPUT_DIR}"
echo "epochs=${EPOCHS}  w_bits=${W_BITS_MIN}-${W_BITS_MAX}  a_bits=${A_BITS_MIN}-${A_BITS_MAX}"
date

source "${VENV}/bin/activate"
python --version
python -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available())"

cd "${PROJECT_DIR}"

python -u scripts/run_qit_merlin.py \
    --full_data                    \
    --data_dir   "${DATA_DIR}"     \
    --reports    "${REPORTS}"      \
    --labels     "${LABELS}"       \
    --metadata   "${METADATA}"     \
    --conditions ${CONDITIONS}     \
    --output_dir "${OUTPUT_DIR}"   \
    --epochs     "${EPOCHS}"       \
    --w_bits_min "${W_BITS_MIN}"   \
    --w_bits_max "${W_BITS_MAX}"   \
    --a_bits_min "${A_BITS_MIN}"   \
    --a_bits_max "${A_BITS_MAX}"   \
    --grad_accum "${GRAD_ACCUM}"   \
    --num_workers "${NUM_WORKERS}" \
    --amp_dtype  "${AMP_DTYPE}"    \
    --target_shape ${TARGET_SHAPE}

echo "=== done ===" && date
