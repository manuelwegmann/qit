#!/bin/bash

#SBATCH --job-name=ptq_cmp_merlin
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
# Default: both models x w8a8/w4a8/w4a4 x pct only (6 extraction passes,
# ~1h46m/pass on A100-40GB per the dynamic-mode eval => ~11h total —
# comfortably fits in one 24h job).
# results.json is written incrementally and finished combos are skipped on rerun.
#
# Examples:
#   sbatch scripts/run_slurm_ptq_comparison_merlin.sbatch
#
#   # Split by model if needed:
#   MODELS=student sbatch scripts/run_slurm_ptq_comparison_merlin.sbatch
#   MODELS=teacher sbatch scripts/run_slurm_ptq_comparison_merlin.sbatch
#
#   # Add minmax calibration too:
#   MODES="minmax pct" sbatch scripts/run_slurm_ptq_comparison_merlin.sbatch
#
# ─────────────────────────────────────────────────────────────────────────────

DATA_DIR="${DATA_DIR:-${SLURM_SUBMIT_DIR}/../CT-CLIP/data/merlin_data}"
REPORTS="${REPORTS:-${SLURM_SUBMIT_DIR}/../CT-CLIP/data/reports_final.xlsx}"
LABELS="${LABELS:-${SLURM_SUBMIT_DIR}/../CT-CLIP/data/zero_shot_findings_disease_cls.csv}"
METADATA="${METADATA:-${SLURM_SUBMIT_DIR}/../CT-CLIP/data/metadata.csv}"

STUDENT_CKPT="${STUDENT_CKPT:-runs/qit_merlin_unfrozen_bn/checkpoint_best.pt}"
N_TRAIN="${N_TRAIN:-2000}"
N_VAL="${N_VAL:-500}"
TARGET_SHAPE="${TARGET_SHAPE:-160 224 224}"

MODELS="${MODELS:-student teacher}"
CONFIGS="${CONFIGS:-w8a8 w4a8 w4a4}"
MODES="${MODES:-pct}"
PERCENTILE="${PERCENTILE:-99.99}"
N_CALIB_BATCHES="${N_CALIB_BATCHES:-32}"
CALIB_BATCH_SIZE="${CALIB_BATCH_SIZE:-16}"
WEIGHT_GRAN="${WEIGHT_GRAN:-per_channel}"

BATCH_SIZE="${BATCH_SIZE:-16}"
AMP_DTYPE="${AMP_DTYPE:-bf16}"
CONDITIONS="${CONDITIONS:-atelectasis pleural_effusion renal_cyst}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/ptq_comparison_merlin}"

PROJECT_DIR="${SLURM_SUBMIT_DIR}"
VENV="${PROJECT_DIR}/venv"

# /tmp on shared nodes fills up quickly with many concurrent jobs (worker IPC sockets).
export TMPDIR="${PROJECT_DIR}/.tmp_${SLURM_JOB_ID}"
mkdir -p "${TMPDIR}"
trap "rm -rf '${TMPDIR}'" EXIT

echo "=== ptq_comparison_merlin  job=${SLURM_JOB_ID}  node=$(hostname) ==="
echo "student_ckpt=${STUDENT_CKPT}"
echo "models=${MODELS}  configs=${CONFIGS}  modes=${MODES}  percentile=${PERCENTILE}"
echo "n_calib_batches=${N_CALIB_BATCHES}  calib_batch_size=${CALIB_BATCH_SIZE}  weight_gran=${WEIGHT_GRAN}"
echo "output=${OUTPUT_DIR}"
date

source "${VENV}/bin/activate"
python --version
python -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available())"

cd "${PROJECT_DIR}"

python -u scripts/run_ptq_comparison_merlin.py \
    --data_dir          "${DATA_DIR}"        \
    --reports           "${REPORTS}"         \
    --labels            "${LABELS}"          \
    --metadata          "${METADATA}"        \
    --conditions        ${CONDITIONS}        \
    --n_train           "${N_TRAIN}"         \
    --n_val             "${N_VAL}"           \
    --target_shape      ${TARGET_SHAPE}      \
    --student_ckpt      "${STUDENT_CKPT}"    \
    --models            ${MODELS}            \
    --configs           ${CONFIGS}           \
    --modes             ${MODES}             \
    --percentile        "${PERCENTILE}"      \
    --n_calib_batches   "${N_CALIB_BATCHES}" \
    --calib_batch_size  "${CALIB_BATCH_SIZE}" \
    --weight_granularity "${WEIGHT_GRAN}"    \
    --batch_size        "${BATCH_SIZE}"      \
    --amp_dtype         "${AMP_DTYPE}"       \
    --output_dir        "${OUTPUT_DIR}"

echo "=== done ===" && date
