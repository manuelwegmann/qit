#!/bin/bash

#SBATCH --job-name=ptq_text
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=6
#SBATCH --time=04:00:00
#SBATCH --exclude=hendrixgpu05fl,hendrixgpu06fl,hendrixgpu09fl,hendrixgpu10fl,hendrixgpu11fl,hendrixgpu12fl,hendrixgpu13fl,hendrixgpu17fl,hendrixgpu18fl,hendrixgpu19fl,hendrixgpu20fl

GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1)
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)  (${GPU_MEM} MB)"
if [ "${GPU_MEM}" -lt 38000 ]; then
    echo "ERROR: GPU has only ${GPU_MEM} MB — need A100 (40 GB). Exiting."
    exit 1
fi

# ── Configurable via env vars ──────────────────────────────────────────────────
#
# Example:
#   MODEL_A=runs/qit_sst2_distilbert-base-uncased/checkpoint_best.pt \
#   MODEL_A_LABEL="QIT student" \
#   MODEL_B=pretrained \
#   MODEL_B_LABEL="Teacher" \
#   sbatch scripts/run_slurm_ptq_text.sbatch
#
# ──────────────────────────────────────────────────────────────────────────────

MODEL="${MODEL:-distilbert-base-uncased}"
DATASET="${DATASET:-sst2}"
MODEL_A="${MODEL_A:-pretrained}"
MODEL_A_LABEL="${MODEL_A_LABEL:-model_a}"
MODEL_B="${MODEL_B:-pretrained}"
MODEL_B_LABEL="${MODEL_B_LABEL:-model_b}"
N_CALIB_BATCHES="${N_CALIB_BATCHES:-32}"
PERCENTILE="${PERCENTILE:-99.9}"
CALIB_BATCH_SIZE="${CALIB_BATCH_SIZE:-32}"
WEIGHT_GRAN="${WEIGHT_GRAN:-per_channel}"
CONFIGS="${CONFIGS:-}"   # space-separated subset, e.g. "fp w8a8 w4a4"; empty = all
OUTPUT_DIR="${OUTPUT_DIR:-runs/ptq_comparison_${DATASET}_$(echo ${MODEL} | tr '/' '_')}"

PROJECT_DIR="${SLURM_SUBMIT_DIR}"
VENV="${PROJECT_DIR}/venv"

echo "=== ptq_text  job=${SLURM_JOB_ID}  node=$(hostname) ==="
echo "model=${MODEL}  dataset=${DATASET}"
echo "model_a=[${MODEL_A_LABEL}] ${MODEL_A}"
echo "model_b=[${MODEL_B_LABEL}] ${MODEL_B}"
echo "n_calib_batches=${N_CALIB_BATCHES}  percentile=${PERCENTILE}  weight_gran=${WEIGHT_GRAN}  configs=${CONFIGS:-all}"
date

export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

source "${VENV}/bin/activate"
python --version
python -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available())"

cd "${PROJECT_DIR}"

CONFIGS_FLAG=""
if [ -n "${CONFIGS}" ]; then
    CONFIGS_FLAG="--configs ${CONFIGS}"
fi

python scripts/run_ptq_text.py \
    --model               "${MODEL}"             \
    --dataset             "${DATASET}"           \
    --model_a             "${MODEL_A}"           \
    --model_a_label       "${MODEL_A_LABEL}"     \
    --model_b             "${MODEL_B}"           \
    --model_b_label       "${MODEL_B_LABEL}"     \
    --n_calib_batches     "${N_CALIB_BATCHES}"   \
    --percentile          "${PERCENTILE}"        \
    --calib_batch_size    "${CALIB_BATCH_SIZE}"  \
    --weight_granularity  "${WEIGHT_GRAN}"       \
    --output_dir          "${OUTPUT_DIR}"        \
    ${CONFIGS_FLAG}

echo "=== done ===" && date
