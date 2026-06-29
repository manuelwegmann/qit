#!/bin/bash

#SBATCH --job-name=cache_teacher_feats
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=80G
#SBATCH --cpus-per-task=14
#SBATCH --time=48:00:00
#SBATCH --exclude=hendrixgpu05fl,hendrixgpu06fl,hendrixgpu09fl,hendrixgpu10fl,hendrixgpu11fl,hendrixgpu12fl,hendrixgpu13fl,hendrixgpu17fl,hendrixgpu18fl,hendrixgpu19fl,hendrixgpu20fl

GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1)
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)  (${GPU_MEM} MB)"

DATA_DIR="${DATA_DIR:-${SLURM_SUBMIT_DIR}/../CT-CLIP/data/merlin_data}"
REPORTS="${REPORTS:-${SLURM_SUBMIT_DIR}/../CT-CLIP/data/reports_final.xlsx}"
LABELS="${LABELS:-${SLURM_SUBMIT_DIR}/../CT-CLIP/data/zero_shot_findings_disease_cls.csv}"
METADATA="${METADATA:-${SLURM_SUBMIT_DIR}/../CT-CLIP/data/metadata.csv}"

OUTPUT_DIR="${OUTPUT_DIR:-runs/teacher_features}"
CONFIGS="${CONFIGS:-fp w8a8 w6a6 w5a5 w4a8 w4a4 w3a6 w2a8 w2a4}"
SPLITS="${SPLITS:-train val test}"
PERCENTILE="${PERCENTILE:-99.99}"
N_CALIB_BATCHES="${N_CALIB_BATCHES:-32}"
CALIB_BATCH_SIZE="${CALIB_BATCH_SIZE:-16}"
WEIGHT_GRAN="${WEIGHT_GRAN:-per_channel}"
TARGET_SHAPE="${TARGET_SHAPE:-160 224 224}"
BATCH_SIZE="${BATCH_SIZE:-16}"
AMP_DTYPE="${AMP_DTYPE:-bf16}"

PROJECT_DIR="${SLURM_SUBMIT_DIR}"
VENV="${PROJECT_DIR}/venv"

export TMPDIR="${PROJECT_DIR}/.tmp_${SLURM_JOB_ID}"
mkdir -p "${TMPDIR}"
trap "rm -rf '${TMPDIR}'" EXIT

echo "=== cache_teacher_feats  job=${SLURM_JOB_ID}  node=$(hostname) ==="
echo "output_dir=${OUTPUT_DIR}"
echo "configs=${CONFIGS}  splits=${SPLITS}"
date

source "${VENV}/bin/activate"
python --version
python -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available())"

cd "${PROJECT_DIR}"

python -u scripts/cache_merlin_features.py \
    --model              teacher             \
    --output_dir         "${OUTPUT_DIR}"     \
    --data_dir           "${DATA_DIR}"       \
    --reports            "${REPORTS}"        \
    --labels             "${LABELS}"         \
    --metadata           "${METADATA}"       \
    --configs            ${CONFIGS}          \
    --splits             ${SPLITS}           \
    --percentile         "${PERCENTILE}"     \
    --n_calib_batches    "${N_CALIB_BATCHES}"\
    --calib_batch_size   "${CALIB_BATCH_SIZE}"\
    --weight_granularity "${WEIGHT_GRAN}"    \
    --target_shape       ${TARGET_SHAPE}     \
    --batch_size         "${BATCH_SIZE}"     \
    --amp_dtype          "${AMP_DTYPE}"

echo "=== done ===" && date
