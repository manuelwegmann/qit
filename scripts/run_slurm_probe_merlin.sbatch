#!/bin/bash

#SBATCH --job-name=probe_merlin
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=40G
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --exclude=hendrixgpu05fl,hendrixgpu06fl,hendrixgpu09fl,hendrixgpu10fl,hendrixgpu11fl,hendrixgpu12fl,hendrixgpu13fl,hendrixgpu17fl,hendrixgpu18fl,hendrixgpu19fl,hendrixgpu20fl

GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1)
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)  (${GPU_MEM} MB)"

# Set FEAT_DIR to the feature directory to probe.
# Examples:
#   FEAT_DIR=runs/qit_merlin_fulldata/features sbatch scripts/run_slurm_probe_merlin.sbatch
#   FEAT_DIR=runs/teacher_features             sbatch scripts/run_slurm_probe_merlin.sbatch
FEAT_DIR="${FEAT_DIR:-runs/qit_merlin_fulldata/features}"

CONFIGS="${CONFIGS:-fp w8a8 w4a8 w4a4}"
MIN_TEST_PER_CLASS="${MIN_TEST_PER_CLASS:-20}"
PROBE_EPOCHS="${PROBE_EPOCHS:-200}"
PROBE_SEEDS="${PROBE_SEEDS:-5}"
PROBE_LR="${PROBE_LR:-1e-3}"
WEIGHT_DECAYS="${WEIGHT_DECAYS:-0.0 1e-4 1e-3 1e-2 1e-1}"
OUTPUT_DIR="${OUTPUT_DIR:-${FEAT_DIR}/probe_results}"

PROJECT_DIR="${SLURM_SUBMIT_DIR}"
VENV="${PROJECT_DIR}/venv"

export TMPDIR="${PROJECT_DIR}/.tmp_${SLURM_JOB_ID}"
mkdir -p "${TMPDIR}"
trap "rm -rf '${TMPDIR}'" EXIT

echo "=== probe_merlin  job=${SLURM_JOB_ID}  node=$(hostname) ==="
echo "feat_dir=${FEAT_DIR}"
echo "configs=${CONFIGS}"
echo "probe: epochs=${PROBE_EPOCHS}  seeds=${PROBE_SEEDS}  lr=${PROBE_LR}  weight_decays=${WEIGHT_DECAYS}"
echo "output=${OUTPUT_DIR}"
date

source "${VENV}/bin/activate"
python --version
python -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available())"

cd "${PROJECT_DIR}"

python -u scripts/run_probe_merlin.py \
    --feat_dir           "${FEAT_DIR}"            \
    --configs            ${CONFIGS}               \
    --min_test_per_class "${MIN_TEST_PER_CLASS}"  \
    --probe_epochs       "${PROBE_EPOCHS}"        \
    --probe_seeds        "${PROBE_SEEDS}"         \
    --probe_lr           "${PROBE_LR}"            \
    --weight_decays      ${WEIGHT_DECAYS}         \
    --output_dir         "${OUTPUT_DIR}"

echo "=== done ===" && date
