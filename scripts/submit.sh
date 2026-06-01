#!/bin/bash
# Submit a QIT job with a proper name and log file derived from BACKBONE/DATASET/MODEL.
#
# Vision:  BACKBONE=resnet18 DATASET=stl10 bash scripts/submit.sh
# Text:    MODEL=distilbert-base-uncased DATASET=sst2 bash scripts/submit.sh --text

TEXT=0
for arg in "$@"; do
    [ "$arg" = "--text" ] && TEXT=1
done

if [ "$TEXT" = "1" ]; then
    DATASET="${DATASET:-sst2}"
    MODEL="${MODEL:-distilbert-base-uncased}"
    SLUG=$(echo "${MODEL}" | tr '/' '_')
    NAME="qit_${DATASET}_${SLUG}"
    SCRIPT="scripts/run_slurm_text.sbatch"
else
    DATASET="${DATASET:-stl10}"
    BACKBONE="${BACKBONE:-resnet18}"
    NAME="qit_${DATASET}_${BACKBONE}"
    SCRIPT="scripts/run_slurm.sbatch"
fi

sbatch \
    --job-name="${NAME}" \
    --output="${NAME}-%j.out" \
    --error="${NAME}-%j.out" \
    "${SCRIPT}"
