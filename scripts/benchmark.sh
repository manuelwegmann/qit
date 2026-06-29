#!/bin/bash
# Unified driver for the CIFAR-10 / STL-10 vision benchmark.
#
# Thin orchestration only — it submits the existing scripts with their existing
# flags and changes no behaviour. Use it to launch the canonical sweep (the one
# previously living in README prose) with one command, then summarise.
#
# Usage:
#   bash scripts/benchmark.sh train [cifar10|stl10|both]   # submit QIT jobs
#   bash scripts/benchmark.sh plot  [cifar10|stl10|both]   # grid figures (after jobs finish)
#
# Examples:
#   bash scripts/benchmark.sh train both
#   bash scripts/benchmark.sh plot  stl10

set -euo pipefail

BACKBONES=(resnet18 efficientnet_b0 mobilenet_v3_small vit_b_16 swin_t)
STAGE="${1:-}"
WHICH="${2:-both}"

case "$WHICH" in
    cifar10) DATASETS=(cifar10) ;;
    stl10)   DATASETS=(stl10) ;;
    both)    DATASETS=(cifar10 stl10) ;;
    *) echo "second arg must be cifar10|stl10|both (got '$WHICH')"; exit 1 ;;
esac

case "$STAGE" in
    train)
        for ds in "${DATASETS[@]}"; do
            for bb in "${BACKBONES[@]}"; do
                echo "submit QIT: backbone=$bb dataset=$ds"
                BACKBONE="$bb" DATASET="$ds" bash scripts/submit.sh
            done
        done
        echo "All QIT jobs submitted. After they finish: bash scripts/benchmark.sh plot $WHICH"
        ;;
    plot)
        for ds in "${DATASETS[@]}"; do
            echo "plot grid: dataset=$ds"
            python scripts/plot_all_backbones.py --dataset "$ds"
        done
        ;;
    *)
        echo "usage: bash scripts/benchmark.sh {train|plot} [cifar10|stl10|both]"
        exit 1
        ;;
esac
