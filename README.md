# QIT — Quantization-Invariant Training

Fine-tunes a pretrained model to produce stable representations across arbitrary quantization bit-widths, enabling deployment at W2–W8 / A4–A8 without retraining from scratch. Implemented for vision backbones (`run_qit.py`), text encoders (`run_qit_text.py`), and the Merlin CT foundation model (`run_qit_merlin.py`).

## Idea

When a model is already pretrained and fits the target data, full retraining is wasteful. QIT instead uses knowledge distillation: a frozen teacher (the original pretrained model) supervises a student that sees its own weights and activations randomly quantized each forward pass. The student learns to match the teacher's full-precision representations regardless of the quantization applied, making it robust to low bit-width deployment.

## Method

- **Teacher**: frozen pretrained model, full precision
- **Student**: same architecture and weights, fine-tuned with QIT
- **Training**: each step samples random bit-widths `(w_bits, a_bits)` and runs a fake-quantized forward pass through the student; loss is the distillation distance to the teacher's cached features
- **Loss options**: cosine similarity (default), MSE, KL divergence
- **Quantization**: fake-quantization with straight-through estimator (STE); weights per-tensor or per-channel, activations per-tensor

## Validation

Evaluated on STL-10 with five ImageNet-pretrained backbones: ResNet18, EfficientNet-B0, MobileNetV3-Small, ViT-B/16, and Swin-T. After QIT fine-tuning, a linear probe is evaluated at FP, W8A8, W6A6, W4A6, W4A4, and W2A4 — checking that student accuracy degrades gracefully compared to the teacher across all backbones and bit-width configs.

CNN backbones (ResNet18, EfficientNet-B0, MobileNetV3-Small) undergo BatchNorm recalibration after training. ViT-B/16 and Swin-T use LayerNorm and skip this step.

Teacher features are computed once per backbone and cached to disk (`data/stl10/teacher_feats_<backbone>.pt`), so repeated runs and hyperparameter sweeps do not recompute them.

## Usage

```bash
# download data once before submitting any jobs
python scripts/download_data.py

# run all five backbones on Slurm
BACKBONE=resnet18           sbatch scripts/run_slurm.sbatch
BACKBONE=efficientnet_b0    sbatch scripts/run_slurm.sbatch
BACKBONE=mobilenet_v3_small sbatch scripts/run_slurm.sbatch
BACKBONE=vit_b_16           sbatch scripts/run_slurm.sbatch
BACKBONE=swin_t             sbatch scripts/run_slurm.sbatch

# quick trial (e.g. to validate a new backbone)
BACKBONE=resnet18 N_TRAIN=500 EPOCHS=3 PATIENCE=0 DATASET=cifar10 sbatch scripts/run_slurm.sbatch
```

Results are written to `runs/qit_<dataset>_<backbone>/results.json`.

## Merlin (CT)

The Merlin track applies QIT to the Merlin foundation model (I3ResNet-152) on CT
volumes, evaluated by linear-probe AUROC over 30 disease conditions. It expects
the `quantized_ft` and `CT-CLIP` repositories as siblings of this one.

Unlike the vision/text scripts, training and evaluation are **separate**:
`run_qit_merlin.py` only produces checkpoints, and probing is done afterwards on
cached, statically-calibrated features.

```bash
# 1. train (add --full_data to train on combined train+val with no early stopping)
sbatch scripts/run_slurm_qit_merlin.sbatch

# 2. cache features for all bit-width configs (student + teacher)
sbatch scripts/run_slurm_cache_student_features.sbatch
sbatch scripts/run_slurm_cache_teacher_features.sbatch

# 3. per-condition linear-probe AUROC (early-stopped per condition), then plot
FEAT_DIR=runs/qit_merlin_fulldata/features sbatch scripts/run_slurm_probe_merlin.sbatch
python scripts/plot_probe_merlin.py --student .../probe_results/results.json \
    --teacher .../probe_results/results.json --out_dir plots/merlin
```

Other evaluation helpers apply across all three settings: `run_qat*.py`
(quantization-aware training), `run_ptq_comparison*.py` (dynamic vs. calibrated
PTQ), and `compute_bitops.py` (theoretical compute savings).

Key arguments:

| Flag | Default | Description |
|---|---|---|
| `--backbone` | `resnet18` | `resnet18`, `efficientnet_b0`, `mobilenet_v3_small`, `vit_b_16`, `swin_t` |
| `--dataset` | `stl10` | `stl10` or `cifar10` |
| `--epochs` | `50` | Max QIT training epochs (0 = skip training, probe only) |
| `--patience` | `10` | Early stopping patience (epochs without improvement in moving-avg val loss) |
| `--val_window` | `3` | Window size for moving-average val loss; best checkpoint is the middle epoch of the best window |
| `--loss` | `cosine` | `cosine`, `mse`, or `kl` |
| `--weight_granularity` | `per_channel` | `per_tensor` or `per_channel` |
| `--n_train` | `7000` | Number of QIT training images (-1 = all) |
| `--w_bits_min/max` | `2/4` | Weight bit-width sampling range |
| `--a_bits_min/max` | `4/8` | Activation bit-width sampling range |

## Setup

```bash
python3.11 -m venv venv
pip install torch==2.10.0+cu128 torchvision==0.25.0+cu128 \
    --index-url https://download.pytorch.org/whl/cu128 --no-deps
pip install -r requirements.txt
```
