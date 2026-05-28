# QIT Learnings

Quantization Invariance Training (QIT): fine-tune a student backbone (same weights as
teacher) to produce similar feature representations under random fake-quantization, so
that the model degrades less when deployed at lower bit-widths.

---

## CT-CLIP experiments

All CT-CLIP runs use the pre-VQ features (512-dim) from the frozen teacher.
Teacher features for all 5097 scans are cached at `runs/feature_cache_full/pretrained_pre_vq/`.
Evaluation: macro-mean AUROC across 6 conditions, linear probe, N=all.

### Baseline (PTQ, no QIT)
| Config | AUROC |
|--------|-------|
| FP     | 0.657 |
| W8A8   | 0.653 |
| W4A4   | 0.531 |

### `runs/qit_test` — cosine loss, per-tensor, 10 epochs, 500 scans
3 quantizations per step (original setup). **First positive signal.**

| Config | Teacher | Student | Δ | feat_sim |
|--------|---------|---------|---|---------|
| FP     | 0.657   | 0.646   | −0.011 | 0.982 |
| W8A8   | 0.653   | 0.646   | −0.007 | 0.844 |
| W4A4   | 0.547   | **0.566** | **+0.019** | 0.205 |

W4A4 improves over naive PTQ (+3.5pp). FP features well-preserved (feat_sim=0.982).
Val_loss plateau after epoch 6; no meaningful further improvement with more epochs.

### `runs/qit_mse` — MSE loss, single quantization/step, 10 epochs, 500 scans
### `runs/qit_kl` — KL loss, single quantization/step, 10 epochs, 500 scans (partial)

Both significantly worse than cosine:

| Config | Teacher | MSE student | KL student |
|--------|---------|-------------|------------|
| FP     | 0.657   | 0.537 (−0.120) | 0.537 (−0.120) |
| W8A8   | 0.653   | 0.526 (−0.127) | — |
| W4A4   | 0.547   | 0.508 (−0.039) | — |

**Why:** MSE and KL penalise feature magnitude as well as direction. The optimizer
reduces loss by shrinking feature vectors, destroying the class-discriminative structure.
Cosine loss only penalises angular deviation, leaving magnitude structure intact.

**Lesson: use cosine loss.**

### `runs/qit_cosine_pc` — cosine, per-channel, 5 epochs, 1000 scans
Failed with `RuntimeError: GET was unable to find an engine` (CTViT PEG Conv3d).
Root cause: job landed on `hendrixgpu12fl` (Quadro RTX 6000, 24 GB).
Per-channel quantization requires more cuDNN workspace; just enough memory to
trigger a failure that per-tensor quantization avoided.
Fix: added 12fl to node exclusion list.

---

## CIFAR-10 / ResNet18 experiments

Used for rapid iteration on training pipeline design. All runs use
`scripts/run_qit_cifar.py` with `--weight_granularity per_channel`.

**Important caveat:** ResNet18 is already highly quantization-robust (W4A4 only −17.7pp),
so absolute improvements are small. Results are better interpreted as relative comparisons
than evidence of method effectiveness.

### `runs/qit_cifar` — bs=256, no early stopping, sklearn probe (broken — sklearn API issue)

### `runs/qit_cifar_pc` — bs=16, cosine decay, PyTorch linear probe, student BN recal only

| Config | Teacher | Student | Δ | feat_sim |
|--------|---------|---------|---|---------|
| FP     | 0.869   | 0.854   | −0.015 | 0.891 |
| W8A8   | 0.868   | 0.846   | −0.022 | 0.892 |
| W4A4   | 0.692   | **0.775** | **+0.083** | 0.647 |
| W2A4   | 0.225   | 0.230   | +0.005 | 0.819 |

Best result for CIFAR-10. **But caveat:** early stopping selected epoch 1 (barely
trained); the gain may partly reflect BN recalibration adapting ImageNet BN stats
to CIFAR-10 rather than QIT learning quantization robustness.

### `runs/qit_cifar_pc_v2` — same + teacher BN recalibration added, 50 epochs

| Config | Teacher | Student | Δ |
|--------|---------|---------|---|
| FP     | 0.805   | 0.753   | −0.052 |
| W4A4   | 0.542   | 0.516   | −0.026 |

Much worse. Teacher BN recalibration on CIFAR-10 *hurt* the teacher (0.869→0.805)
because its ImageNet-trained BN stats were already appropriate for its learned features.
**Lesson: recalibrate student BN only, not teacher.**

---

## Key lessons

### Loss function
- **Cosine is the only viable option** tested so far. MSE and KL destroy FP features.
- Combined loss `α·L_QIT + β·L_FP` (QIT on quantized path + cosine preservation on
  FP path) is theoretically sound as a way to prevent FP drift but needs `β` small
  (~0.1). At β=0.5 the FP term dominates and fights the QIT objective.

### Quantization granularity
- **Per-tensor** (one scale per tensor): coarser, safer, no cuDNN issues.
- **Per-channel** (one scale per output channel): closer to real deployment, but caused
  cuDNN workspace failures on 24 GB GPU node. Should work on A100 with 12fl excluded.
- CT-CLIP Linear weights have near-uniform distribution (kurtosis ≈ −1.2); no dramatic
  outlier channels. The difference between per-tensor and per-channel may be less
  impactful here than for transformer models with known outlier channels.

### Training setup
- **Single quantization per step**: cleaner gradient signal than averaging N quantizations.
- **Teacher feature cache** (`runs/feature_cache_full/pretrained_pre_vq/feats.pt`):
  eliminates teacher forward pass during training — major speedup.
- **LR cosine annealing**: smoother convergence, especially in later epochs.
- **Weight regularisation toward teacher** (`reg_weight`): prevents weight range from
  growing monotonically (observed in all runs). Weight range increase = larger
  quantization step size = worse W4A4 performance. Start with `reg_weight=1e-5`.
- **Batch size**: smaller = more diverse quantization configs per epoch. For CT-CLIP
  batch_size=2 is the memory limit.

### Evaluation / BN
- **CT-CLIP uses LayerNorm** throughout — BN recalibration is irrelevant for CT-CLIP.
  Student features are clean at eval time without any extra steps.
- **CIFAR-10 / ResNet18**: student BN stats are corrupted by mixed quantized/FP training
  passes. One FP recalibration pass is essential before probe evaluation.
- **Do not recalibrate teacher BN**: ImageNet-trained stats are already correct for the
  teacher's features.

### Early stopping
- Val_loss is noisy (single random bit-width per validation batch).
- Rolling 3-epoch mean + middle-of-window checkpoint selection reduces noise.
- Criterion: smoothed val_loss (lower = better quantization robustness), patience=5.

### Why CIFAR-10 is a weak testbed
- ResNet18 W4A4 only degrades 17.7pp — small room for improvement.
- CT-CLIP W4A4 degrades ~12pp on a harder task with more complex weight structure —
  more meaningful signal for QIT to improve.

---

## Recommended CT-CLIP setup (not yet run)

```bash
N_SCANS=1000  EPOCHS=15  LOSS=cosine  \
WEIGHT_GRAN=per_channel  \
FP_LOSS_WEIGHT=0.1  REG_WEIGHT=1e-5  \
OUTPUT_DIR=runs/qit_proper  \
sbatch scripts/run_qit_slurm.sbatch
```

Missing from `run_qit.py` (ported to CIFAR script only, needs backport):
- `--fp_loss_weight`
- `--reg_weight`
- `--lr_schedule` (cosine annealing)
- Early stopping (`--patience`, `--es_window`)

These should be ported from `scripts/run_qit_cifar.py` before running.
