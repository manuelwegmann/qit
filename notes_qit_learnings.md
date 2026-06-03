# QIT Learnings

Quantization Invariance Training (QIT): fine-tune a student backbone (same init as teacher)
so that its features under random fake-quantization match the frozen teacher's full-precision
features. At deployment, the student is more robust to low bit-widths without full retraining.

---

## Evaluation setup

Five ImageNet-pretrained backbones on CIFAR-10 and STL-10:
ResNet18, EfficientNet-B0, MobileNetV3-Small, ViT-B/16, Swin-T.

After QIT, a **linear probe** is trained on student features at FP, W8A8, W6A6, W4A6, W4A4,
and W2A4 and compared against the same probe on teacher features. Accuracy = top-1.

CNN backbones (ResNet18, EfficientNet, MobileNet) receive a BN recalibration pass (FP,
student weights) before probing. ViT and Swin use LayerNorm — no recal needed.

Teacher features are cached to disk once per backbone so training never re-runs the teacher.

---

## Results

### Transformers — very large gains

ViT-B/16 and Swin-T are highly quantization-sensitive without QIT; the student recovers
near-FP accuracy down to W4A4:

**STL-10 / ViT-B/16** (teacher FP = 0.985)
| Config | Teacher | Student | Δ |
|--------|---------|---------|---|
| FP     | 0.985   | 0.981   | −0.005 |
| W8A8   | 0.984   | 0.981   | −0.004 |
| W6A6   | 0.928   | 0.981   | +0.053 |
| W4A6   | 0.704   | 0.980   | +0.276 |
| W4A4   | 0.261   | **0.967** | **+0.707** |
| W2A4   | 0.186   | 0.244   | +0.058 |

**STL-10 / Swin-T** (teacher FP = 0.983)
| Config | Teacher | Student | Δ |
|--------|---------|---------|---|
| W4A4   | 0.846   | **0.971** | **+0.124** |
| W2A4   | 0.302   | **0.951** | **+0.650** |

**CIFAR-10 / ViT-B/16**: W4A4 +62.9pp (0.305 → 0.934), W4A6 +29.3pp.
**CIFAR-10 / Swin-T**: W2A4 +56.5pp (0.335 → 0.900), W4A4 +19.0pp.

### CNNs — mixed

**MobileNetV3-Small** benefits substantially (architecture is quantization-sensitive):
- STL-10 W6A6: +61.8pp (0.202 → 0.819); W8A8: +22.7pp
- CIFAR-10 W6A6: +33.4pp; W8A8: +18.4pp

**EfficientNet-B0**: modest gains (W8A8 ~+2–4pp; W6A6 ~+18pp on STL-10).

**ResNet18**: already robust on STL-10 (W4A4 +8.0pp). On CIFAR-10 student
is worse than teacher FP across all configs — failure case, possibly BN instability.

### PTQ calibration comparison — ViT-B/16 STL-10

QIT student vs. teacher with best per-tensor/percentile PTQ calibration:

| Config | Teacher (best PTQ) | QIT student |
|--------|--------------------|-------------|
| FP     | 0.986              | 0.981       |
| W8A8   | 0.986              | 0.981       |
| W6A6   | 0.969              | **0.981**   |
| W4A6   | 0.829              | **0.980**   |
| W4A4   | 0.576              | **0.977**   |
| W2A4   | 0.237              | 0.329       |

QIT accuracy is essentially flat from FP through W4A4; teacher accuracy collapses.

### QAT warm-start — QIT init vs. pretrained init

QIT provides a better starting point for quantization-aware training:

| Model + config          | Pretrained init | QIT init | Δ |
|-------------------------|-----------------|----------|---|
| ViT-B/16 STL-10 W4A4   | 0.608           | **0.951** | +34.3pp |
| MobileNet CIFAR-10 W8A8 | 0.888           | 0.931    | +4.3pp |

### Text (DistilBERT / SST-2) — failed

Student degraded at all configs (FP: 0.820 → 0.611; W8A8: 0.824 → 0.529).
NLP models appear to have different quantization dynamics; training config not tuned for text.

### Merlin CT — not completed

Runs crashed with CUDA OOM when caching teacher features for 250 training scans
on a 40 GB A100 (needed ~13 GB extra, only ~8 GB free). Full Merlin evaluation pending.

---

## Key lessons

**Loss:** cosine is essential. MSE/KL penalise feature magnitude as well as direction —
the optimizer collapses features. Cosine penalises angular deviation only.

**Architecture matters most:** transformers (ViT, Swin) degrade severely under PTQ because
of outlier activations; QIT eliminates that sensitivity almost entirely. CNNs vary —
MobileNet benefits strongly, ResNet18 less so (already robust).

**QIT as QAT warm-start:** even a partial reduction in quantization sensitivity translates
to a large QAT accuracy jump, especially for transformers.

**W2A4 ceiling:** even the best-case student does not recover at W2A4 (gains are small or
negative). W4A4 appears to be the practical floor for QIT.

**BN recalibration:** always recalibrate student BN, never teacher BN. Teacher's
ImageNet-trained stats are already correct for its own features.

**Feature caching:** cache teacher features before training. Avoids a full teacher forward
pass every step and enables large batch sizes without memory pressure.
