"""
Cross-dataset transfer eval for a QIT-trained backbone.

Loads a frozen QIT student checkpoint (e.g. trained via run_qit.py on STL-10),
trains fresh linear probes on a *different* dataset's labeled splits, and
sweeps the same FP/W8A8/.../W2A4 bit-width configs used elsewhere in this repo.
The ImageNet-pretrained teacher is evaluated alongside as a baseline, so the
student-vs-teacher deltas are directly comparable to the same-dataset QIT
results.json.

Usage:
    python scripts/run_cross_dataset_eval.py \\
        --backbone vit_b_16 \\
        --source_dataset stl10 --target_dataset cifar10
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from models.eval_utils import (
    _pbar, QUANT_CONFIGS, _TRANSFORM, build_backbone, load_backbone,
    extract_features, run_probe, split_train_val,
)


def load_target_dataset(dataset: str, data_dir: str):
    """Returns (probe_train_ds, probe_test_ds) with labels, for linear-probe eval."""
    if dataset == "stl10":
        probe_train_ds = datasets.STL10(data_dir, split="train", download=True, transform=_TRANSFORM)
        probe_test_ds  = datasets.STL10(data_dir, split="test",  download=True, transform=_TRANSFORM)
    elif dataset == "cifar10":
        probe_train_ds = datasets.CIFAR10(data_dir, train=True,  download=True, transform=_TRANSFORM)
        probe_test_ds  = datasets.CIFAR10(data_dir, train=False, download=True, transform=_TRANSFORM)
    else:
        raise ValueError(f"Unknown dataset: {dataset!r}")
    return probe_train_ds, probe_test_ds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", required=True,
                        choices=["resnet18", "efficientnet_b0",
                                 "mobilenet_v3_small", "vit_b_16", "swin_t"])
    parser.add_argument("--checkpoint", default=None,
                        help="QIT student checkpoint to evaluate. Defaults to "
                             "runs/qit_<source_dataset>_<backbone>/checkpoint_best.pt")
    parser.add_argument("--source_dataset", default="stl10", choices=["cifar10", "stl10"],
                        help="Dataset the checkpoint was QIT-trained on "
                             "(used for the default checkpoint path and labeling).")
    parser.add_argument("--target_dataset", default="cifar10", choices=["cifar10", "stl10"],
                        help="Dataset to train/eval the linear probes on.")
    parser.add_argument("--data_dir", default=None,
                        help="Target dataset root dir. Defaults to data/<target_dataset>.")
    parser.add_argument("--n_probe_train", type=int, default=-1,
                        help="Number of target-dataset training images for the probe. "
                             "-1 = all.")
    parser.add_argument("--n_probe_test", type=int, default=-1,
                        help="Number of target-dataset test images for evaluation. "
                             "-1 = all. Useful for quick smoke tests.")
    parser.add_argument("--weight_granularity", default="per_tensor",
                        choices=["per_tensor", "per_channel"])
    parser.add_argument("--configs", nargs="+",
                        choices=[qk for qk, _, _ in QUANT_CONFIGS], default=None,
                        help="Quant configs to evaluate. Default: all.")
    parser.add_argument("--recalibrate_bn", action="store_true",
                        help="Reset BatchNorm running stats with a FP pass over the "
                             "target dataset's probe-train split before evaluation. "
                             "BN stats calibrated on the source dataset's input "
                             "distribution may not transfer to the target domain. "
                             "No-op for ViT/Swin (LayerNorm).")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = not args.no_amp and device.type == "cuda"

    checkpoint = args.checkpoint or f"runs/qit_{args.source_dataset}_{args.backbone}/checkpoint_best.pt"
    data_dir   = args.data_dir or f"data/{args.target_dataset}"
    out = Path(args.output_dir or
               f"runs/cross_{args.source_dataset}_to_{args.target_dataset}_{args.backbone}")
    out.mkdir(parents=True, exist_ok=True)

    active_configs = (
        [c for c in QUANT_CONFIGS if c[0] in args.configs]
        if args.configs else QUANT_CONFIGS
    )

    print("=" * 70)
    print(f"Cross-dataset QIT transfer — {args.backbone}")
    print(f"  checkpoint      : {checkpoint}")
    print(f"  trained on      : {args.source_dataset.upper()}")
    print(f"  probing on      : {args.target_dataset.upper()}")
    print(f"  recalibrate_bn  : {args.recalibrate_bn}")
    print(f"  configs         : {[c[0] for c in active_configs]}")
    print(f"  output          : {out}")
    print("=" * 70)

    # ── data ──────────────────────────────────────────────────────────────
    print(f"\nLoading {args.target_dataset.upper()}...")
    probe_train_ds, probe_test_ds = load_target_dataset(args.target_dataset, data_dir)
    if args.n_probe_train > 0 and args.n_probe_train < len(probe_train_ds):
        probe_train_ds = Subset(probe_train_ds, list(range(args.n_probe_train)))
    if args.n_probe_test > 0 and args.n_probe_test < len(probe_test_ds):
        probe_test_ds = Subset(probe_test_ds, list(range(args.n_probe_test)))

    probe_train_loader = DataLoader(probe_train_ds, batch_size=args.batch_size, shuffle=False,
                                    num_workers=4, pin_memory=True)
    probe_test_loader  = DataLoader(probe_test_ds,  batch_size=args.batch_size, shuffle=False,
                                    num_workers=4, pin_memory=True)
    print(f"  Probe train: {len(probe_train_ds)} | Probe test: {len(probe_test_ds)}")

    # ── models ────────────────────────────────────────────────────────────
    print("\nLoading models...")
    print(f"  Student: {args.backbone} QIT checkpoint ({checkpoint})")
    student = load_backbone(args.backbone, checkpoint, device)
    print(f"  Teacher: ImageNet-pretrained {args.backbone}")
    teacher = build_backbone(args.backbone).to(device)
    student.eval()
    teacher.eval()

    if args.recalibrate_bn:
        for model, label in [(teacher, "teacher"), (student, "student")]:
            has_bn = any(isinstance(m, nn.modules.batchnorm._BatchNorm) for m in model.modules())
            if not has_bn:
                continue
            print(f"  Recalibrating BN stats for [{label}] on {args.target_dataset.upper()}...")
            model.train()
            with torch.no_grad():
                for batch in _pbar(probe_train_loader, desc=f"  {label} BN recal", leave=False):
                    x = batch[0].to(device)
                    if use_amp:
                        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                            model(x)
                    else:
                        model(x)
            model.eval()

    # ── probe sweep ───────────────────────────────────────────────────────
    print(f"\n{'Probe evaluation':=^70}")
    print(f"  {'Config':<8}  {'Teacher acc':>12}  {'Student acc':>12}  "
          f"{'Δ':>8}  {'feat_sim':>10}")

    results = {}
    for qk, wb, ab in active_configs:
        print(f"\n  [{qk}] extracting features...")
        t_tr, l_tr = extract_features(teacher, probe_train_loader, device, use_amp, wb, ab, args.weight_granularity)
        s_tr, _    = extract_features(student, probe_train_loader, device, use_amp, wb, ab, args.weight_granularity)
        t_te, l_te = extract_features(teacher, probe_test_loader,  device, use_amp, wb, ab, args.weight_granularity)
        s_te, _    = extract_features(student, probe_test_loader,  device, use_amp, wb, ab, args.weight_granularity)

        # Hold out a slice of the probe-train set for early stopping / best-
        # checkpoint selection (same split for teacher and student features).
        tr_idx, va_idx = split_train_val(len(l_tr))
        feat_sim    = float(F.cosine_similarity(
            torch.tensor(s_tr), torch.tensor(t_tr)).mean())
        teacher_acc = run_probe(t_tr[tr_idx], l_tr[tr_idx], t_tr[va_idx], l_tr[va_idx], t_te, l_te, device)
        student_acc = run_probe(s_tr[tr_idx], l_tr[tr_idx], s_tr[va_idx], l_tr[va_idx], s_te, l_te, device)
        delta       = student_acc - teacher_acc

        print(f"  {qk:<8}  {teacher_acc:>12.4f}  {student_acc:>12.4f}  "
              f"  {delta:>+8.4f}  {feat_sim:>10.4f}")
        results[qk] = {
            "teacher_acc": teacher_acc,
            "student_acc": student_acc,
            "delta":       delta,
            "feat_sim":    feat_sim,
        }

    out_path = out / "results.json"
    with open(out_path, "w") as f:
        json.dump({
            "backbone":           args.backbone,
            "checkpoint":         checkpoint,
            "source_dataset":     args.source_dataset,
            "target_dataset":     args.target_dataset,
            "weight_granularity": args.weight_granularity,
            "recalibrate_bn":     args.recalibrate_bn,
            "n_probe_train":      len(probe_train_ds),
            "results":            results,
        }, f, indent=2)
    print(f"\nResults → {out_path}")


if __name__ == "__main__":
    main()
