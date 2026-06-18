"""
PTQ calibration comparison: dynamic vs calibrated quantization for two models.

Simulates realistic deployment PTQ by replacing per-sample dynamic activation
ranges (what the current QIT eval uses) with static ranges computed over a
calibration dataset — exactly what TensorRT / ONNX Runtime do in practice.

Runs three quantization modes per (model, config) pair:
  dynamic  — per-sample min-max (current eval baseline)
  minmax   — dataset-level min-max calibration (full range)
  pct<N>   — percentile calibration, clips the top/bottom (100-N)/2 % of values
              Handles activation outliers that inflate the quantization step size.

Both models are calibrated independently on the same calibration set.

Usage:
    python scripts/run_ptq_comparison.py \\
        --backbone resnet18 --dataset stl10 \\
        --model_a runs/qit_stl10_resnet18/checkpoint_best.pt \\
        --model_a_label "QIT student" \\
        --model_b pretrained \\
        --model_b_label "Teacher (pretrained)" \\
        --n_calib_batches 32 --percentile 99.99 \\
        --output_dir runs/ptq_comparison_stl10_resnet18
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from models.quantization import calibrate_activations
from models.eval_utils import (
    _pbar, QUANT_CONFIGS, _TRANSFORM, load_backbone,
    extract_features, run_probe, split_train_val,
)


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def load_datasets(dataset: str, data_dir: str):
    """Returns (calib_ds, probe_train_ds, probe_test_ds)."""
    if dataset == "stl10":
        calib_ds       = datasets.STL10(data_dir, split="unlabeled", download=True, transform=_TRANSFORM)
        probe_train_ds = datasets.STL10(data_dir, split="train",     download=True, transform=_TRANSFORM)
        probe_test_ds  = datasets.STL10(data_dir, split="test",      download=True, transform=_TRANSFORM)
    elif dataset == "cifar10":
        train_ds       = datasets.CIFAR10(data_dir, train=True,  download=True, transform=_TRANSFORM)
        probe_test_ds  = datasets.CIFAR10(data_dir, train=False, download=True, transform=_TRANSFORM)
        # Use first 5 000 images for calibration, rest for probe training
        calib_ds       = Subset(train_ds, list(range(5_000)))
        probe_train_ds = Subset(train_ds, list(range(5_000, len(train_ds))))
    else:
        raise ValueError(f"Unknown dataset: {dataset!r}")
    return calib_ds, probe_train_ds, probe_test_ds


# ---------------------------------------------------------------------------
# Evaluation for one model
# ---------------------------------------------------------------------------

def evaluate_model(model, label, calib_loader,
                   probe_train_loader, probe_test_loader,
                   n_calib_batches, percentile,
                   weight_granularity, device, use_amp,
                   active_configs=None):
    print(f"\n  [{label}] calibrating ({n_calib_batches} batches, pct={percentile})...")
    stats_minmax = calibrate_activations(model, calib_loader, n_calib_batches,
                                         device, percentile=100.0, use_amp=use_amp)
    stats_pct    = calibrate_activations(model, calib_loader, n_calib_batches,
                                         device, percentile=percentile, use_amp=use_amp)

    model_results = {}
    pct_label = f"pct{percentile:.4g}"
    modes = [("dynamic", None), ("minmax", stats_minmax), (pct_label, stats_pct)]
    configs = active_configs if active_configs is not None else QUANT_CONFIGS

    for qk, wb, ab in configs:
        print(f"\n  [{label}] [{qk}] extracting features...")
        row = {}
        for mode_name, calib_stats in modes:
            tr_f, tr_l = extract_features(model, probe_train_loader, device, use_amp,
                                          wb, ab, weight_granularity, calib_stats)
            te_f, te_l = extract_features(model, probe_test_loader,  device, use_amp,
                                          wb, ab, weight_granularity, calib_stats)
            # Hold out a slice of the probe-train set for early stopping /
            # best-checkpoint selection.
            tr_idx, va_idx = split_train_val(len(tr_l))
            acc = run_probe(tr_f[tr_idx], tr_l[tr_idx], tr_f[va_idx], tr_l[va_idx], te_f, te_l, device)
            row[mode_name] = round(acc, 4)
            print(f"    {mode_name:<12} acc={acc:.4f}")
        model_results[qk] = row

    return model_results, pct_label


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

def print_table(results, labels, pct_label, quant_configs):
    modes = ["dynamic", "minmax", pct_label]
    col_w = 10

    header_models = "".join(
        f"  {'─' * ((col_w + 1) * len(modes) - 1)}  " for _ in labels
    )
    print(f"\n{'Config':<8}", end="")
    for lbl in labels:
        print(f"  {lbl[:((col_w + 1) * len(modes) - 1)]:<{(col_w + 1) * len(modes) - 1}}", end="")
    print()

    print(f"{'':8}", end="")
    for _ in labels:
        for m in modes:
            print(f"  {m:<{col_w - 2}}", end="")
    print()

    print("─" * (8 + len(labels) * ((col_w + 1) * len(modes) + 2)))
    for qk, _, _ in quant_configs:
        print(f"{qk:<8}", end="")
        for lbl in labels:
            row = results.get(lbl, {}).get(qk, {})
            for m in modes:
                v = row.get(m)
                print(f"  {v:.4f}" if v is not None else f"  {'—':>{col_w - 2}}", end="")
        print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone",         required=True,
                        choices=["resnet18", "efficientnet_b0",
                                 "mobilenet_v3_small", "vit_b_16", "swin_t"])
    parser.add_argument("--dataset",          required=True,
                        choices=["cifar10", "stl10"])
    parser.add_argument("--data_dir",         default=None)
    parser.add_argument("--model_a",          required=True,
                        help="Path to .pt checkpoint (key='student') or 'pretrained'.")
    parser.add_argument("--model_a_label",    default="model_a")
    parser.add_argument("--model_b",          required=True,
                        help="Path to .pt checkpoint (key='student') or 'pretrained'.")
    parser.add_argument("--model_b_label",    default="model_b")
    parser.add_argument("--n_calib_batches",  type=int, default=32,
                        help="Number of batches (~512 images at bs=16) for calibration.")
    parser.add_argument("--percentile",       type=float, default=99.9,
                        help="Percentile for activation clipping. 99.9 clips the top/bottom "
                             "0.05%% of observed values — appropriate for transformer models "
                             "with known outlier channels (e.g. ViT). Use 99.99 for CNNs.")
    parser.add_argument("--calib_batch_size", type=int, default=16)
    parser.add_argument("--weight_granularity", default="per_channel",
                        choices=["per_tensor", "per_channel"])
    parser.add_argument("--configs",          nargs="+",
                        choices=[qk for qk, _, _ in QUANT_CONFIGS],
                        default=None,
                        help="Quant configs to evaluate. Default: all. "
                             "E.g. --configs fp w8a8 w4a4")
    parser.add_argument("--output_dir",       default=None)
    parser.add_argument("--no_amp",           action="store_true")
    args = parser.parse_args()

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = not args.no_amp and device.type == "cuda"
    data_dir = args.data_dir or f"data/{args.dataset}"
    out = Path(args.output_dir or
               f"runs/ptq_comparison_{args.dataset}_{args.backbone}")
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"PTQ calibration comparison — {args.dataset.upper()} / {args.backbone}")
    print(f"  model_a : {args.model_a_label!r:30s}  ({args.model_a})")
    print(f"  model_b : {args.model_b_label!r:30s}  ({args.model_b})")
    print(f"  calib   : {args.n_calib_batches} batches × bs={args.calib_batch_size}"
          f"  percentile={args.percentile}")
    active_configs = (
        [c for c in QUANT_CONFIGS if c[0] in args.configs]
        if args.configs else QUANT_CONFIGS
    )
    print(f"  configs : {[c[0] for c in active_configs]}")
    print("=" * 70)

    # ── datasets ──────────────────────────────────────────────────────────
    print("\nLoading datasets...")
    calib_ds, probe_train_ds, probe_test_ds = load_datasets(args.dataset, data_dir)

    calib_loader = DataLoader(calib_ds, batch_size=args.calib_batch_size,
                              shuffle=False, num_workers=4, pin_memory=True)
    probe_train_loader = DataLoader(probe_train_ds, batch_size=args.calib_batch_size,
                                    shuffle=False, num_workers=4, pin_memory=True)
    probe_test_loader  = DataLoader(probe_test_ds,  batch_size=args.calib_batch_size,
                                    shuffle=False, num_workers=4, pin_memory=True)

    # ── models ────────────────────────────────────────────────────────────
    print("Loading models...")
    model_a = load_backbone(args.backbone, args.model_a, device)
    model_b = load_backbone(args.backbone, args.model_b, device)

    model_pairs = [
        (model_a, args.model_a_label),
        (model_b, args.model_b_label),
    ]

    # ── BN recalibration for CNN backbones ────────────────────────────────
    import torch.nn as nn
    for model, lbl in model_pairs:
        has_bn = any(isinstance(m, nn.BatchNorm2d) for m in model.modules())
        if has_bn:
            print(f"  BN recalibration for [{lbl}]...")
            model.train()
            with torch.no_grad():
                for i, batch in enumerate(calib_loader):
                    if i >= args.n_calib_batches:
                        break
                    model(batch[0].to(device))
            model.eval()

    # ── evaluate ──────────────────────────────────────────────────────────
    all_results = {}
    pct_label   = None
    for model, label in model_pairs:
        res, pct_label = evaluate_model(
            model, label, calib_loader,
            probe_train_loader, probe_test_loader,
            args.n_calib_batches, args.percentile,
            args.weight_granularity, device, use_amp,
            active_configs=active_configs,
        )
        all_results[label] = res

    # ── print summary table ───────────────────────────────────────────────
    print(f"\n{'Results':=^70}")
    labels = [args.model_a_label, args.model_b_label]
    print_table(all_results, labels, pct_label, active_configs)

    # ── save ──────────────────────────────────────────────────────────────
    out_path = out / "results.json"
    with open(out_path, "w") as f:
        json.dump({
            "backbone":         args.backbone,
            "dataset":          args.dataset,
            "model_a":          args.model_a,
            "model_a_label":    args.model_a_label,
            "model_b":          args.model_b,
            "model_b_label":    args.model_b_label,
            "n_calib_batches":  args.n_calib_batches,
            "percentile":       args.percentile,
            "weight_granularity": args.weight_granularity,
            "results":          all_results,
        }, f, indent=2)
    print(f"\nResults → {out_path}")


if __name__ == "__main__":
    main()
