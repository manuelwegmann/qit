"""
Feature-space comparison: pretrained vs QIT, across quantization configs.

Given a QIT run directory (e.g. runs/qit_cifar10_vit_b_16), compares the
ImageNet-pretrained backbone against that run's checkpoint_best.pt on a
class-stratified subsample of the CIFAR-10 test set. For each quant config
(default FP and W4A4), extracts features and projects them to 2D with
t-SNE (default) or PCA. Saves a single grid figure (rows = pretrained/QIT,
cols = configs) into the run directory.

Usage:
    python scripts/plot_tsne_comparison.py runs/qit_cifar10_vit_b_16
    python scripts/plot_tsne_comparison.py runs/qit_cifar10_vit_b_16 --configs fp w4a4 w2a4
    python scripts/plot_tsne_comparison.py runs/qit_cifar10_vit_b_16 --method pca
    python scripts/plot_tsne_comparison.py runs/qit_cifar10_vit_b_16 --calibrate
"""

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from models.eval_utils import _TRANSFORM, load_backbone, extract_features
from models.quantization import calibrate_activations


def parse_config(cfg: str) -> tuple:
    """'fp' -> (None, None); 'w<bits>a<bits>' -> (w_bits, a_bits)."""
    if cfg == "fp":
        return None, None
    m = re.fullmatch(r"w(\d+)a(\d+)", cfg)
    if not m:
        raise ValueError(f"Invalid config {cfg!r}, expected 'fp' or 'w<bits>a<bits>' (e.g. 'w4a4').")
    return int(m.group(1)), int(m.group(2))


def stratified_indices(targets: np.ndarray, n_per_class: int, seed: int) -> np.ndarray:
    """Up to n_per_class indices per class, shuffled, deterministic given seed."""
    rng = np.random.default_rng(seed)
    idx = []
    for c in np.unique(targets):
        cls_idx = np.flatnonzero(targets == c)
        n = min(n_per_class, len(cls_idx))
        idx.append(rng.choice(cls_idx, size=n, replace=False))
    idx = np.concatenate(idx)
    rng.shuffle(idx)
    return idx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir",
                        help="QIT run directory containing checkpoint_best.pt and results.json "
                             "(e.g. runs/qit_cifar10_vit_b_16).")
    parser.add_argument("--configs", nargs="+", default=["fp", "w4a4"],
                        help="'fp' or 'w<bits>a<bits>', e.g. --configs fp w8a8 w2a8")
    parser.add_argument("--method", default="tsne", choices=["tsne", "pca"],
                        help="2D projection method.")
    parser.add_argument("--n_per_class", type=int, default=200,
                        help="Max CIFAR-10 test images per class in the embedding subsample.")
    parser.add_argument("--perplexity", type=float, default=30.0,
                        help="t-SNE only.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--data_dir", default="data/cifar10")
    parser.add_argument("--calibrate", action="store_true",
                        help="Use static percentile-based activation calibration instead of "
                             "per-sample dynamic min-max. Each model is calibrated independently "
                             "on the CIFAR-10 train set.")
    parser.add_argument("--percentile", type=float, default=99.9,
                        help="Percentile for activation clipping, only used with --calibrate. "
                             "99.9 clips the top/bottom 0.05%% of observed values.")
    parser.add_argument("--n_calib_batches", type=int, default=32,
                        help="Number of batches used for calibration, only used with --calibrate.")
    parser.add_argument("--out", default=None,
                        help="Output PNG path. Defaults to <run_dir>/<method>_comparison.png.")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    with open(run_dir / "results.json") as f:
        config = json.load(f)
    backbone = config["backbone"]
    weight_granularity = config.get("weight_granularity", "per_tensor")

    checkpoint = run_dir / "checkpoint_best.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"No checkpoint_best.pt in {run_dir}")

    suffix = f"_pct{args.percentile:g}" if args.calibrate else ""
    out_path = Path(args.out) if args.out else run_dir / f"{args.method}_comparison{suffix}.png"

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = not args.no_amp and device.type == "cuda"

    print("=" * 70)
    print(f"{args.method.upper()} comparison — CIFAR-10 / {backbone}")
    print(f"  run_dir : {run_dir}")
    print(f"  configs : {args.configs}")
    if args.calibrate:
        print(f"  calib   : {args.n_calib_batches} batches, percentile={args.percentile}")
    print("=" * 70)

    print("\nLoading CIFAR-10 test set...")
    test_ds = datasets.CIFAR10(args.data_dir, train=False, download=True, transform=_TRANSFORM)
    targets = np.array(test_ds.targets)
    class_names = test_ds.classes
    idx = stratified_indices(targets, args.n_per_class, args.seed)
    loader = DataLoader(Subset(test_ds, idx.tolist()), batch_size=args.batch_size,
                        shuffle=False, num_workers=args.num_workers, pin_memory=True)
    print(f"  {len(idx)} images ({args.n_per_class}/class x {len(class_names)} classes)")

    model_pairs = [
        ("Pretrained", load_backbone(backbone, "pretrained", device)),
        ("QIT",        load_backbone(backbone, str(checkpoint), device)),
    ]

    calib_loader = None
    if args.calibrate:
        print("\nLoading CIFAR-10 train set for calibration...")
        calib_ds = datasets.CIFAR10(args.data_dir, train=True, download=True, transform=_TRANSFORM)
        calib_loader = DataLoader(calib_ds, batch_size=args.batch_size, shuffle=False,
                                  num_workers=args.num_workers, pin_memory=True)

    if args.method == "tsne":
        from sklearn.manifold import TSNE
        def project(feats):
            return TSNE(n_components=2, perplexity=args.perplexity,
                        init="pca", random_state=args.seed, n_jobs=-1).fit_transform(feats)
    else:
        from sklearn.decomposition import PCA
        def project(feats):
            return PCA(n_components=2, random_state=args.seed).fit_transform(feats)

    y_ref = None
    embeddings = {}
    for label, model in model_pairs:
        model.eval()
        calib_stats = None
        if args.calibrate:
            print(f"  [{label}] calibrating ({args.n_calib_batches} batches, "
                  f"percentile={args.percentile})...")
            calib_stats = calibrate_activations(model, calib_loader, args.n_calib_batches,
                                                device, percentile=args.percentile, use_amp=use_amp)
        embeddings[label] = {}
        for cfg in args.configs:
            wb, ab = parse_config(cfg)
            print(f"  [{label}] [{cfg}] extracting features...")
            feats, y = extract_features(model, loader, device, use_amp, wb, ab,
                                        weight_granularity, calib_stats)
            if y_ref is None:
                y_ref = y
            print(f"  [{label}] [{cfg}] running {args.method}...")
            embeddings[label][cfg] = project(feats)

    # ── plot ───────────────────────────────────────────────────────────────
    model_labels = [label for label, _ in model_pairs]
    n_rows, n_cols = len(model_labels), len(args.configs)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 4.5 * n_rows), squeeze=False)

    cmap = plt.get_cmap("tab10")
    colors = [cmap(i) for i in range(len(class_names))]

    for r, label in enumerate(model_labels):
        for c, cfg in enumerate(args.configs):
            ax = axes[r][c]
            emb = embeddings[label][cfg]
            for ci in range(len(class_names)):
                mask = y_ref == ci
                ax.scatter(emb[mask, 0], emb[mask, 1], s=8, color=colors[ci],
                          alpha=0.7, linewidths=0, label=class_names[ci])
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(cfg.upper(), fontsize=13, fontweight="bold")
            if c == 0:
                ax.set_ylabel(label, fontsize=13, fontweight="bold")

    handles, labels_ = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc="center left", bbox_to_anchor=(1.0, 0.5),
              fontsize=9, frameon=False, markerscale=2, title="Class")
    title = f"{args.method.upper()} — CIFAR-10 / {backbone}"
    if args.calibrate:
        title += f" (calibrated, pct={args.percentile:g})"
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nFigure → {out_path}")


if __name__ == "__main__":
    main()
