"""
Quantization-Aware Training (QAT) for classification.

Fine-tunes a backbone + linear head end-to-end with fake quantization at a
fixed bit-depth. Designed to run twice — once from a QIT checkpoint and once
from the ImageNet-pretrained baseline — to test whether QIT provides a better
starting point for QAT convergence and accuracy ceiling.

The backbone is quantized (quantized_forward); the classification head runs
in full precision (one linear layer, no benefit to quantizing it separately).

Per-epoch log: train_loss, train_acc, val_loss, val_acc — written to
results.json for cross-run comparison.

Usage:
    # Pretrained baseline
    python scripts/run_qat.py \\
        --backbone resnet18 --dataset cifar10 \\
        --init pretrained \\
        --w_bits 4 --a_bits 4 --epochs 30 \\
        --output_dir runs/qat_cifar10_resnet18_pretrained_w4a4

    # QIT-initialised
    python scripts/run_qat.py \\
        --backbone resnet18 --dataset cifar10 \\
        --init runs/qit_cifar10_resnet18/checkpoint_best.pt \\
        --w_bits 4 --a_bits 4 --epochs 30 \\
        --output_dir runs/qat_cifar10_resnet18_qit_w4a4
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from models.quantization import quantized_forward
from models.eval_utils import (
    _pbar, FEATURE_DIMS, load_backbone, load_vision_dataset,
)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

N_CLASSES = {"cifar10": 10, "stl10": 10}


def load_datasets(dataset: str, data_dir: str, val_frac: float = 0.1):
    """Returns (train_ds, val_ds, test_ds) with labeled splits only."""
    full_train = load_vision_dataset(dataset, "train", data_dir)
    test_ds    = load_vision_dataset(dataset, "test",  data_dir)

    n      = len(full_train)
    n_val  = max(1, int(n * val_frac))
    rng    = np.random.default_rng(42)
    idx    = rng.permutation(n).tolist()
    train_ds = Subset(full_train, idx[n_val:])
    val_ds   = Subset(full_train, idx[:n_val])
    return train_ds, val_ds, test_ds


# ---------------------------------------------------------------------------
# One epoch
# ---------------------------------------------------------------------------

def run_epoch(backbone, head, loader, optimizer, w_bits, a_bits,
              weight_granularity, device, use_amp, train: bool):
    backbone.train() if train else backbone.eval()
    head.train()     if train else head.eval()

    total_loss = total_correct = n = 0

    ctx_grad = torch.enable_grad() if train else torch.no_grad()
    with ctx_grad:
        for batch in _pbar(loader, desc="  train" if train else "  val  ", leave=False):
            x, y = batch[0].to(device), batch[1].to(device)

            if use_amp:
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    with quantized_forward([backbone], w_bits, a_bits, weight_granularity):
                        feats = backbone(x)
                    logits = head(feats.float())
            else:
                with quantized_forward([backbone], w_bits, a_bits, weight_granularity):
                    feats = backbone(x)
                logits = head(feats)

            loss = F.cross_entropy(logits, y)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss    += loss.item() * len(y)
            total_correct += (logits.argmax(1) == y).sum().item()
            n             += len(y)

    return total_loss / n, total_correct / n


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone",   required=True,
                        choices=["resnet18", "efficientnet_b0",
                                 "mobilenet_v3_small", "vit_b_16", "swin_t"])
    parser.add_argument("--dataset",    required=True, choices=["cifar10", "stl10"])
    parser.add_argument("--data_dir",   default=None)
    parser.add_argument("--init",       required=True,
                        help="'pretrained' or path to QIT checkpoint (.pt with 'student' key).")
    parser.add_argument("--w_bits",     type=int, default=4)
    parser.add_argument("--a_bits",     type=int, default=4)
    parser.add_argument("--epochs",     type=int, default=30)
    parser.add_argument("--patience",   type=int, default=0,
                        help="Early stopping: stop if val_acc does not improve for "
                             "this many epochs. 0 = disabled.")
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--lr_schedule", default="cosine", choices=["cosine", "constant"])
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--weight_granularity", default="per_channel",
                        choices=["per_tensor", "per_channel"])
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--no_amp",     action="store_true")
    args = parser.parse_args()

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = not args.no_amp and device.type == "cuda"
    data_dir = args.data_dir or f"data/{args.dataset}"
    init_tag = "pretrained" if args.init == "pretrained" else "qit"
    out = Path(args.output_dir or
               f"runs/qat_{args.dataset}_{args.backbone}_{init_tag}_w{args.w_bits}a{args.a_bits}")
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"QAT — {args.dataset.upper()} / {args.backbone}  W{args.w_bits}A{args.a_bits}")
    print(f"  init            : {args.init}")
    print(f"  epochs          : {args.epochs}  patience={args.patience}  lr={args.lr} ({args.lr_schedule})")
    print(f"  batch_size      : {args.batch_size}")
    print(f"  weight_gran     : {args.weight_granularity}")
    print(f"  amp             : {use_amp}")
    print(f"  output          : {out}")
    print("=" * 70)

    # ── data ──────────────────────────────────────────────────────────────
    train_ds, val_ds, test_ds = load_datasets(args.dataset, data_dir)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False,
                              num_workers=4, pin_memory=True)
    print(f"\n  train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}\n")

    # ── model ─────────────────────────────────────────────────────────────
    backbone = load_backbone(args.backbone, args.init, device)
    feat_dim = FEATURE_DIMS[args.backbone]
    n_cls    = N_CLASSES[args.dataset]
    head     = nn.Linear(feat_dim, n_cls).to(device)

    # ── optimizer / scheduler ─────────────────────────────────────────────
    optimizer = torch.optim.Adam(
        list(backbone.parameters()) + list(head.parameters()), lr=args.lr
    )
    scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(
                     optimizer, T_max=args.epochs, eta_min=0)
                 if args.lr_schedule == "cosine" else None)

    # ── training ──────────────────────────────────────────────────────────
    print(f"\n{'Training':=^70}")
    print(f"{'Epoch':>6}  {'train_loss':>10}  {'train_acc':>9}  "
          f"{'val_loss':>9}  {'val_acc':>8}")

    epoch_log     = []
    best_val_acc  = 0.0
    best_val_epoch = 0
    no_improve    = 0

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = run_epoch(backbone, head, train_loader, optimizer,
                                    args.w_bits, args.a_bits, args.weight_granularity,
                                    device, use_amp, train=True)
        va_loss, va_acc = run_epoch(backbone, head, val_loader, None,
                                    args.w_bits, args.a_bits, args.weight_granularity,
                                    device, use_amp, train=False)

        if scheduler is not None:
            scheduler.step()

        improved = va_acc > best_val_acc
        marker   = " *" if improved else ""
        if improved:
            best_val_acc   = va_acc
            best_val_epoch = epoch
            no_improve     = 0
            torch.save({"epoch": epoch,
                        "backbone": backbone.state_dict(),
                        "head":     head.state_dict()}, out / "checkpoint_best.pt")
        else:
            no_improve += 1

        print(f"{epoch:6d}  {tr_loss:10.4f}  {tr_acc:9.4f}  "
              f"{va_loss:9.4f}  {va_acc:8.4f}{marker}")
        epoch_log.append({
            "epoch": epoch,
            "train_loss": round(tr_loss, 6), "train_acc": round(tr_acc, 6),
            "val_loss":   round(va_loss, 6), "val_acc":   round(va_acc, 6),
        })

        if args.patience > 0 and no_improve >= args.patience:
            print(f"\n[early stop] val_acc did not improve for {args.patience} epochs.")
            break

    torch.save({"epoch": epoch,
                "backbone": backbone.state_dict(),
                "head":     head.state_dict()}, out / "checkpoint_final.pt")

    # ── test evaluation ───────────────────────────────────────────────────
    print(f"\n{'Test evaluation':=^70}")
    ckpt = torch.load(out / "checkpoint_best.pt", map_location=device)
    backbone.load_state_dict(ckpt["backbone"])
    head.load_state_dict(ckpt["head"])

    _, test_acc = run_epoch(backbone, head, test_loader, None,
                            args.w_bits, args.a_bits, args.weight_granularity,
                            device, use_amp, train=False)
    print(f"  Test accuracy (best val ckpt, W{args.w_bits}A{args.a_bits}): {test_acc:.4f}")

    # ── save ──────────────────────────────────────────────────────────────
    out_path = out / "results.json"
    with open(out_path, "w") as f:
        json.dump({
            "backbone":           args.backbone,
            "dataset":            args.dataset,
            "init":               args.init,
            "w_bits":             args.w_bits,
            "a_bits":             args.a_bits,
            "epochs":             args.epochs,
            "patience":           args.patience,
            "lr":                 args.lr,
            "lr_schedule":        args.lr_schedule,
            "batch_size":         args.batch_size,
            "weight_granularity": args.weight_granularity,
            "n_train":            len(train_ds),
            "n_val":              len(val_ds),
            "n_test":             len(test_ds),
            "test_acc":           round(test_acc, 6),
            "best_val_acc":       round(best_val_acc, 6),
            "best_val_epoch":     best_val_epoch,
            "epoch_log":          epoch_log,
        }, f, indent=2)
    print(f"\nResults → {out_path}")


if __name__ == "__main__":
    main()
