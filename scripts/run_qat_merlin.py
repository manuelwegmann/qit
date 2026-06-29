"""
QAT (Quantization-Aware Training) — Merlin CT, single clinical condition.

Two-phase linear-probe-then-finetune (LP-FT) recipe with fake quantization
(quantized_forward) at a fixed bit-width, supervised by one clinical
condition's label (BCE loss):

  Phase 1 (--phase1_epochs): the whole encoder is frozen (eval mode, fixed
    BN stats) and only the linear head is trained, on quantized features.
    This lets the head converge to a sensible decision boundary before any
    encoder weights move, so phase 2 doesn't start by backpropagating
    through a fresh random head into the encoder.

  Phase 2 (--phase2_epochs): unfreeze the last residual stage
    (image_encoder.i3_resnet.layer4) in addition to the head, and continue
    training. Earlier stages (conv1/layer1-3) stay frozen — full end-to-end
    fine-tuning on ~1.4k single-condition examples was found to collapse the
    encoder's representation within a few epochs regardless of init or LR.

Run twice — once from the pretrained Merlin checkpoint and once from a QIT
checkpoint — to compare whether QIT gives QAT a better starting point, and
whether the quantized model can replicate the FP linear-probe AUROC reported
by run_qit_merlin.py --full_data for the same condition.

Train/val/test splits come from MerlinDataset(label_cols=[--condition],
require_labeled=True), so only scans with a non-missing label for that
condition are used.

Per-epoch log: train_loss, train_auroc, val_loss, val_auroc, phase — written
to results.json. Test AUROC is reported both with fake-quantization
(deployment-realistic) and in FP (same weights, no quant) for comparison.

By default trains for the full phase1+phase2 budget (patience=0, no early
stopping) — val AUROC on ~300 examples for a single condition is too noisy
to pick a reliable "best epoch" early; inspect the full epoch_log curve
instead (see plot_qat_merlin_comparison.py).

Usage:
    # Pretrained baseline
    python scripts/run_qat_merlin.py \\
        --condition renal_cyst --init pretrained \\
        --w_bits 4 --a_bits 4 --phase1_epochs 10 --phase2_epochs 20 \\
        --output_dir runs/qat_merlin_renal_cyst_pretrained_w4a4

    # QIT-initialised
    python scripts/run_qat_merlin.py \\
        --condition renal_cyst --init runs/qit_merlin_fulldata/checkpoint_final.pt \\
        --w_bits 4 --a_bits 4 --phase1_epochs 10 --phase2_epochs 20 \\
        --output_dir runs/qat_merlin_renal_cyst_qit_w4a4
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

_ROOT = Path(__file__).parent.parent
_QFT  = _ROOT.parent / "quantized_ft"
sys.path.insert(0, str(_ROOT))
sys.path.insert(1, str(_QFT))

warnings.filterwarnings("ignore", category=UserWarning)

from models.quantization import quantized_forward
from models.eval_utils import _pbar, MerlinEncoder
from models.merlin_utils import (
    ResizedDataset as _ResizedDataset,
    DEFAULT_DATA_DIR, DEFAULT_REPORTS, DEFAULT_LABELS, DEFAULT_METADATA,
)
from downstream.dataset import MerlinDataset

TARGET_SHAPE = (160, 224, 224)
AMP_DTYPE    = torch.bfloat16
MERLIN_FEAT_DIM = 2048


# ---------------------------------------------------------------------------
# Train / eval
# ---------------------------------------------------------------------------

def _set_phase(model, head, layer4, phase):
    """Phase 1: encoder fully frozen (eval mode), only the head trains.
    Phase 2: also unfreeze the last residual stage (layer4, train mode) so
    its BatchNorm stats can adapt alongside its weights."""
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    if phase == 2:
        for p in layer4.parameters():
            p.requires_grad_(True)
        layer4.train()
    for p in head.parameters():
        p.requires_grad_(True)
    head.train()


def train_epoch(model, head, loader, optimizer, scheduler, grad_accum,
                w_bits, a_bits, weight_granularity, device):
    from sklearn.metrics import roc_auc_score

    total_loss = n = 0
    all_logits, all_labels = [], []
    n_batches = len(loader)

    optimizer.zero_grad()
    for step, (x, y) in enumerate(_pbar(loader, desc="  train", leave=False), start=1):
        x, y = x.to(device), y.to(device)
        with torch.amp.autocast("cuda", dtype=AMP_DTYPE):
            with quantized_forward([model.image_encoder], w_bits, a_bits, weight_granularity):
                feats = model(x)
            logits = head(feats.float())
        loss = F.binary_cross_entropy_with_logits(logits, y)

        (loss / grad_accum).backward()
        if step % grad_accum == 0 or step == n_batches:
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * len(y)
        n          += len(y)
        all_logits.append(logits.detach().float().cpu())
        all_labels.append(y.cpu())

    if scheduler is not None:
        scheduler.step()

    logits = torch.cat(all_logits).squeeze(-1).numpy()
    labels = torch.cat(all_labels).squeeze(-1).numpy()
    try:
        auroc = float(roc_auc_score(labels, logits))
    except ValueError:
        auroc = float("nan")
    return total_loss / n, auroc


@torch.no_grad()
def evaluate(model, head, loader, w_bits, a_bits, weight_granularity, device):
    from sklearn.metrics import roc_auc_score

    model.eval()
    head.eval()
    all_logits, all_labels = [], []
    for x, y in _pbar(loader, desc="  eval ", leave=False):
        x = x.to(device)
        with torch.amp.autocast("cuda", dtype=AMP_DTYPE):
            if w_bits is not None:
                with quantized_forward([model.image_encoder], w_bits, a_bits, weight_granularity):
                    feats = model(x)
            else:
                feats = model(x)
            logits = head(feats.float())
        all_logits.append(logits.cpu())
        all_labels.append(y)

    logits = torch.cat(all_logits).squeeze(-1).float()
    labels = torch.cat(all_labels).squeeze(-1).float()
    loss   = F.binary_cross_entropy_with_logits(logits, labels).item()
    try:
        auroc = float(roc_auc_score(labels.numpy(), logits.numpy()))
    except ValueError:
        auroc = float("nan")
    return loss, auroc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--reports",  default=DEFAULT_REPORTS)
    parser.add_argument("--labels",   default=DEFAULT_LABELS)
    parser.add_argument("--metadata", default=DEFAULT_METADATA)
    parser.add_argument("--condition", required=True,
                        help="Single label column from --labels, e.g. renal_cyst.")
    parser.add_argument("--init", required=True,
                        help="'pretrained' or path to a QIT checkpoint (.pt with 'student' key).")
    parser.add_argument("--w_bits", type=int, default=4)
    parser.add_argument("--a_bits", type=int, default=4)
    parser.add_argument("--phase1_epochs", type=int, default=10,
                        help="Phase 1: encoder fully frozen, train only the linear "
                             "head under fake-quant forward (LP warmup).")
    parser.add_argument("--phase2_epochs", type=int, default=20,
                        help="Phase 2: unfreeze the last residual stage "
                             "(image_encoder.i3_resnet.layer4) + head, continue training.")
    parser.add_argument("--patience", type=int, default=0,
                        help="Early stopping across the whole run (both phases): stop "
                             "if val AUROC does not improve for this many epochs. "
                             "0 = disabled (default; val AUROC on ~300 examples is too "
                             "noisy to early-stop on).")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--lr_schedule", default="cosine", choices=["cosine", "constant"])
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--weight_granularity", default="per_channel",
                        choices=["per_tensor", "per_channel"])
    parser.add_argument("--num_workers", type=int, default=12)
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    device   = torch.device("cuda")
    init_tag = "pretrained" if args.init == "pretrained" else "qit"
    out = Path(args.output_dir or
               f"runs/qat_merlin_{args.condition}_{init_tag}_w{args.w_bits}a{args.a_bits}")
    out.mkdir(parents=True, exist_ok=True)

    print(f"\n{'QAT — Merlin CT':=^70}")
    print(f"  condition  : {args.condition}")
    print(f"  init       : {args.init}")
    print(f"  W{args.w_bits}A{args.a_bits}  weight_gran={args.weight_granularity}")
    print(f"  phase1={args.phase1_epochs}ep (head only)  "
          f"phase2={args.phase2_epochs}ep (+layer4)  patience={args.patience}")
    print(f"  lr={args.lr} ({args.lr_schedule})  batch_size={args.batch_size} "
          f"grad_accum={args.grad_accum} | amp=bf16")
    print(f"  output: {out}\n{'='*70}\n")

    # ── dataset ───────────────────────────────────────────────────────────
    meta_file = args.metadata if Path(args.metadata).exists() else None
    common = dict(
        data_folder=args.data_dir,
        reports_file=args.reports,
        labels_file=args.labels,
        meta_file=meta_file,
        label_cols=[args.condition],
        require_labeled=True,
    )
    train_ds = _ResizedDataset(MerlinDataset(**common, split="train"), TARGET_SHAPE)
    val_ds   = _ResizedDataset(MerlinDataset(**common, split="val"),   TARGET_SHAPE)
    test_ds  = _ResizedDataset(MerlinDataset(**common, split="test"),  TARGET_SHAPE)
    print(f"  train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}\n")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)

    # ── model ─────────────────────────────────────────────────────────────
    print("Loading Merlin...")
    model = MerlinEncoder().to(device)
    if args.init != "pretrained":
        ckpt = torch.load(args.init, map_location=device)
        model.load_state_dict(ckpt["student"])
    head   = nn.Linear(MERLIN_FEAT_DIM, 1).to(device)
    layer4 = model.image_encoder.i3_resnet.layer4

    # ── training ──────────────────────────────────────────────────────────
    print(f"\n{'Training':=^70}")
    print(f"{'Epoch':>6}  {'train_loss':>10}  {'train_auroc':>11}  "
          f"{'val_loss':>9}  {'val_auroc':>9}")

    epoch_log      = []
    best_val_auroc = -1.0
    best_val_epoch = 0
    no_improve     = 0
    epoch          = 0
    stop           = False

    phases = [(p, n, params) for p, n, params in (
        (1, args.phase1_epochs, list(head.parameters())),
        (2, args.phase2_epochs, list(layer4.parameters()) + list(head.parameters())),
    ) if n > 0]

    for phase, n_phase_epochs, params in phases:
        if stop:
            break
        label = "head only" if phase == 1 else "+ layer4"
        print(f"\n--- Phase {phase} ({label}): {n_phase_epochs} epochs ---")

        optimizer = torch.optim.Adam(params, lr=args.lr)
        scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(
                         optimizer, T_max=n_phase_epochs, eta_min=0)
                     if args.lr_schedule == "cosine" else None)

        for _ in range(n_phase_epochs):
            epoch += 1
            _set_phase(model, head, layer4, phase)
            tr_loss, tr_auroc = train_epoch(model, head, train_loader, optimizer, scheduler,
                                            args.grad_accum, args.w_bits, args.a_bits,
                                            args.weight_granularity, device)
            va_loss, va_auroc = evaluate(model, head, val_loader,
                                         args.w_bits, args.a_bits, args.weight_granularity, device)

            improved = va_auroc > best_val_auroc
            marker   = " *" if improved else ""
            if improved:
                best_val_auroc = va_auroc
                best_val_epoch = epoch
                no_improve      = 0
                torch.save({"epoch": epoch, "student": model.state_dict(),
                            "head": head.state_dict()}, out / "checkpoint_best.pt")
            else:
                no_improve += 1

            print(f"{epoch:6d}  {tr_loss:10.4f}  {tr_auroc:11.4f}  "
                  f"{va_loss:9.4f}  {va_auroc:9.4f}{marker}", flush=True)
            epoch_log.append({
                "epoch": epoch, "phase": phase,
                "train_loss": round(tr_loss, 6), "train_auroc": round(tr_auroc, 6),
                "val_loss":   round(va_loss, 6), "val_auroc":   round(va_auroc, 6),
            })

            if args.patience > 0 and no_improve >= args.patience:
                print(f"\n[early stop] val_auroc did not improve for {args.patience} epochs.")
                stop = True
                break

    torch.save({"epoch": epoch, "student": model.state_dict(),
                "head": head.state_dict()}, out / "checkpoint_final.pt")

    # ── test evaluation ───────────────────────────────────────────────────
    print(f"\n{'Test evaluation':=^70}")
    ckpt = torch.load(out / "checkpoint_best.pt", map_location=device)
    model.load_state_dict(ckpt["student"])
    head.load_state_dict(ckpt["head"])

    test_loss_q, test_auroc_q = evaluate(model, head, test_loader,
                                         args.w_bits, args.a_bits, args.weight_granularity, device)
    test_loss_fp, test_auroc_fp = evaluate(model, head, test_loader,
                                           None, None, args.weight_granularity, device)
    print(f"  Test AUROC (best val ckpt, W{args.w_bits}A{args.a_bits}): {test_auroc_q:.4f}")
    print(f"  Test AUROC (best val ckpt, FP)         : {test_auroc_fp:.4f}")

    # ── save ──────────────────────────────────────────────────────────────
    out_path = out / "results.json"
    with open(out_path, "w") as f:
        json.dump({
            "script":             "run_qat_merlin",
            "condition":          args.condition,
            "init":               args.init,
            "w_bits":             args.w_bits,
            "a_bits":             args.a_bits,
            "phase1_epochs":      args.phase1_epochs,
            "phase2_epochs":      args.phase2_epochs,
            "patience":           args.patience,
            "lr":                 args.lr,
            "lr_schedule":        args.lr_schedule,
            "batch_size":         args.batch_size,
            "grad_accum":         args.grad_accum,
            "weight_granularity": args.weight_granularity,
            "n_train":            len(train_ds),
            "n_val":              len(val_ds),
            "n_test":             len(test_ds),
            "best_val_auroc":     round(best_val_auroc, 6),
            "best_val_epoch":     best_val_epoch,
            "test_auroc":         {"quantized": round(test_auroc_q, 6), "fp": round(test_auroc_fp, 6)},
            "test_loss":          {"quantized": round(test_loss_q, 6), "fp": round(test_loss_fp, 6)},
            "epoch_log":          epoch_log,
        }, f, indent=2)
    print(f"\nResults → {out_path}")


if __name__ == "__main__":
    main()
