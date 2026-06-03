"""
QIT (Quantization Invariance Training) for Merlin CT foundation model.

Teacher : frozen Merlin(ImageEmbedding=True)
Student : Merlin(ImageEmbedding=True) trained to match teacher features under
          random fake quantization (straight-through estimator).

Loss   : mean cosine distance between student and teacher 512D feature vectors.
Monitor: per-epoch QIT cosine loss on train and validation sets.
Eval   : mean AUROC over 30 disease conditions (multi-label linear probe on
         student features — FP and key quantized configs — vs. teacher FP).

Dataset: MerlinDataset (70/15/15 train/val/test split) from quantized_ft.
Labels : zero_shot_findings_disease_cls.csv  — 30 binary conditions
         Values: 1=positive, 0=negative, -1=unknown (masked during probe training).

Usage
-----
  python scripts/run_qit_merlin.py \\
      --output_dir runs/qit_merlin_w2-8a4-8

All data paths default to ../CT-CLIP/data/ relative to this project root.
Override with --data_dir, --reports, --labels, --metadata as needed.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

_ROOT = Path(__file__).parent.parent
_QFT  = _ROOT.parent / "quantized_ft"
sys.path.insert(0, str(_ROOT))   # qit/models has priority over quantized_ft/models
sys.path.insert(1, str(_QFT))

from models.quantization import quantized_forward, sample_bits
from models.eval_utils import _pbar, MerlinEncoder
from downstream.dataset import MerlinDataset


# ---------------------------------------------------------------------------
# Default data paths (relative to project root)
# ---------------------------------------------------------------------------

_CT_DATA = _ROOT.parent / "CT-CLIP" / "data"
_DEFAULT_DATA_DIR = str(_CT_DATA / "merlin_data")
_DEFAULT_REPORTS  = str(_CT_DATA / "reports_final.xlsx")
_DEFAULT_LABELS   = str(_CT_DATA / "zero_shot_findings_disease_cls.csv")
_DEFAULT_METADATA = str(_CT_DATA / "metadata.csv")

_EVAL_CONFIGS = [
    ("fp",   None, None),
    ("w8a8", 8,    8),
    ("w4a8", 4,    8),
    ("w4a4", 4,    4),
]


# ---------------------------------------------------------------------------
# Dataset wrappers
# ---------------------------------------------------------------------------

class _IndexedDataset(torch.utils.data.Dataset):
    """Wraps any dataset to return (index, *original_items)."""
    def __init__(self, ds):
        self._ds = ds
    def __len__(self):
        return len(self._ds)
    def __getitem__(self, i):
        x, y = self._ds[i]
        return i, x, y


class _ResizedDataset(torch.utils.data.Dataset):
    """Trilinearly resizes CT volumes to a fixed (D, H, W) on load.

    Used to downsample from the quantized_ft default of (240, 480, 480) to the
    original Merlin training resolution of (160, 224, 224), reducing volume size
    ~7× and allowing much larger batch sizes.
    """
    def __init__(self, ds, target_shape: tuple):
        self._ds = ds
        self._target = tuple(target_shape)  # (D, H, W)

    def __len__(self):
        return len(self._ds)

    @property
    def label_names(self):
        return self._ds.label_names

    def __getitem__(self, i):
        x, y = self._ds[i]               # x: (1, D, H, W)
        if tuple(x.shape[1:]) != self._target:
            x = F.interpolate(
                x.unsqueeze(0),          # (1, 1, D, H, W)
                size=self._target,
                mode="trilinear",
                align_corners=False,
            ).squeeze(0)                 # (1, D, H, W)
        return x, y


# ---------------------------------------------------------------------------
# QIT helpers
# ---------------------------------------------------------------------------

def _qit_loss(student_f: torch.Tensor, teacher_f: torch.Tensor) -> torch.Tensor:
    """Mean cosine distance: 1 − cos(student, teacher)."""
    s = F.normalize(student_f.float(), dim=-1)
    t = F.normalize(teacher_f.float(), dim=-1)
    return (1.0 - (s * t).sum(-1)).mean()


@torch.no_grad()
def _fp_forward(model: nn.Module, x: torch.Tensor, use_amp: bool,
                amp_dtype: torch.dtype = torch.float16) -> torch.Tensor:
    if use_amp:
        with torch.amp.autocast("cuda", dtype=amp_dtype):
            return model(x).float()
    return model(x)


@torch.no_grad()
def cache_teacher_features(teacher: MerlinEncoder, loader: DataLoader,
                            device: torch.device, use_amp: bool,
                            amp_dtype: torch.dtype = torch.float16) -> dict:
    """Cache teacher (FP) features for all indexed training samples."""
    teacher.eval()
    cache = {}
    for batch in _pbar(loader, desc="caching teacher features"):
        idx, x, _ = batch[0], batch[1], batch[2]
        f = _fp_forward(teacher, x.to(device), use_amp, amp_dtype)
        for i, fi in zip(idx.tolist(), f.unbind(0)):
            cache[i] = fi.cpu()
    return cache


# ---------------------------------------------------------------------------
# Train / validate one epoch
# ---------------------------------------------------------------------------

def train_epoch(student, teacher_cache, loader, optimizer, scheduler,
                w_bits_range, a_bits_range, device, use_amp, amp_dtype, grad_accum, scaler):
    student.train()
    # Re-freeze BN after .train() — .train() re-enables BN update mode on all submodules
    for m in student.modules():
        if isinstance(m, (nn.BatchNorm3d, nn.SyncBatchNorm)):
            m.eval()
    total_loss = n_samples = step = 0
    optimizer.zero_grad()

    for batch in loader:
        idx, x, _ = batch[0], batch[1], batch[2]
        x = x.to(device)
        w_bits, a_bits = sample_bits(w_bits_range, a_bits_range)

        if use_amp:
            with torch.amp.autocast("cuda", dtype=amp_dtype):
                with quantized_forward([student.image_encoder], w_bits, a_bits, "per_channel"):
                    s_f = student(x)
        else:
            with quantized_forward([student.image_encoder], w_bits, a_bits, "per_channel"):
                s_f = student(x)

        t_f  = torch.stack([teacher_cache[i] for i in idx.tolist()]).to(device)
        loss = _qit_loss(s_f, t_f) / grad_accum

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        total_loss += loss.item() * grad_accum * len(x)
        n_samples  += len(x)
        step       += 1

        if step % grad_accum == 0:
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

    if step % grad_accum != 0:
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad()

    if scheduler is not None:
        scheduler.step()

    return total_loss / max(n_samples, 1)


@torch.no_grad()
def validate_epoch(student, teacher, val_loader, device, use_amp, amp_dtype, val_configs):
    """Deterministic validation cycling through val_configs.

    For each batch, a config from val_configs is selected by index (round-robin), so
    over a full pass every config is exercised roughly equally regardless of val-set size.

    Returns (mean_qit_loss, mean_fp_sim) where:
      - mean_qit_loss  : mean cosine distance between student (at cycled quant config) and teacher FP
      - mean_fp_sim    : mean cosine similarity between student FP features and teacher FP features
    """
    student.eval()
    teacher.eval()
    total_qit_loss = total_fp_sim = n_batches = 0

    for i, batch in enumerate(val_loader):
        _, x, _ = batch[0], batch[1], batch[2]
        x = x.to(device)
        _, w_bits, a_bits = val_configs[i % len(val_configs)]

        t_f  = _fp_forward(teacher, x, use_amp, amp_dtype)
        s_fp = _fp_forward(student,  x, use_amp, amp_dtype)

        if w_bits is not None:
            if use_amp:
                with torch.amp.autocast("cuda", dtype=amp_dtype):
                    with quantized_forward([student.image_encoder], w_bits, a_bits, "per_channel"):
                        s_q = student(x)
            else:
                with quantized_forward([student.image_encoder], w_bits, a_bits, "per_channel"):
                    s_q = student(x)
        else:
            s_q = s_fp

        total_qit_loss += _qit_loss(s_q, t_f).item()
        total_fp_sim   += F.cosine_similarity(
            F.normalize(s_fp.float(), dim=-1),
            F.normalize(t_f.float(),  dim=-1),
        ).mean().item()
        n_batches += 1

    return total_qit_loss / max(n_batches, 1), total_fp_sim / max(n_batches, 1)


# ---------------------------------------------------------------------------
# Feature extraction & AUROC probe
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_features(model, loader, device, use_amp, amp_dtype,
                     w_bits=None, a_bits=None):
    """Returns (features_np, labels_np). Quantizes if w_bits/a_bits given."""
    model.eval()
    all_f, all_y = [], []

    for batch in _pbar(loader, desc="    extracting", leave=False):
        x = batch[0].to(device) if len(batch) == 2 else batch[1].to(device)
        y = batch[1]             if len(batch) == 2 else batch[2]

        if use_amp:
            with torch.amp.autocast("cuda", dtype=amp_dtype):
                if w_bits is not None:
                    with quantized_forward([model.image_encoder], w_bits, a_bits, "per_channel"):
                        f = model(x)
                else:
                    f = model(x)
        else:
            if w_bits is not None:
                with quantized_forward([model.image_encoder], w_bits, a_bits, "per_channel"):
                    f = model(x)
            else:
                f = model(x)

        all_f.append(f.cpu().float())
        all_y.append(y)

    return torch.cat(all_f).numpy(), torch.cat(all_y).numpy()


def run_auroc_probe(train_feats, train_labels, test_feats, test_labels,
                    label_names, device, epochs=300, lr=1e-3, batch_size=256):
    """Masked-BCE linear probe for multi-label AUROC.

    Labels: 1=positive, 0=negative, -1=unknown (masked out during training).
    Returns (mean_auroc, per_condition_auroc_dict).
    """
    from sklearn.metrics import roc_auc_score

    X_tr = torch.tensor(train_feats, dtype=torch.float32)
    y_tr = torch.tensor(np.clip(train_labels, 0, 1).astype(np.float32))
    m_tr = torch.tensor((train_labels >= 0).astype(np.float32))
    X_te = torch.tensor(test_feats,  dtype=torch.float32)

    mu  = X_tr.mean(0)
    std = X_tr.std(0).clamp(min=1e-8)
    X_tr = (X_tr - mu) / std
    X_te = (X_te - mu) / std

    n_cond = y_tr.shape[1]
    head   = nn.Linear(X_tr.shape[1], n_cond).to(device)
    opt    = torch.optim.Adam(head.parameters(), lr=lr)
    sch    = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=0)

    ds     = torch.utils.data.TensorDataset(X_tr.to(device), y_tr.to(device), m_tr.to(device))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    best_loss, no_imp = float("inf"), 0
    for _ in range(epochs):
        head.train()
        ep_loss = 0.0
        for xb, yb, mb in loader:
            opt.zero_grad()
            raw  = F.binary_cross_entropy_with_logits(head(xb), yb, reduction="none")
            loss = (raw * mb).sum() / mb.sum().clamp(min=1)
            loss.backward()
            opt.step()
            ep_loss += loss.item()
        sch.step()
        ep_loss /= len(loader)
        if best_loss - ep_loss > 1e-5:
            best_loss, no_imp = ep_loss, 0
        else:
            no_imp += 1
        if no_imp >= 30:
            break

    head.eval()
    with torch.no_grad():
        preds = torch.sigmoid(head(X_te.to(device))).cpu().numpy()

    per_cond = {}
    for c, name in enumerate(label_names):
        mask = test_labels[:, c] >= 0
        if mask.sum() < 10:
            continue
        y_true = test_labels[mask, c]
        y_pred = preds[mask, c]
        if y_true.max() == y_true.min():
            continue
        try:
            per_cond[name] = float(roc_auc_score(y_true, y_pred))
        except Exception:
            pass

    mean_auroc = float(np.mean(list(per_cond.values()))) if per_cond else float("nan")
    return mean_auroc, per_cond


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    # Data
    parser.add_argument("--data_dir",   default=_DEFAULT_DATA_DIR)
    parser.add_argument("--reports",    default=_DEFAULT_REPORTS,
                        help="XLSX/CSV with report accession IDs (used to whitelist scans).")
    parser.add_argument("--labels",     default=_DEFAULT_LABELS,
                        help="CSV with 30 binary disease condition labels.")
    parser.add_argument("--metadata",    default=_DEFAULT_METADATA,
                        help="CSV with per-scan spacing metadata (optional, improves preprocessing).")
    parser.add_argument("--patience",    type=int, default=10,
                        help="Early stopping: stop if val QIT loss does not improve for "
                             "this many validation checks. 0 = disabled.")
    parser.add_argument("--n_train",     type=int, default=None,
                        help="Cap training set size (random subset, seed=42). "
                             "Default: use full split.")
    parser.add_argument("--n_val",       type=int, default=None,
                        help="Cap validation set size (random subset, seed=42). "
                             "Test set is always kept full.")
    parser.add_argument("--conditions",  nargs="+",
                        default=["atelectasis", "pleural_effusion", "renal_cyst"],
                        help="Label columns to train/evaluate on. Defaults to the three "
                             "conditions with the best label coverage in this dataset.")
    # Training
    parser.add_argument("--epochs",          type=int,   default=30)
    parser.add_argument("--batch_size",      type=int,   default=16)
    parser.add_argument("--grad_accum",      type=int,   default=2,
                        help="Gradient accumulation steps. Effective batch = batch_size × grad_accum.")
    parser.add_argument("--lr",              type=float, default=1e-5,
                        help="Lower than standard backbones — Merlin is a large pretrained model.")
    parser.add_argument("--w_bits_min",      type=int,   default=4)
    parser.add_argument("--w_bits_max",      type=int,   default=8)
    parser.add_argument("--a_bits_min",      type=int,   default=4)
    parser.add_argument("--a_bits_max",      type=int,   default=8)
    parser.add_argument("--val_every",       type=int,   default=1,
                        help="Compute val metrics every N epochs. "
                             "Validation cycles through all eval configs deterministically.")
    # System
    parser.add_argument("--num_workers",          type=int,   default=4)
    parser.add_argument("--output_dir",            default=None)
    parser.add_argument("--no_amp",                action="store_true")
    parser.add_argument("--amp_dtype",  default="fp16", choices=["fp16", "bf16"],
                        help="AMP dtype. fp16 matches the original Merlin training; "
                             "bf16 is more numerically stable but slower on some hardware.")
    parser.add_argument("--target_shape", type=int, nargs=3, default=None,
                        metavar=("D", "H", "W"),
                        help="Resize volumes to (D H W) after loading via trilinear interpolation. "
                             "Use '160 224 224' to match original Merlin training resolution and "
                             "reduce memory ~7× vs the default (240 480 480). Default: no resize.")
    args = parser.parse_args()

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = not args.no_amp and device.type == "cuda"
    amp_dtype = torch.float16 if args.amp_dtype == "fp16" else torch.bfloat16

    out = Path(args.output_dir or
               f"runs/qit_merlin_w{args.w_bits_min}-{args.w_bits_max}"
               f"a{args.a_bits_min}-{args.a_bits_max}")
    out.mkdir(parents=True, exist_ok=True)

    meta_file = args.metadata if Path(args.metadata).exists() else None

    print("=" * 70)
    print(f"QIT — Merlin CT  W{args.w_bits_min}-{args.w_bits_max}"
          f" A{args.a_bits_min}-{args.a_bits_max}")
    print(f"  data_dir  : {args.data_dir}")
    print(f"  epochs    : {args.epochs}  lr={args.lr}  bs={args.batch_size}"
          f"  grad_accum={args.grad_accum}")
    _cfg_str = ", ".join(n for n, *_ in _EVAL_CONFIGS)
    print(f"  val_every : {args.val_every}  val_configs=[{_cfg_str}] (cycled deterministically)")
    print(f"  amp       : {use_amp} ({args.amp_dtype})")
    _shape_str = "×".join(map(str, args.target_shape)) if args.target_shape else "native (240×480×480)"
    print(f"  vol_shape : {_shape_str}")
    print(f"  output    : {out}")
    print("=" * 70)

    # ── datasets ──────────────────────────────────────────────────────────
    print("\nIndexing dataset...")
    common = dict(data_folder=args.data_dir, reports_file=args.reports,
                  labels_file=args.labels, meta_file=meta_file,
                  label_cols=args.conditions)
    train_ds = MerlinDataset(**common, split="train")
    val_ds   = MerlinDataset(**common, split="val")
    test_ds  = MerlinDataset(**common, split="test")
    label_names = list(train_ds.label_names)

    rng = np.random.default_rng(42)
    if args.n_train is not None and args.n_train < len(train_ds):
        idx = rng.choice(len(train_ds), args.n_train, replace=False).tolist()
        train_ds = torch.utils.data.Subset(train_ds, sorted(idx))
    if args.n_val is not None and args.n_val < len(val_ds):
        idx = rng.choice(len(val_ds), args.n_val, replace=False).tolist()
        val_ds = torch.utils.data.Subset(val_ds, sorted(idx))

    if args.target_shape is not None:
        target = tuple(args.target_shape)
        train_ds = _ResizedDataset(train_ds, target)
        val_ds   = _ResizedDataset(val_ds,   target)
        test_ds  = _ResizedDataset(test_ds,  target)

    print(f"  train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}")
    print(f"  conditions ({len(label_names)}): {', '.join(label_names)}\n")

    pin = device.type == "cuda"
    train_loader = DataLoader(_IndexedDataset(train_ds), batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers,
                              pin_memory=pin, persistent_workers=args.num_workers > 0)
    val_loader   = DataLoader(_IndexedDataset(val_ds),   batch_size=args.batch_size,
                              shuffle=False, num_workers=min(2, args.num_workers),
                              pin_memory=pin,
                              persistent_workers=args.num_workers > 0)

    # ── models ────────────────────────────────────────────────────────────
    print("Loading Merlin models (teacher + student)...")
    teacher = MerlinEncoder().to(device)
    for p in teacher.parameters():
        p.requires_grad_(False)

    student = MerlinEncoder().to(device)
    # Freeze BatchNorm: prevents (1) running-stat corruption from the double-update
    # caused by gradient checkpointing recompute, and (2) small-batch degeneracy.
    # Pre-trained stats from Merlin are already well-calibrated.
    _bn_count = 0
    for m in student.modules():
        if isinstance(m, (nn.BatchNorm3d, nn.SyncBatchNorm)):
            m.eval()
            m.weight.requires_grad_(False)
            m.bias.requires_grad_(False)
            _bn_count += 1

    print(f"  encoder params: {sum(p.numel() for p in student.image_encoder.parameters()):,}")
    print(f"  BatchNorm layers frozen: {_bn_count} (running stats fixed, no grad-checkpoint double-update)")
    print("  gradient checkpointing: always-on (built into I3ResNet)")

    # ── cache teacher features ────────────────────────────────────────────
    print("\nCaching teacher features for training set...")
    teacher_cache = cache_teacher_features(teacher, train_loader, device, use_amp, amp_dtype)
    print(f"  cached {len(teacher_cache)} samples ({len(teacher_cache) * 2048 * 4 / 1e6:.1f} MB)\n")

    # ── optimizer / scheduler / scaler ───────────────────────────────────
    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr,
                                   betas=(0.9, 0.999), weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
    # GradScaler only needed for FP16 (BF16 has sufficient dynamic range without it)
    scaler = (torch.amp.GradScaler("cuda")
              if use_amp and amp_dtype == torch.float16 else None)

    # ── training ──────────────────────────────────────────────────────────
    print(f"\n{'Training':=^70}")
    print(f"{'Epoch':>6}  {'train_loss':>12}  {'val_loss':>10}  {'val_fp_sim':>12}")

    epoch_log      = []
    best_val_loss  = float("inf")
    best_val_epoch = 0
    no_improve     = 0

    for epoch in range(1, args.epochs + 1):
        tr_loss = train_epoch(
            student, teacher_cache, train_loader, optimizer, scheduler,
            (args.w_bits_min, args.w_bits_max), (args.a_bits_min, args.a_bits_max),
            device, use_amp, amp_dtype, args.grad_accum, scaler,
        )

        val_loss = val_fp_sim = None
        saved    = ""
        if epoch % args.val_every == 0 or epoch == args.epochs:
            val_loss, val_fp_sim = validate_epoch(
                student, teacher, val_loader, device, use_amp, amp_dtype, _EVAL_CONFIGS)
            if val_loss < best_val_loss:
                best_val_loss  = val_loss
                best_val_epoch = epoch
                no_improve     = 0
                torch.save({"epoch": epoch, "student": student.state_dict()},
                           out / "checkpoint_best.pt")
                saved = "*"
            else:
                no_improve += 1

        val_str    = f"{val_loss:10.6f}"   if val_loss    is not None else f"{'—':>10}"
        fp_sim_str = f"{val_fp_sim:12.4f}" if val_fp_sim  is not None else f"{'—':>12}"
        marker     = " *" if saved else ""
        print(f"{epoch:6d}  {tr_loss:12.6f}  {val_str}  {fp_sim_str}{marker}", flush=True)

        epoch_log.append({
            "epoch":      epoch,
            "train_loss": round(tr_loss, 7),
            "val_loss":   round(val_loss,   7) if val_loss   is not None else None,
            "val_fp_sim": round(val_fp_sim, 4) if val_fp_sim is not None else None,
        })

        if args.patience > 0 and no_improve >= args.patience:
            print(f"\n[early stop] val loss did not improve for "
                  f"{args.patience} validation checks (every {args.val_every} epochs).")
            break

    torch.save({"epoch": args.epochs, "student": student.state_dict()},
               out / "checkpoint_final.pt")

    # ── test evaluation ───────────────────────────────────────────────────
    print(f"\n{'Test evaluation':=^70}")
    ckpt = torch.load(out / "checkpoint_best.pt", map_location=device)
    student.load_state_dict(ckpt["student"])

    eval_loader_kw = dict(batch_size=args.batch_size, shuffle=False,
                          num_workers=2, pin_memory=device.type == "cuda")
    train_eval_loader = DataLoader(train_ds, **eval_loader_kw)
    test_loader       = DataLoader(test_ds,  **eval_loader_kw)

    auroc_results = {}

    # Student at every quant config
    for cfg_name, wb, ab in _EVAL_CONFIGS:
        print(f"\n  [student {cfg_name}] extracting features...")
        tr_f, tr_l = extract_features(student, train_eval_loader, device, use_amp, amp_dtype, wb, ab)
        te_f, te_l = extract_features(student, test_loader,       device, use_amp, amp_dtype, wb, ab)
        mean_auc, per_cond = run_auroc_probe(tr_f, tr_l, te_f, te_l, label_names, device)
        auroc_results[f"student_{cfg_name}"] = {"mean_auroc": round(mean_auc, 4), "per_condition": per_cond}
        print(f"    mean AUROC = {mean_auc:.4f}")

    # Teacher at every quant config (direct comparison baseline)
    for cfg_name, wb, ab in _EVAL_CONFIGS:
        print(f"\n  [teacher {cfg_name}] extracting features...")
        tr_f, tr_l = extract_features(teacher, train_eval_loader, device, use_amp, amp_dtype, wb, ab)
        te_f, te_l = extract_features(teacher, test_loader,       device, use_amp, amp_dtype, wb, ab)
        mean_auc, per_cond = run_auroc_probe(tr_f, tr_l, te_f, te_l, label_names, device)
        auroc_results[f"teacher_{cfg_name}"] = {"mean_auroc": round(mean_auc, 4), "per_condition": per_cond}
        print(f"    mean AUROC = {mean_auc:.4f}")

    # ── save ──────────────────────────────────────────────────────────────
    results = {
        "model":         "merlin",
        "w_bits_range":  [args.w_bits_min, args.w_bits_max],
        "a_bits_range":  [args.a_bits_min, args.a_bits_max],
        "epochs":        args.epochs,
        "lr":            args.lr,
        "batch_size":    args.batch_size,
        "grad_accum":    args.grad_accum,
        "n_train":       len(train_ds),
        "n_val":         len(val_ds),
        "n_test":        len(test_ds),
        "label_names":   label_names,
        "patience":       args.patience,
        "best_val_epoch": best_val_epoch,
        "best_val_loss":  round(best_val_loss, 7),
        "auroc":          auroc_results,
        "epoch_log":      epoch_log,
    }
    out_path = out / "results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults → {out_path}")


if __name__ == "__main__":
    main()
