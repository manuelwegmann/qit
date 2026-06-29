"""
QIT (Quantization Invariance Training) for Merlin CT foundation model.

Teacher : frozen Merlin(ImageEmbedding=True)
Student : Merlin(ImageEmbedding=True) trained to match teacher features under
          random fake quantization (straight-through estimator).

Loss   : mean cosine distance between student and teacher 2048-D feature vectors.
Output : QIT checkpoints only. This script trains and saves checkpoints; it does
         NOT probe. Downstream evaluation is done separately and more carefully
         via cache_merlin_features.py (static percentile-calibrated features) →
         run_probe_merlin.py (per-condition early-stopping AUROC probe).

Two training regimes (select with --full_data):
  default      : 70/15 train/val split, per-epoch validation on the quantized
                 bit-width grid, moving-average early stopping, saves
                 checkpoint_best.pt + checkpoint_final.pt.
  --full_data  : train on the combined train+val split, no validation loop and
                 no early stopping, saves a checkpoint every epoch plus
                 checkpoint_final.pt. Use for final full-data runs.

Dataset: MerlinDataset (70/15/15 train/val/test split) from quantized_ft.

Usage
-----
  # train/val + early stopping
  python scripts/run_qit_merlin.py --output_dir runs/qit_merlin_w4-8a4-8

  # full-data run (combined train+val, fixed epochs)
  python scripts/run_qit_merlin.py --full_data --epochs 5 --grad_accum 4 \\
      --amp_dtype bf16 --target_shape 160 224 224 \\
      --output_dir runs/qit_merlin_fulldata

All data paths default to ../CT-CLIP/data/ relative to this project root.
Override with --data_dir, --reports, --labels, --metadata as needed.
"""

import argparse
import itertools
import json
import sys
import warnings
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader

_ROOT = Path(__file__).parent.parent
_QFT  = _ROOT.parent / "quantized_ft"
sys.path.insert(0, str(_ROOT))   # qit/models has priority over quantized_ft/models
sys.path.insert(1, str(_QFT))

from models.quantization import quantized_forward, sample_bits
from models.eval_utils import _pbar, MerlinEncoder
from models.merlin_utils import (
    IndexedDataset as _IndexedDataset,
    ResizedDataset as _ResizedDataset,
    DEFAULT_DATA_DIR as _DEFAULT_DATA_DIR,
    DEFAULT_REPORTS as _DEFAULT_REPORTS,
    DEFAULT_LABELS as _DEFAULT_LABELS,
    DEFAULT_METADATA as _DEFAULT_METADATA,
)
from downstream.dataset import MerlinDataset

warnings.filterwarnings("ignore", category=UserWarning)


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
    """Compute teacher (FP) features for all indexed samples in loader."""
    teacher.eval()
    cache = {}
    for batch in _pbar(loader, desc="caching teacher features"):
        idx, x, _ = batch[0], batch[1], batch[2]
        f = _fp_forward(teacher, x.to(device), use_amp, amp_dtype)
        for i, fi in zip(idx.tolist(), f.unbind(0)):
            cache[i] = fi.cpu()
    return cache


def _unwrap_subset_indices(ds):
    """If ds is a (possibly _ResizedDataset-wrapped) torch.utils.data.Subset,
    return its .indices into the underlying full-split dataset; else None."""
    if isinstance(ds, _ResizedDataset):
        ds = ds._ds
    if isinstance(ds, torch.utils.data.Subset):
        return list(ds.indices)
    return None


def load_or_compute_teacher_cache(teacher, loader, device, use_amp, amp_dtype,
                                   cache_dir: Path, split: str,
                                   n_samples: int, target_shape,
                                   full_cache_dir: Path = None,
                                   subset_indices=None) -> dict:
    """Load teacher feature cache from disk if available, otherwise compute and save it.

    Cache files are keyed by split, sample count and volume shape so different
    experimental configs that share these parameters reuse the same cache.

    If full_cache_dir is given and contains a full-split cache (e.g. produced by
    a previous --full_data run, keyed by original dataset index), derive this
    run's cache from it instead of recomputing — remapping via subset_indices
    when this run uses a subsampled split.
    """
    shape_str = "x".join(map(str, target_shape)) if target_shape else "native"
    cache_path = cache_dir / f"teacher_{split}_n{n_samples}_{shape_str}.pt"

    if cache_path.exists():
        print(f"  {split}: loading cached features from {cache_path}", flush=True)
        cache = torch.load(cache_path, map_location="cpu", weights_only=True)
        print(f"  {split}: loaded {len(cache)} cached features", flush=True)
        return cache

    if full_cache_dir is not None:
        full_matches = list(full_cache_dir.glob(f"{split}_n*_{shape_str}.pt"))
        if full_matches:
            full_path  = max(full_matches, key=lambda p: int(p.stem.split("_n")[1].split("_")[0]))
            full_cache = torch.load(full_path, map_location="cpu", weights_only=True)
            if subset_indices is not None:
                cache = {i: full_cache[orig_i] for i, orig_i in enumerate(subset_indices)}
            elif len(full_cache) == n_samples:
                cache = full_cache
            else:
                cache = None
            if cache is not None:
                cache_dir.mkdir(parents=True, exist_ok=True)
                torch.save(cache, cache_path)
                print(f"  {split}: derived {len(cache)} samples from {full_path}", flush=True)
                return cache

    print(f"  {split}: computing teacher features ({n_samples} samples, "
          f"will save to {cache_path})...", flush=True)
    cache = cache_teacher_features(teacher, loader, device, use_amp, amp_dtype)
    cache_dir.mkdir(parents=True, exist_ok=True)
    torch.save(cache, cache_path)
    print(f"  {split}: computing teacher features done "
          f"({len(cache)} samples, {len(cache) * 2048 * 4 / 1e6:.1f} MB)", flush=True)
    return cache


# ---------------------------------------------------------------------------
# Train / validate one epoch
# ---------------------------------------------------------------------------

def train_epoch(student, teacher_cache, loader, optimizer, scheduler,
                w_bits_range, a_bits_range, device, use_amp, amp_dtype, grad_accum, scaler,
                freeze_bn=False, needs_input_grad=False):
    student.train()
    if freeze_bn:
        # .train() re-enables BN update mode on all submodules; re-freeze immediately.
        for m in student.modules():
            if isinstance(m, nn.modules.batchnorm._BatchNorm):
                m.eval()
    total_loss = n_samples = step = 0
    optimizer.zero_grad()

    for batch in loader:
        idx, x, _ = batch[0], batch[1], batch[2]
        x = x.to(device)
        if needs_input_grad:
            # Gradient checkpointing (use_reentrant=True) only builds the autograd
            # graph when at least one input requires grad. With frozen early layers,
            # x has no grad — marking it here ensures the graph is built for the
            # trainable later layers. The grad wrt x is computed but never used.
            x = x.requires_grad_(True)
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
def validate_epoch(student, teacher_val_cache, val_loader, device, use_amp, amp_dtype, val_quant_configs):
    """Deterministic validation cycling through val_quant_configs.

    val_quant_configs is a list of (w_bits, a_bits) int tuples covering the full
    bit-width grid.  Each batch is assigned a config by round-robin so every config
    is exercised roughly equally regardless of val-set size.  No fp config is included:
    the val_loss is a pure measure of quantized robustness, avoiding the near-zero fp
    contribution that would dilute and destabilise the metric.

    Teacher features are read from teacher_val_cache (pre-computed once at startup)
    rather than recomputed each epoch.

    Returns (mean_qit_loss, mean_fp_sim) where:
      - mean_qit_loss : mean cosine distance between student (at cycled quant config) and teacher FP
      - mean_fp_sim   : mean cosine similarity between student FP features and teacher FP features
    """
    student.eval()
    total_qit_loss = total_fp_sim = n_batches = 0

    for i, batch in enumerate(val_loader):
        idx, x, _ = batch[0], batch[1], batch[2]
        x   = x.to(device)
        t_f = torch.stack([teacher_val_cache[i] for i in idx.tolist()]).to(device)
        w_bits, a_bits = val_quant_configs[i % len(val_quant_configs)]

        s_fp = _fp_forward(student, x, use_amp, amp_dtype)

        if use_amp:
            with torch.amp.autocast("cuda", dtype=amp_dtype):
                with quantized_forward([student.image_encoder], w_bits, a_bits, "per_channel"):
                    s_q = student(x)
        else:
            with quantized_forward([student.image_encoder], w_bits, a_bits, "per_channel"):
                s_q = student(x)

        total_qit_loss += _qit_loss(s_q, t_f).item()
        total_fp_sim   += F.cosine_similarity(
            F.normalize(s_fp.float(), dim=-1),
            F.normalize(t_f.float(),  dim=-1),
        ).mean().item()
        n_batches += 1

    return total_qit_loss / max(n_batches, 1), total_fp_sim / max(n_batches, 1)


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
    parser.add_argument("--conditions",  nargs="+",
                        default=["atelectasis", "pleural_effusion", "renal_cyst"],
                        help="Label columns passed to MerlinDataset. Only affects which "
                             "labels are loaded; QIT training itself is label-free.")
    parser.add_argument("--full_data",  action="store_true",
                        help="Train on the combined train+val split with no validation "
                             "loop or early stopping (a checkpoint is saved every epoch). "
                             "Default: train/val split with moving-average early stopping.")
    parser.add_argument("--n_train",     type=int, default=None,
                        help="Cap training set size (random subset, seed=42). "
                             "Default: use full split.")
    parser.add_argument("--n_val",       type=int, default=None,
                        help="Cap validation set size (random subset, seed=42).")
    # Early stopping (default mode only)
    parser.add_argument("--patience",    type=int, default=10,
                        help="Early stopping: stop if moving-avg val loss does not improve for "
                             "this many validation checks. 0 = disabled. Ignored with --full_data.")
    parser.add_argument("--val_window",  type=int, default=3,
                        help="Window size for moving-average val loss used in early stopping "
                             "and best-checkpoint selection. Ignored with --full_data.")
    parser.add_argument("--val_every",   type=int, default=1,
                        help="Compute val metrics every N epochs. Ignored with --full_data.")
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
    # System
    parser.add_argument("--num_workers",          type=int,   default=4)
    parser.add_argument("--output_dir",            default=None)
    parser.add_argument("--teacher_cache_dir",      default=None,
                        help="Directory for persistent teacher feature caches. "
                             "Defaults to <data_dir>/teacher_cache. "
                             "Shared across runs with the same data config.")
    parser.add_argument("--freeze_layers",           type=int, default=0,
                        choices=[0, 1, 2, 3, 4],
                        help="Freeze the first N layer groups of I3ResNet for gradient updates "
                             "(0=none, 1=stem, 2=+layer1, 3=+layer2, 4=+layer3). "
                             "Frozen layers are still quantized during forward passes.")
    parser.add_argument("--freeze_bn",              action="store_true",
                        help="Freeze BatchNorm layers throughout training. Off by default — "
                             "empirically causes divergence on Merlin (ImageNet 2D stats "
                             "don't hold under QIT on 3D CT data).")
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
    print(f"  mode      : {'full-data (train+val, no early stopping)' if args.full_data else 'train/val + early stopping'}")
    print(f"  data_dir  : {args.data_dir}")
    print(f"  epochs    : {args.epochs}  lr={args.lr}  bs={args.batch_size}"
          f"  grad_accum={args.grad_accum}")
    if not args.full_data:
        _val_quant_configs = list(itertools.product(
            range(args.w_bits_min, args.w_bits_max + 1),
            range(args.a_bits_min, args.a_bits_max + 1),
        ))
        print(f"  val_every : {args.val_every}  val_configs=full integer grid "
              f"w=[{args.w_bits_min},{args.w_bits_max}] a=[{args.a_bits_min},{args.a_bits_max}] "
              f"({len(_val_quant_configs)} configs, cycled deterministically)")
        print(f"  val_window: {args.val_window} epochs (moving-avg for early stopping)")
    print(f"  amp       : {use_amp} ({args.amp_dtype})")
    _shape_str = "×".join(map(str, args.target_shape)) if args.target_shape else "native (240×480×480)"
    print(f"  vol_shape : {_shape_str}")
    print(f"  output    : {out}")
    print("  (probing is done separately: cache_merlin_features.py → run_probe_merlin.py)")
    print("=" * 70)

    # ── datasets ──────────────────────────────────────────────────────────
    print("\nIndexing dataset...")
    common = dict(data_folder=args.data_dir, reports_file=args.reports,
                  labels_file=args.labels, meta_file=meta_file,
                  label_cols=args.conditions)
    train_ds = MerlinDataset(**common, split="train")
    val_ds   = MerlinDataset(**common, split="val")
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

    n_tr, n_val = len(train_ds), len(val_ds)
    print(f"  train={n_tr}  val={n_val}"
          + (f"  (combined train+val={n_tr + n_val})" if args.full_data else ""))
    print(f"  conditions ({len(label_names)}): {', '.join(label_names)}\n")

    pin = device.type == "cuda"
    # Fixed-order loaders for teacher feature caching (one per split).
    train_cache_loader = DataLoader(_IndexedDataset(train_ds), batch_size=args.batch_size,
                                    shuffle=False, num_workers=min(2, args.num_workers),
                                    pin_memory=pin, persistent_workers=args.num_workers > 0)
    val_cache_loader   = DataLoader(_IndexedDataset(val_ds),   batch_size=args.batch_size,
                                    shuffle=False, num_workers=min(2, args.num_workers),
                                    pin_memory=pin, persistent_workers=args.num_workers > 0)

    # ── models ────────────────────────────────────────────────────────────
    print("Loading Merlin models (teacher + student)...")
    teacher = MerlinEncoder().to(device)
    for p in teacher.parameters():
        p.requires_grad_(False)

    student = MerlinEncoder().to(device)

    # Partial layer freezing: gradients only — frozen layers are still quantized.
    # Segments in order: stem (conv1+bn1), layer1, layer2, layer3, layer4.
    _I3_SEGMENTS = ["stem", "layer1", "layer2", "layer3", "layer4"]
    _frozen_params = 0
    if args.freeze_layers > 0:
        i3 = student.image_encoder.i3_resnet
        segments_to_freeze = _I3_SEGMENTS[:args.freeze_layers]
        for seg in segments_to_freeze:
            if seg == "stem":
                modules = [i3.conv1, i3.bn1]
            else:
                modules = [getattr(i3, seg)]
            for mod in modules:
                for p in mod.parameters():
                    p.requires_grad_(False)
                    _frozen_params += p.numel()

    _bn_count = 0
    if args.freeze_bn:
        for m in student.modules():
            if isinstance(m, nn.modules.batchnorm._BatchNorm):
                m.eval()
                m.weight.requires_grad_(False)
                m.bias.requires_grad_(False)
                _bn_count += 1

    _total_params   = sum(p.numel() for p in student.image_encoder.parameters())
    _trainable_params = sum(p.numel() for p in student.image_encoder.parameters() if p.requires_grad)
    print(f"  encoder params: {_total_params:,}  trainable: {_trainable_params:,}"
          + (f"  ({_frozen_params:,} frozen via --freeze_layers {args.freeze_layers})" if args.freeze_layers > 0 else ""))
    print(f"  BatchNorm layers frozen: {_bn_count}")
    print("  gradient checkpointing: always-on (built into I3ResNet)")

    # ── cache teacher features (train + val; test not needed — no probe here) ──
    cache_dir      = Path(args.teacher_cache_dir or args.data_dir) / "teacher_cache"
    full_cache_dir = Path(args.data_dir) / "teacher_features_all"
    print("\nTeacher feature cache...")
    teacher_train_cache = load_or_compute_teacher_cache(
        teacher, train_cache_loader, device, use_amp, amp_dtype,
        cache_dir, "train", n_tr, args.target_shape,
        full_cache_dir=full_cache_dir, subset_indices=_unwrap_subset_indices(train_ds))
    teacher_val_cache = load_or_compute_teacher_cache(
        teacher, val_cache_loader, device, use_amp, amp_dtype,
        cache_dir, "val", n_val, args.target_shape,
        full_cache_dir=full_cache_dir, subset_indices=_unwrap_subset_indices(val_ds))
    print()

    # ── training-set loader + teacher cache for the training loss ──────────
    if args.full_data:
        # Combine train+val; val indices offset by n_tr to align with ConcatDataset.
        combined = ConcatDataset([train_ds, val_ds])
        teacher_cache = {**teacher_train_cache,
                         **{i + n_tr: v for i, v in teacher_val_cache.items()}}
        train_loader = DataLoader(_IndexedDataset(combined), batch_size=args.batch_size,
                                  shuffle=True, num_workers=args.num_workers,
                                  pin_memory=pin, persistent_workers=args.num_workers > 0)
    else:
        teacher_cache = teacher_train_cache
        train_loader = DataLoader(_IndexedDataset(train_ds), batch_size=args.batch_size,
                                  shuffle=True, num_workers=args.num_workers,
                                  pin_memory=pin, persistent_workers=args.num_workers > 0)
        val_loader   = DataLoader(_IndexedDataset(val_ds), batch_size=args.batch_size,
                                  shuffle=False, num_workers=min(2, args.num_workers),
                                  pin_memory=pin, persistent_workers=args.num_workers > 0)

    # ── optimizer / scheduler / scaler ───────────────────────────────────
    # One recipe for both regimes: AdamW (weight_decay=1e-4) with a cosine LR
    # schedule annealing to lr*0.01. Applied regardless of --full_data so the
    # full-data run uses the same weight-decay + cosine schedule as the default
    # train/val run.
    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr,
                                   betas=(0.9, 0.999), weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
    scaler = (torch.amp.GradScaler("cuda")
              if use_amp and amp_dtype == torch.float16 else None)

    # ── training ──────────────────────────────────────────────────────────
    print(f"\n{'Training':=^70}")
    epoch_log      = []
    best_avg_val   = float("inf")
    best_val_epoch = 0

    if args.full_data:
        print(f"{'Epoch':>6}  {'train_loss':>12}")
        for epoch in range(1, args.epochs + 1):
            tr_loss = train_epoch(
                student, teacher_cache, train_loader, optimizer, scheduler,
                (args.w_bits_min, args.w_bits_max), (args.a_bits_min, args.a_bits_max),
                device, use_amp, amp_dtype, args.grad_accum, scaler,
                freeze_bn=args.freeze_bn, needs_input_grad=args.freeze_layers > 0,
            )
            if tr_loss != tr_loss:  # NaN check
                print(f"\n[abort] train_loss is NaN at epoch {epoch}")
                sys.exit(1)
            print(f"{epoch:6d}  {tr_loss:12.6f}", flush=True)
            epoch_log.append({"epoch": epoch, "train_loss": round(tr_loss, 7)})
            torch.save({"epoch": epoch, "student": student.state_dict()},
                       out / f"checkpoint_epoch{epoch}.pt")
    else:
        print(f"{'Epoch':>6}  {'train_loss':>12}  {'val_loss':>10}  {'val_avg':>10}  {'val_fp_sim':>12}")
        no_improve  = 0
        val_history = deque(maxlen=args.val_window)
        for epoch in range(1, args.epochs + 1):
            tr_loss = train_epoch(
                student, teacher_cache, train_loader, optimizer, scheduler,
                (args.w_bits_min, args.w_bits_max), (args.a_bits_min, args.a_bits_max),
                device, use_amp, amp_dtype, args.grad_accum, scaler,
                freeze_bn=args.freeze_bn, needs_input_grad=args.freeze_layers > 0,
            )
            if tr_loss != tr_loss:  # NaN check
                print(f"\n[abort] train_loss is NaN at epoch {epoch} — "
                      f"likely fp16 overflow or frozen-BN mismatch. "
                      f"Try --amp_dtype bf16 or remove --freeze_bn.")
                sys.exit(1)

            val_loss = val_fp_sim = avg_val = None
            saved    = ""
            if epoch % args.val_every == 0 or epoch == args.epochs:
                val_loss, val_fp_sim = validate_epoch(
                    student, teacher_val_cache, val_loader, device, use_amp, amp_dtype, _val_quant_configs)
                val_history.append(val_loss)
                window_full = len(val_history) == args.val_window
                avg_val     = sum(val_history) / len(val_history)
                if window_full and avg_val < best_avg_val:
                    best_avg_val   = avg_val
                    best_val_epoch = epoch
                    no_improve     = 0
                    torch.save({"epoch": epoch, "student": student.state_dict()},
                               out / "checkpoint_best.pt")
                    saved = "*"
                elif window_full:
                    no_improve += 1

            val_str    = f"{val_loss:10.6f}"   if val_loss    is not None else f"{'—':>10}"
            avg_str    = f"{avg_val:10.6f}"    if avg_val     is not None else f"{'—':>10}"
            fp_sim_str = f"{val_fp_sim:12.4f}" if val_fp_sim  is not None else f"{'—':>12}"
            marker     = " *" if saved else ""
            print(f"{epoch:6d}  {tr_loss:12.6f}  {val_str}  {avg_str}  {fp_sim_str}{marker}", flush=True)

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

    # ── save run metadata ──────────────────────────────────────────────────
    results = {
        "model":         "merlin",
        "full_data":     args.full_data,
        "w_bits_range":  [args.w_bits_min, args.w_bits_max],
        "a_bits_range":  [args.a_bits_min, args.a_bits_max],
        "epochs":        args.epochs,
        "lr":            args.lr,
        "batch_size":    args.batch_size,
        "grad_accum":    args.grad_accum,
        "n_train":       n_tr,
        "n_val":         n_val,
        "label_names":   label_names,
        "patience":       None if args.full_data else args.patience,
        "val_window":     None if args.full_data else args.val_window,
        "best_val_epoch": None if args.full_data else best_val_epoch,
        "best_avg_val":   None if (args.full_data or best_avg_val == float("inf"))
                          else round(best_avg_val, 7),
        "epoch_log":      epoch_log,
    }
    out_path = out / "results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nTraining complete. Checkpoints + metadata → {out}")
    print("Next: cache features (cache_merlin_features.py) then probe (run_probe_merlin.py).")


if __name__ == "__main__":
    main()
