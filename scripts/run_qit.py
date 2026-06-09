"""
QIT validation on CIFAR-10 with a pretrained ResNet18.

Mirrors the CT-CLIP QIT pipeline on standard 2D natural images:
  - Teacher : ImageNet-pretrained ResNet18, fc=Identity → 512-dim features
  - Student : same weights, QIT-trained to match teacher under random quantization
  - Loss    : cosine | mse | kl  (--loss)
  - Probe   : PyTorch linear head (nn.Linear + Adam), top-1 accuracy at FP / W8A8 / W4A4 / W2A4

Teacher features for the full 50k training set are cached in memory at startup
so no teacher forward pass is needed during training.

If QIT works here but not on CT-CLIP the problem is CT-CLIP-specific.

Usage:
    python scripts/run_qit_cifar.py
    python scripts/run_qit_cifar.py --loss cosine --epochs 50
"""

import argparse
import json
import shutil
import sys
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from models.quantization import quantized_forward, BitWidthSampler
from models.eval_utils import (
    _pbar, QUANT_CONFIGS, _TRANSFORM, build_backbone,
    extract_features, run_probe,
)


# ── helpers ───────────────────────────────────────────────────────────────────

class _IndexedSubset(torch.utils.data.Dataset):
    """Subset that also returns the original full-dataset index per item."""
    def __init__(self, subset: Subset):
        self.subset = subset

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        x, y = self.subset[idx]
        return x, y, self.subset.indices[idx]


# ---------------------------------------------------------------------------
# Layer-wise CKA
# ---------------------------------------------------------------------------

# Verified layer names for each backbone (named_modules paths).
# Chosen to give ~4 evenly-spaced checkpoints through the network depth.
_CKA_LAYERS: dict = {
    "resnet18": [
        "layer1", "layer2", "layer3", "layer4",
    ],
    "efficientnet_b0": [
        "features.2", "features.4", "features.6", "features.8",
    ],
    "mobilenet_v3_small": [
        "features.3", "features.6", "features.9", "features.12",
    ],
    "vit_b_16": [
        "encoder.layers.encoder_layer_2",
        "encoder.layers.encoder_layer_5",
        "encoder.layers.encoder_layer_8",
        "encoder.layers.encoder_layer_11",
    ],
    "swin_t": [
        "features.1", "features.3", "features.5", "features.7",
    ],
}


def _pool_to_2d(t: torch.Tensor) -> torch.Tensor:
    """Collapse spatial / sequence dims to (N, D) so CKA can operate on a matrix."""
    if t.dim() == 4:    # (N, C, H, W) — CNN feature map → global avg pool
        return t.flatten(2).mean(2)
    if t.dim() == 3:    # (N, seq, D) — transformer sequence → mean over tokens
        return t.mean(1)
    return t            # already (N, D)


def _linear_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    """Linear CKA between two (N, D) representation matrices.

    Invariant to orthogonal transforms and isotropic scaling — measures how
    similarly the two models organise the same inputs, not just whether their
    feature vectors point in the same direction.
    """
    X = X - X.mean(0)
    Y = Y - Y.mean(0)
    num   = torch.norm(Y.T @ X, "fro").item() ** 2
    denom = torch.norm(X.T @ X, "fro").item() * torch.norm(Y.T @ Y, "fro").item()
    return num / (denom + 1e-8)


@torch.no_grad()
def compute_layer_cka(teacher: nn.Module, student: nn.Module,
                      probe_x: torch.Tensor, backbone: str) -> dict:
    """Compute linear CKA between student and teacher at each named layer.

    Both models are run in eval / no-grad mode on a single probe batch.
    Spatial and sequence dimensions are collapsed via global average before CKA
    so the metric is architecture-agnostic.

    Returns an empty dict if the backbone is not in _CKA_LAYERS.
    """
    layer_names = _CKA_LAYERS.get(backbone, [])
    if not layer_names:
        return {}

    t_acts: dict = {}
    s_acts: dict = {}
    hooks  = []

    def _make_hook(store: dict, key: str):
        def _hook(mod, inp, out):
            store[key] = _pool_to_2d(out.detach().float().cpu())
        return _hook

    for model, store in [(teacher, t_acts), (student, s_acts)]:
        model.eval()
        for ln in layer_names:
            mod = model
            for part in ln.split("."):
                mod = getattr(mod, part)
            hooks.append(mod.register_forward_hook(_make_hook(store, ln)))

    teacher(probe_x)
    student(probe_x)

    for h in hooks:
        h.remove()

    return {
        ln: round(_linear_cka(s_acts[ln], t_acts[ln]), 4)
        for ln in layer_names
        if ln in s_acts and ln in t_acts
    }


def _freeze_bn(model: nn.Module) -> int:
    """Set all BatchNorm layers to eval mode with frozen parameters.

    Called once after loading and again after every student.train() call,
    since .train() recursively re-enables update mode on all submodules.
    Returns the number of layers frozen (0 for ViT/Swin which use LayerNorm).
    """
    count = 0
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.eval()
            m.weight.requires_grad_(False)
            m.bias.requires_grad_(False)
            count += 1
    return count


def compute_qit_loss(z_student: torch.Tensor, z_teacher: torch.Tensor,
                     loss_type: str) -> torch.Tensor:
    zt = z_teacher.detach()
    if loss_type == "cosine":
        return 1 - F.cosine_similarity(z_student, zt).mean()
    if loss_type == "mse":
        return F.mse_loss(z_student, zt)
    if loss_type == "kl":
        return F.kl_div(F.log_softmax(z_student, dim=-1),
                        F.softmax(zt, dim=-1), reduction="batchmean")
    raise ValueError(f"Unknown loss_type: {loss_type!r}")


# ── feature extraction ────────────────────────────────────────────────────────

@torch.no_grad()
def cache_teacher_features(teacher, loader, device, use_amp) -> torch.Tensor:
    """Extract teacher features for the full training set (shuffle=False)."""
    teacher.eval()
    feats = []
    for batch in _pbar(loader, desc="  caching teacher features", leave=False):
        x = batch[0].to(device)
        if use_amp:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                feats.append(teacher(x).cpu().float())
        else:
            feats.append(teacher(x).cpu())
    return torch.cat(feats)  # [N, 512]


# ── training loop ─────────────────────────────────────────────────────────────

def _fp_forward(student, x, use_amp):
    if use_amp:
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            return student(x).float()
    return student(x)


def train_epoch(student, loader, optimizer, loss_type,
                bit_sampler, device, use_amp,
                teacher_feats_cache, weight_granularity="per_tensor",
                fp_loss_weight=0.0, teacher_params=None, reg_weight=0.0,
                freeze_bn=False):
    student.train()
    if freeze_bn:
        # .train() re-enables BN update mode on all submodules; re-freeze immediately.
        _freeze_bn(student)
    total_loss = total_fp_sim = n = 0

    for x, _, full_idx in _pbar(loader, desc="  batches", leave=False):
        x = x.to(device)
        z_teacher = teacher_feats_cache[full_idx].to(device)

        w_bits, a_bits = bit_sampler.sample()
        optimizer.zero_grad()

        # Quantized forward (always needed for QIT loss)
        if use_amp:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                with quantized_forward([student], w_bits, a_bits, weight_granularity):
                    z_student_q = student(x)
        else:
            with quantized_forward([student], w_bits, a_bits, weight_granularity):
                z_student_q = student(x)

        loss = compute_qit_loss(z_student_q.float(), z_teacher, loss_type)

        # Optional FP preservation term: penalises FP feature drift from teacher
        if fp_loss_weight > 0.0:
            z_student_fp = _fp_forward(student, x, use_amp)
            loss = loss + fp_loss_weight * compute_qit_loss(z_student_fp, z_teacher, loss_type)

        # Optional teacher-anchored weight regularisation: discourages weight outliers
        if reg_weight > 0.0 and teacher_params is not None:
            for (_, tp), (_, sp) in zip(teacher_params, student.named_parameters()):
                loss = loss + reg_weight * ((sp - tp.detach()) ** 2).mean()

        loss.backward()
        optimizer.step()

        # FP sim monitoring (reuse z_student_fp if already computed)
        with torch.no_grad():
            z_fp = (z_student_fp.detach() if fp_loss_weight > 0.0
                    else _fp_forward(student, x, use_amp))
            fp_sim = F.cosine_similarity(z_fp, z_teacher).mean().item()

        total_loss   += loss.item()
        total_fp_sim += fp_sim
        n += 1

    return total_loss / max(n, 1), total_fp_sim / max(n, 1)


@torch.no_grad()
def validate_epoch(student, loader, loss_type,
                   all_configs, device, use_amp,
                   teacher_feats_cache, weight_granularity="per_tensor"):
    """Deterministic val: cycles through all (w, a) configs in fixed order.

    Each val batch is assigned configs[i % len(configs)], giving equal
    representation of every bit-width and a stable, reproducible val loss.
    """
    student.eval()
    total_loss = total_fp_sim = n = 0

    for i, (x, _, full_idx) in enumerate(loader):
        x = x.to(device)
        z_teacher = teacher_feats_cache[full_idx].to(device)
        w_bits, a_bits = all_configs[i % len(all_configs)]

        if use_amp:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                with quantized_forward([student], w_bits, a_bits, weight_granularity):
                    z_student = student(x)
                z_fp = student(x)
        else:
            with quantized_forward([student], w_bits, a_bits, weight_granularity):
                z_student = student(x)
            z_fp = student(x)

        total_loss   += compute_qit_loss(z_student.float(), z_teacher, loss_type).item()
        total_fp_sim += F.cosine_similarity(z_fp.float(), z_teacher).mean().item()
        n += 1

    return total_loss / max(n, 1), total_fp_sim / max(n, 1)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",      type=int,   default=50)
    parser.add_argument("--patience",    type=int,   default=10,
                        help="Early stopping: halt if moving-avg val loss does not "
                             "improve for this many epochs. 0 = disabled.")
    parser.add_argument("--val_window",  type=int,   default=3,
                        help="Window size for moving-average val loss used in "
                             "early stopping and best-checkpoint selection.")
    parser.add_argument("--lr",          type=float, default=1e-4)
    parser.add_argument("--lr_schedule", type=str,   default="cosine",
                        choices=["cosine", "constant"],
                        help="LR schedule. cosine = anneal to 0 over all epochs.")
    parser.add_argument("--batch_size",  type=int,   default=16)
    parser.add_argument("--loss",        type=str,   default="cosine",
                        choices=["cosine", "mse", "kl"])
    parser.add_argument("--weight_granularity", type=str, default="per_tensor",
                        choices=["per_tensor", "per_channel"])
    parser.add_argument("--fp_loss_weight", type=float, default=0.0,
                        help="Weight for FP preservation term: "
                             "loss += fp_loss_weight * loss(student_fp, teacher). "
                             "0 = disabled (pure QIT). Try 0.5.")
    parser.add_argument("--reg_weight",    type=float, default=0.0,
                        help="Teacher-anchored L2 regularisation weight: "
                             "loss += reg_weight * ||w_student - w_teacher||². "
                             "Discourages weight outliers. Try 1e-4.")
    parser.add_argument("--bit_sampling", type=str, default="wor",
                        choices=["wor", "random"],
                        help="Bit-width sampling strategy for training: "
                             "wor = without-replacement; random = uniform with replacement.")
    parser.add_argument("--w_bits_min",  type=int,   default=4)
    parser.add_argument("--w_bits_max",  type=int,   default=8)
    parser.add_argument("--a_bits_min",  type=int,   default=4)
    parser.add_argument("--a_bits_max",  type=int,   default=8)
    parser.add_argument("--n_train",     type=int,   default=5000,
                        help="QIT training images to use. -1 = all.")
    parser.add_argument("--num_workers", type=int,   default=8)
    parser.add_argument("--backbone",     type=str,   default="resnet18",
                        choices=["resnet18", "efficientnet_b0",
                                 "mobilenet_v3_small", "vit_b_16", "swin_t"])
    parser.add_argument("--dataset",     type=str,   default="stl10",
                        choices=["cifar10", "stl10"],
                        help="Dataset. stl10 uses the unlabeled split for QIT "
                             "and the labeled train/test splits for probing.")
    parser.add_argument("--data_dir",    type=str,   default=None,
                        help="Dataset root dir. Defaults to data/<dataset>.")
    parser.add_argument("--log_layer_cka", action="store_true",
                        help="After each validation step, compute linear CKA between "
                             "student (FP) and teacher at ~4 key layers and log to "
                             "results.json. Default: off.")
    parser.add_argument("--cka_n_batches", type=int, default=1,
                        help="Number of consecutive val batches to use as the CKA probe. "
                             "More batches give a more stable estimate at the cost of "
                             "one extra forward pass per additional batch (~1 s each). "
                             "Default: 1 (single batch, fast but noisy for deep layers).")
    parser.add_argument("--freeze_bn",    action="store_true",
                        help="Freeze BatchNorm layers throughout training (eval mode, "
                             "no grad on scale/bias). Recommended for CNN backbones on "
                             "low-resolution datasets (e.g. ResNet18/CIFAR-10): QIT mixes "
                             "FP and quantized passes each step, which corrupts BN running "
                             "stats; the student weights then adapt to the wrong "
                             "normalisation, and post-hoc recalibration cannot undo this. "
                             "ViT/Swin are unaffected (they use LayerNorm). Default: off, "
                             "preserving the behaviour validated on STL-10.")
    parser.add_argument("--no_amp",      action="store_true")
    parser.add_argument("--output_dir",  type=str,   default=None,
                        help="Output dir. Defaults to runs/qit_<dataset>_<backbone>.")
    args = parser.parse_args()

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = not args.no_amp and device.type == "cuda"
    out     = Path(args.output_dir or f"runs/qit_{args.dataset}_{args.backbone}")
    out.mkdir(parents=True, exist_ok=True)

    w_range = (args.w_bits_min, args.w_bits_max)
    a_range = (args.a_bits_min, args.a_bits_max)

    data_dir = args.data_dir or f"data/{args.dataset}"
    print("=" * 70)
    print(f"QIT validation — {args.dataset.upper()} / {args.backbone}")
    print("=" * 70)
    print(f"  backbone     : {args.backbone}")
    print(f"  dataset      : {args.dataset}")
    print(f"  n_train      : {args.n_train} (-1 = all)")
    print(f"  epochs       : {args.epochs}")
    print(f"  lr           : {args.lr:.1e}  schedule: {args.lr_schedule}")
    print(f"  batch_size   : {args.batch_size}")
    print(f"  loss               : {args.loss}")
    print(f"  weight_granularity : {args.weight_granularity}")
    print(f"  fp_loss_weight     : {args.fp_loss_weight}")
    print(f"  reg_weight         : {args.reg_weight}")
    print(f"  w_bits range : {w_range}")
    print(f"  a_bits range : {a_range}")
    print(f"  amp          : {use_amp}")
    print(f"  output       : {out}\n")

    # ── dataset ───────────────────────────────────────────────────────────
    if args.dataset == "stl10":
        print("Loading STL-10...")
        # Unlabeled split (100k) for QIT training — no labels needed
        qit_full       = datasets.STL10(data_dir, split="unlabeled", download=True,
                                        transform=_TRANSFORM)
        # Labeled splits for probe evaluation
        probe_train_ds = datasets.STL10(data_dir, split="train", download=True,
                                        transform=_TRANSFORM)
        probe_test_ds  = datasets.STL10(data_dir, split="test",  download=True,
                                        transform=_TRANSFORM)
        full_train = qit_full   # teacher feature cache indexes into this
    else:
        print("Loading CIFAR-10...")
        full_train    = datasets.CIFAR10(data_dir, train=True,  download=True,
                                         transform=_TRANSFORM)
        probe_test_ds = datasets.CIFAR10(data_dir, train=False, download=True,
                                         transform=_TRANSFORM)

    # QIT train/val split within the QIT dataset
    rng      = np.random.default_rng(42)
    all_idx  = rng.permutation(len(full_train)).tolist()
    n_subset = len(full_train) if args.n_train < 0 else min(args.n_train, len(full_train))
    subset   = all_idx[:n_subset]
    n_train  = int(0.9 * n_subset)

    # CIFAR-10: probe trains on the held-out portion not used for QIT
    if args.dataset == "cifar10":
        probe_train_ds = Subset(full_train, all_idx[n_subset:])
    train_ds = _IndexedSubset(Subset(full_train, subset[:n_train]))
    val_ds   = _IndexedSubset(Subset(full_train, subset[n_train:]))

    # Loaders
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=4, pin_memory=True)
    # cache_loader: full QIT dataset in fixed order for teacher feature alignment
    cache_loader       = DataLoader(full_train,     batch_size=args.batch_size,
                                    shuffle=False,  num_workers=args.num_workers,
                                    pin_memory=True)
    probe_train_loader = DataLoader(probe_train_ds, batch_size=args.batch_size,
                                    shuffle=False,  num_workers=4, pin_memory=True)
    probe_test_loader  = DataLoader(probe_test_ds,  batch_size=args.batch_size,
                                    shuffle=False,  num_workers=4, pin_memory=True)
    print(f"  QIT train: {len(train_ds)} | QIT val: {len(val_ds)} | "
          f"Probe train: {len(probe_train_ds)} | Probe test: {len(probe_test_ds)}\n")

    # ── teacher ───────────────────────────────────────────────────────────
    print(f"Loading teacher (ImageNet-pretrained {args.backbone})...")
    teacher = build_backbone(args.backbone).to(device)
    for p in teacher.parameters():
        p.requires_grad_(False)
    teacher.eval()

    # ── student ───────────────────────────────────────────────────────────
    print(f"Loading student (same ImageNet-pretrained weights)...")
    student = build_backbone(args.backbone).to(device)

    if args.freeze_bn:
        n_frozen = _freeze_bn(student)
        if n_frozen:
            print(f"  BN freeze enabled: {n_frozen} layers frozen for training.")
        else:
            print("  BN freeze requested but no BatchNorm layers found (no-op for this backbone).")

    results        = {"epochs": [], "probe": {}}
    best_ckpt_path = out / "checkpoint_best.pt"

    if args.epochs > 0:
        feat_cache_path = Path(data_dir) / f"teacher_feats_{args.backbone}.pt"
        if feat_cache_path.exists():
            print(f"Loading cached teacher features from {feat_cache_path}...")
            teacher_feats_cache = torch.load(feat_cache_path, map_location="cpu")
        else:
            print("Computing teacher features (first run for this backbone)...")
            teacher_feats_cache = cache_teacher_features(teacher, cache_loader, device, use_amp)
            torch.save(teacher_feats_cache, feat_cache_path)
            print(f"  Saved to {feat_cache_path}")
        print(f"  Cache shape: {teacher_feats_cache.shape}\n")

        bit_sampler = BitWidthSampler(w_range, a_range, mode=args.bit_sampling)
        all_configs = bit_sampler.configs  # fixed order for deterministic val
        print(f"  bit-width grid : {len(all_configs)} configs  "
              f"({w_range[1]-w_range[0]+1}w × {a_range[1]-a_range[0]+1}a)\n")

        optimizer = torch.optim.Adam(student.parameters(), lr=args.lr)
        scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(
                         optimizer, T_max=args.epochs, eta_min=0)
                     if args.lr_schedule == "cosine" else None)
        teacher_params = (list(teacher.named_parameters())
                          if args.reg_weight > 0.0 else None)

        val_history       = deque(maxlen=args.val_window)
        ckpt_buffer       = deque(maxlen=args.val_window)  # (epoch, buf_slot)
        best_avg_val      = float("inf")
        best_mid_epoch    = -1
        epochs_no_improve = 0
    else:
        print("[epochs=0] Skipping QIT training — BN recalibration control.\n")

    # ── training ──────────────────────────────────────────────────────────
    epoch = 0
    if args.epochs > 0:
        print(f"\n{'Training':=^70}")
    for epoch in range(1, args.epochs + 1):
        loss, fp_sim = train_epoch(
            student, train_loader, optimizer, args.loss,
            bit_sampler, device, use_amp, teacher_feats_cache,
            weight_granularity=args.weight_granularity,
            fp_loss_weight=args.fp_loss_weight,
            teacher_params=teacher_params,
            reg_weight=args.reg_weight,
            freeze_bn=args.freeze_bn,
        )
        val_loss, val_fp_sim = validate_epoch(
            student, val_loader, args.loss,
            all_configs, device, use_amp, teacher_feats_cache,
            weight_granularity=args.weight_granularity,
        )

        val_history.append(val_loss)
        avg_val  = sum(val_history) / len(val_history)
        window_full = len(val_history) == args.val_window
        is_best  = window_full and avg_val < best_avg_val
        if is_best:
            best_avg_val      = avg_val
            epochs_no_improve = 0
        elif window_full:
            epochs_no_improve += 1

        # Optional layer-wise CKA: FP forward pass on cka_n_batches consecutive
        # val batches. More batches → more stable estimate, ~1 s overhead each.
        layer_cka = {}
        if args.log_layer_cka:
            chunks = []
            for i, (bx, _, _) in enumerate(val_loader):
                chunks.append(bx)
                if i + 1 >= args.cka_n_batches:
                    break
            probe_x = torch.cat(chunks, dim=0).to(device)
            layer_cka = compute_layer_cka(
                teacher, student, probe_x, args.backbone)

        marker = f" *best* (avg={avg_val:.4f})" if is_best else (
                 f" (avg={avg_val:.4f})" if window_full else "")
        print(f"[epoch {epoch:3d}/{args.epochs}]  "
              f"loss={loss:.4f}  fp_sim={fp_sim:.4f}  |  "
              f"val_loss={val_loss:.4f}  val_fp_sim={val_fp_sim:.4f}{marker}")

        log_entry = {
            "epoch": epoch, "loss": loss, "fp_sim": fp_sim,
            "val_loss": val_loss, "val_fp_sim": val_fp_sim,
        }
        if layer_cka:
            log_entry["layer_cka"] = layer_cka
        results["epochs"].append(log_entry)

        slot = (epoch - 1) % args.val_window
        ckpt = {"epoch": epoch, "student": student.state_dict()}
        torch.save(ckpt, out / f"_buf{slot}.pt")
        torch.save(ckpt, out / "checkpoint_latest.pt")
        ckpt_buffer.append((epoch, slot))
        if is_best:
            mid_epoch, mid_slot = list(ckpt_buffer)[args.val_window // 2]
            shutil.copy(out / f"_buf{mid_slot}.pt", best_ckpt_path)
            best_mid_epoch = mid_epoch

        if scheduler is not None:
            scheduler.step()

        if args.patience > 0 and window_full and epochs_no_improve >= args.patience:
            print(f"\n[early stop] {args.val_window}-epoch avg val loss did not improve "
                  f"for {args.patience} epochs. Best avg: {best_avg_val:.4f}")
            break

    if epoch > 0:
        torch.save({"epoch": epoch, "student": student.state_dict()},
                   out / "checkpoint_final.pt")
        for i in range(args.val_window):
            p = out / f"_buf{i}.pt"
            if p.exists():
                p.unlink()

    # Load best checkpoint for probe evaluation (skip if epochs=0)
    if best_ckpt_path.exists():
        print(f"\nLoading best checkpoint (epoch {best_mid_epoch}, "
              f"window avg={best_avg_val:.4f}) for evaluation...")
        student.load_state_dict(torch.load(best_ckpt_path, map_location=device)["student"])
    else:
        print("\n[epochs=0] Using initial student weights (= teacher) for evaluation.")

    # ── BN recalibration (student only, CNN backbones) ───────────────────
    # QIT mixes quantized and FP forward passes, miscalibrating BatchNorm
    # running stats. Reset them with a FP pass over the QIT training data.
    # ViT/Swin use LayerNorm (no running stats) so recalibration is skipped.
    # The teacher is NOT recalibrated — its ImageNet BN stats are already
    # correct for its learned features, and recalibrating on a different
    # domain degrades feature quality.
    has_bn = any(isinstance(m, nn.modules.batchnorm._BatchNorm)
                 for m in student.modules())
    if has_bn and not args.freeze_bn:
        # Post-hoc recalibration: reset BN running stats with a clean FP pass.
        # Only needed when BN was NOT frozen during training; if freeze_bn was
        # used the stats were never corrupted and this step is unnecessary.
        print("Recalibrating student BN stats...")
        student.train()
        with torch.no_grad():
            for batch in _pbar(cache_loader, desc="  student BN recal", leave=False):
                x = batch[0].to(device)
                if use_amp:
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        student(x)
                else:
                    student(x)
        student.eval()
        print("  Done.\n")
    elif not has_bn:
        print("Skipping BN recalibration (no BatchNorm layers).\n")
    else:
        print("Skipping BN recalibration (BN was frozen throughout training).\n")

    # ── probe ─────────────────────────────────────────────────────────────
    print(f"\n{'Probe evaluation':=^70}")
    print(f"  {'Config':<8}  {'Teacher acc':>12}  {'Student acc':>12}  "
          f"{'Δ':>8}  {'feat_sim':>10}")

    for qk, wb, ab in QUANT_CONFIGS:
        print(f"\n  [{qk}] extracting features...")
        t_tr, l_tr = extract_features(teacher, probe_train_loader, device, use_amp, wb, ab, args.weight_granularity)
        s_tr, _    = extract_features(student, probe_train_loader, device, use_amp, wb, ab, args.weight_granularity)
        t_te, l_te = extract_features(teacher, probe_test_loader,  device, use_amp, wb, ab, args.weight_granularity)
        s_te, _    = extract_features(student, probe_test_loader,  device, use_amp, wb, ab, args.weight_granularity)

        feat_sim    = float(F.cosine_similarity(
            torch.tensor(s_tr), torch.tensor(t_tr)).mean())
        teacher_acc = run_probe(t_tr, l_tr, t_te, l_te, device)
        student_acc = run_probe(s_tr, l_tr, s_te, l_te, device)
        delta       = student_acc - teacher_acc

        print(f"  {qk:<8}  {teacher_acc:>12.4f}  {student_acc:>12.4f}  "
              f"  {delta:>+8.4f}  {feat_sim:>10.4f}")
        results["probe"][qk] = {
            "teacher_acc": teacher_acc,
            "student_acc": student_acc,
            "feat_sim":    feat_sim,
        }

    out_path = out / "results.json"
    with open(out_path, "w") as f:
        json.dump({
            "backbone": args.backbone,
            "epochs": args.epochs, "patience": args.patience,
            "val_window": args.val_window,
            "lr": args.lr, "lr_schedule": args.lr_schedule,
            "batch_size": args.batch_size, "loss": args.loss,
            "weight_granularity": args.weight_granularity,
            "fp_loss_weight": args.fp_loss_weight,
            "reg_weight":     args.reg_weight,
            "w_bits_range":   list(w_range),
            "a_bits_range":   list(a_range),
            "results":        results,
        }, f, indent=2)
    print(f"\nResults → {out_path}")


if __name__ == "__main__":
    main()