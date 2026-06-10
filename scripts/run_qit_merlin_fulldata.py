"""
Merlin QIT — full-data exploratory run.

Trains on the combined train+val split (~4333 scans), no validation loop,
evaluates on the test set after a fixed number of epochs.

Compared to run_qit_merlin.py this script is intentionally minimal:
  - no validation, no early stopping, no patience
  - hardcoded hyperparams (see constants below)
  - teacher features cached under data_dir/teacher_cache/teacher_train_val_n*
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader

_ROOT = Path(__file__).parent.parent
_QFT  = _ROOT.parent / "quantized_ft"
sys.path.insert(0, str(_ROOT))
sys.path.insert(1, str(_QFT))

warnings.filterwarnings("ignore", category=UserWarning)

from models.quantization import quantized_forward, sample_bits
from models.eval_utils import _pbar, MerlinEncoder
from downstream.dataset import MerlinDataset


# ---------------------------------------------------------------------------
# Hyperparams — edit here, not via CLI
# ---------------------------------------------------------------------------

EPOCHS       = 5
BATCH_SIZE   = 16
GRAD_ACCUM   = 4          # effective batch = 64
LR           = 1e-5
W_BITS_MIN   = 4
W_BITS_MAX   = 8
A_BITS_MIN   = 4
A_BITS_MAX   = 8
TARGET_SHAPE = (160, 224, 224)
NUM_WORKERS  = 12
AMP_DTYPE    = torch.bfloat16

_CT_DATA = _ROOT.parent / "CT-CLIP" / "data"

_EVAL_CONFIGS = [
    ("fp",   None, None),
    ("w8a8", 8,    8),
    ("w4a8", 4,    8),
    ("w4a4", 4,    4),
]


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

class _IndexedDataset(torch.utils.data.Dataset):
    def __init__(self, ds):
        self._ds = ds
    def __len__(self):
        return len(self._ds)
    def __getitem__(self, i):
        x, y = self._ds[i]
        return i, x, y


class _ResizedDataset(torch.utils.data.Dataset):
    def __init__(self, ds, target_shape):
        self._ds     = ds
        self._target = tuple(target_shape)

    def __len__(self):
        return len(self._ds)

    @property
    def label_names(self):
        return self._ds.label_names

    def __getitem__(self, i):
        x, y = self._ds[i]
        if tuple(x.shape[1:]) != self._target:
            x = F.interpolate(
                x.unsqueeze(0), size=self._target,
                mode="trilinear", align_corners=False,
            ).squeeze(0)
        return x, y


def _get_labels(ds) -> np.ndarray:
    """Return the (N, n_cond) label array for a (possibly _ResizedDataset-wrapped)
    MerlinDataset, reading directly from .samples — no volume I/O."""
    if isinstance(ds, _ResizedDataset):
        ds = ds._ds
    return np.stack([s[1] for s in ds.samples])


# ---------------------------------------------------------------------------
# Teacher cache
# ---------------------------------------------------------------------------

def _qit_loss(s, t):
    return (1.0 - (F.normalize(s.float(), dim=-1) * F.normalize(t.float(), dim=-1)).sum(-1)).mean()


@torch.no_grad()
def _fp_forward(model, x, amp_dtype):
    with torch.amp.autocast("cuda", dtype=amp_dtype):
        return model(x).float()


@torch.no_grad()
def _compute_cache(teacher, loader, device, amp_dtype):
    teacher.eval()
    cache = {}
    for batch in _pbar(loader, desc="  caching"):
        idx, x, _ = batch
        f = _fp_forward(teacher, x.to(device), amp_dtype)
        for i, fi in zip(idx.tolist(), f.unbind(0)):
            cache[i] = fi.cpu()
    return cache


def load_or_compute_cache(teacher, loader, device, amp_dtype, cache_path):
    if cache_path.exists():
        print(f"  loading cache: {cache_path}", flush=True)
        cache = torch.load(cache_path, map_location="cpu", weights_only=True)
        print(f"  loaded {len(cache)} cached features", flush=True)
        return cache
    print(f"  computing teacher features → {cache_path}...", flush=True)
    cache = _compute_cache(teacher, loader, device, amp_dtype)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, cache_path)
    print(f"  computing teacher features done "
          f"({len(cache)} samples, {len(cache) * 2048 * 4 / 1e6:.1f} MB)", flush=True)
    return cache


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_epoch(student, teacher_cache, loader, optimizer, scheduler,
                device, amp_dtype, grad_accum, scaler):
    student.train()
    total_loss = n_samples = step = 0
    optimizer.zero_grad()

    for idx, x, _ in loader:
        x = x.to(device)
        w_bits, a_bits = sample_bits((W_BITS_MIN, W_BITS_MAX), (A_BITS_MIN, A_BITS_MAX))
        with torch.amp.autocast("cuda", dtype=amp_dtype):
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
                scaler.step(optimizer); scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

    if step % grad_accum != 0:
        if scaler is not None:
            scaler.step(optimizer); scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad()

    scheduler.step()
    return total_loss / max(n_samples, 1)


# ---------------------------------------------------------------------------
# Feature extraction + AUROC probe
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_features(model, loader, device, amp_dtype, w_bits=None, a_bits=None):
    model.eval()
    all_f, all_y = [], []
    for batch in _pbar(loader, desc="    extracting", leave=False):
        x = batch[0].to(device) if len(batch) == 2 else batch[1].to(device)
        y = batch[1]             if len(batch) == 2 else batch[2]
        with torch.amp.autocast("cuda", dtype=amp_dtype):
            if w_bits is not None:
                with quantized_forward([model.image_encoder], w_bits, a_bits, "per_channel"):
                    f = model(x)
            else:
                f = model(x)
        all_f.append(f.cpu().float())
        all_y.append(y)
    return torch.cat(all_f).numpy(), torch.cat(all_y).numpy()


def run_auroc_probe(train_feats, train_labels, test_feats, test_labels,
                    label_names, device, epochs=300, lr=1e-3, batch_size=256, n_seeds=5):
    """Trains n_seeds independently-initialised linear probes and averages
    AUROC across seeds — with only ~750 test scans this reduces variance from
    a single probe initialisation/shuffle."""
    from sklearn.metrics import roc_auc_score

    X_tr = torch.tensor(train_feats, dtype=torch.float32)
    y_tr = torch.tensor(np.clip(train_labels, 0, 1).astype(np.float32))
    m_tr = torch.tensor((train_labels >= 0).astype(np.float32))
    X_te = torch.tensor(test_feats,  dtype=torch.float32)

    mu, std = X_tr.mean(0), X_tr.std(0).clamp(min=1e-8)
    X_tr = (X_tr - mu) / std
    X_te = (X_te - mu) / std

    valid_conds = [
        c for c, _ in enumerate(label_names)
        if (test_labels[:, c] >= 0).sum() >= 10
        and np.ptp(test_labels[test_labels[:, c] >= 0, c]) > 0
    ]

    per_cond_runs = {label_names[c]: [] for c in valid_conds}
    mean_aucs = []

    for seed in range(n_seeds):
        torch.manual_seed(seed)
        head = nn.Linear(X_tr.shape[1], y_tr.shape[1]).to(device)
        opt  = torch.optim.Adam(head.parameters(), lr=lr)
        sch  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=0)
        ds   = torch.utils.data.TensorDataset(X_tr.to(device), y_tr.to(device), m_tr.to(device))
        dl   = DataLoader(ds, batch_size=batch_size, shuffle=True)

        best_loss, no_imp = float("inf"), 0
        for _ in range(epochs):
            head.train()
            ep = 0.0
            for xb, yb, mb in dl:
                opt.zero_grad()
                raw  = F.binary_cross_entropy_with_logits(head(xb), yb, reduction="none")
                loss = (raw * mb).sum() / mb.sum().clamp(min=1)
                loss.backward(); opt.step()
                ep += loss.item()
            sch.step()
            ep /= len(dl)
            if best_loss - ep > 1e-5:
                best_loss, no_imp = ep, 0
            else:
                no_imp += 1
            if no_imp >= 30:
                break

        head.eval()
        with torch.no_grad():
            preds = torch.sigmoid(head(X_te.to(device))).cpu().numpy()

        seed_aucs = []
        for c in valid_conds:
            mask = test_labels[:, c] >= 0
            try:
                auc = float(roc_auc_score(test_labels[mask, c], preds[mask, c]))
                per_cond_runs[label_names[c]].append(auc)
                seed_aucs.append(auc)
            except Exception:
                pass
        if seed_aucs:
            seed_mean = float(np.mean(seed_aucs))
            mean_aucs.append(seed_mean)
            print(f"      probe seed {seed + 1}/{n_seeds} done (auc={seed_mean:.4f})", flush=True)
        else:
            print(f"      probe seed {seed + 1}/{n_seeds} done (no valid conditions)", flush=True)

    per_cond = {name: float(np.mean(vals)) for name, vals in per_cond_runs.items() if vals}
    mean_auc = float(np.mean(mean_aucs)) if mean_aucs else float("nan")
    return mean_auc, per_cond


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",  default=str(_CT_DATA / "merlin_data"))
    parser.add_argument("--reports",   default=str(_CT_DATA / "reports_final.xlsx"))
    parser.add_argument("--labels",    default=str(_CT_DATA / "zero_shot_findings_disease_cls.csv"))
    parser.add_argument("--metadata",  default=str(_CT_DATA / "metadata.csv"))
    parser.add_argument("--conditions",nargs="+",
                        default=["atelectasis", "pleural_effusion", "renal_cyst"])
    parser.add_argument("--output_dir",default="runs/qit_merlin_fulldata")
    args = parser.parse_args()

    device    = torch.device("cuda")
    out       = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    scaler    = torch.amp.GradScaler("cuda") if AMP_DTYPE == torch.float16 else None

    print(f"\n{'QIT — Merlin CT full-data':=^70}")
    print(f"  train+val → test | epochs={EPOCHS} bs={BATCH_SIZE} ga={GRAD_ACCUM} lr={LR}")
    print(f"  W{W_BITS_MIN}-{W_BITS_MAX} A{A_BITS_MIN}-{A_BITS_MAX} | amp=bf16")
    print(f"  output: {out}\n{'='*70}\n")

    # ── dataset ───────────────────────────────────────────────────────────
    meta_file = args.metadata if Path(args.metadata).exists() else None
    common = dict(
        data_folder=args.data_dir,
        reports_file=args.reports,
        labels_file=args.labels,
        meta_file=meta_file,
        label_cols=args.conditions,
    )
    train_raw = _ResizedDataset(MerlinDataset(**common, split="train"), TARGET_SHAPE)
    val_raw   = _ResizedDataset(MerlinDataset(**common, split="val"),   TARGET_SHAPE)
    test_ds   = _ResizedDataset(MerlinDataset(**common, split="test"),  TARGET_SHAPE)

    combined  = ConcatDataset([train_raw, val_raw])
    label_names = train_raw.label_names
    N = len(combined)
    print(f"  train+val={N}  test={len(test_ds)}")
    print(f"  conditions ({len(args.conditions)}): {', '.join(args.conditions)}\n")

    # ── models ────────────────────────────────────────────────────────────
    print("Loading Merlin (teacher + student)...")
    teacher = MerlinEncoder().to(device).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    student = MerlinEncoder().to(device)

    # ── teacher cache ─────────────────────────────────────────────────────
    # All three splits are cached in their own dedicated directory so they
    # can be loaded together later without mixing with per-run caches.
    # Test features are cached now even though this run doesn't train on them.
    shape_str = "x".join(map(str, TARGET_SHAPE))
    cache_dir = Path(args.data_dir) / "teacher_features_all"

    def _split_loader(ds):
        return DataLoader(_IndexedDataset(ds), batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    print("Teacher feature cache...")
    n_tr, n_val, n_te = len(train_raw), len(val_raw), len(test_ds)
    train_cache = load_or_compute_cache(
        teacher, _split_loader(train_raw), device, AMP_DTYPE,
        cache_dir / f"train_n{n_tr}_{shape_str}.pt")
    val_cache = load_or_compute_cache(
        teacher, _split_loader(val_raw), device, AMP_DTYPE,
        cache_dir / f"val_n{n_val}_{shape_str}.pt")
    test_cache = load_or_compute_cache(
        teacher, _split_loader(test_ds), device, AMP_DTYPE,
        cache_dir / f"test_n{n_te}_{shape_str}.pt")

    # merge train+val: val indices offset by n_tr to align with ConcatDataset
    teacher_cache = {**train_cache, **{i + n_tr: v for i, v in val_cache.items()}}

    # ── training ──────────────────────────────────────────────────────────
    train_loader = DataLoader(
        _IndexedDataset(combined), batch_size=BATCH_SIZE,
        shuffle=True, num_workers=NUM_WORKERS, pin_memory=True,
    )
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, student.parameters()), lr=LR
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=0
    )

    print(f"\n{'Training':=^70}")
    print(f"{'Epoch':>6}  {'train_loss':>12}")
    for epoch in range(1, EPOCHS + 1):
        loss = train_epoch(student, teacher_cache, train_loader,
                           optimizer, scheduler, device, AMP_DTYPE, GRAD_ACCUM, scaler)
        if loss != loss:
            print(f"\n[abort] NaN loss at epoch {epoch}")
            sys.exit(1)
        print(f"{epoch:6d}  {loss:12.6f}", flush=True)
        torch.save({"epoch": epoch, "student": student.state_dict()},
                   out / f"checkpoint_epoch{epoch}.pt")

    torch.save({"epoch": EPOCHS, "student": student.state_dict()},
               out / "checkpoint_final.pt")

    # ── test evaluation ───────────────────────────────────────────────────
    print(f"\n{'Test evaluation':=^70}")
    eval_kw = dict(batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)
    train_eval_loader = DataLoader(combined,  **eval_kw)
    test_loader       = DataLoader(test_ds,   **eval_kw)

    auroc_results = {}
    for cfg_name, wb, ab in _EVAL_CONFIGS:
        print(f"\n  [student {cfg_name}] computing train features...", flush=True)
        tr_f, tr_l = extract_features(student, train_eval_loader, device, AMP_DTYPE, wb, ab)
        print(f"  [student {cfg_name}] train features done ({len(tr_f)} samples)", flush=True)
        print(f"  [student {cfg_name}] computing test features...", flush=True)
        te_f, te_l = extract_features(student, test_loader,       device, AMP_DTYPE, wb, ab)
        print(f"  [student {cfg_name}] test features done ({len(te_f)} samples)", flush=True)
        print(f"  [student {cfg_name}] fitting probes...", flush=True)
        mean_auc, per_cond = run_auroc_probe(tr_f, tr_l, te_f, te_l, label_names, device)
        auroc_results[f"student_{cfg_name}"] = {"mean_auroc": round(mean_auc, 4), "per_condition": per_cond}
        print(f"    mean AUROC = {mean_auc:.4f}")

    train_labels = np.concatenate([_get_labels(train_raw), _get_labels(val_raw)])
    test_labels  = _get_labels(test_ds)
    for cfg_name, wb, ab in _EVAL_CONFIGS:
        if wb is None:
            # FP features were already cached above — reuse instead of a
            # second forward pass over the full teacher model.
            tr_f = torch.stack([teacher_cache[i] for i in range(N)]).numpy()
            te_f = torch.stack([test_cache[i]    for i in range(n_te)]).numpy()
            tr_l, te_l = train_labels, test_labels
            print(f"\n  [teacher {cfg_name}] using cached FP features "
                  f"(train={len(tr_f)}, test={len(te_f)})", flush=True)
        else:
            print(f"\n  [teacher {cfg_name}] computing train features...", flush=True)
            tr_f, tr_l = extract_features(teacher, train_eval_loader, device, AMP_DTYPE, wb, ab)
            print(f"  [teacher {cfg_name}] train features done ({len(tr_f)} samples)", flush=True)
            print(f"  [teacher {cfg_name}] computing test features...", flush=True)
            te_f, te_l = extract_features(teacher, test_loader,       device, AMP_DTYPE, wb, ab)
            print(f"  [teacher {cfg_name}] test features done ({len(te_f)} samples)", flush=True)
        print(f"  [teacher {cfg_name}] fitting probes...", flush=True)
        mean_auc, per_cond = run_auroc_probe(tr_f, tr_l, te_f, te_l, label_names, device)
        auroc_results[f"teacher_{cfg_name}"] = {"mean_auroc": round(mean_auc, 4), "per_condition": per_cond}
        print(f"    mean AUROC = {mean_auc:.4f}")

    results = {
        "script": "run_qit_merlin_fulldata",
        "n_train_val": N,
        "n_test": len(test_ds),
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "grad_accum": GRAD_ACCUM,
        "lr": LR,
        "w_bits_range": [W_BITS_MIN, W_BITS_MAX],
        "a_bits_range": [A_BITS_MIN, A_BITS_MAX],
        "label_names": label_names,
        "auroc": auroc_results,
    }
    with open(out / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {out / 'results.json'}")


if __name__ == "__main__":
    main()
