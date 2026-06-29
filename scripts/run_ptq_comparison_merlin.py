"""
PTQ calibration comparison for Merlin QIT.

Compares dataset-calibrated static quantization ("minmax" and percentile-clipped
"pctN", via calibrate_activations / static_quantized_forward) against the dynamic
per-sample quantization used in the main eval, for the QIT student and the frozen
teacher across w8a8/w4a8/w4a4. calibrate_activations / static_quantized_forward
(models/quantization.py) already support the Conv3d layers in Merlin's
I3ResNet-152, so no changes to models/quantization.py were needed.

Uses the same train/val/test split as qit_merlin_unfrozen_bn (n_train=2000,
n_val=500, n_test=757, seed=42, target_shape=160x224x224) and the same
weight_granularity ("per_channel"), so results are directly comparable to the
"dynamic" numbers in runs/qit_merlin_unfrozen_bn_eval/results.json.

Results are written incrementally (one (model, config, mode) combo at a time) and
existing combos in results.json are skipped on rerun unless --overwrite is given —
this lets the work be split across multiple SLURM jobs via --models/--configs/--modes.

By default only the percentile-clipped "pct<N>" mode is computed (2 models x 3
configs = 6 extraction passes); pass --modes minmax pct to also add full-range
static calibration.

Usage:
    python scripts/run_ptq_comparison_merlin.py \\
        --student_ckpt runs/qit_merlin_unfrozen_bn/checkpoint_best.pt \\
        --output_dir runs/ptq_comparison_merlin

    # Split across jobs, e.g. teacher-only:
    python scripts/run_ptq_comparison_merlin.py \\
        --models teacher --output_dir runs/ptq_comparison_merlin
"""

import argparse
import json
import sys
import warnings
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

_ROOT = Path(__file__).parent.parent
_QFT  = _ROOT.parent / "quantized_ft"
sys.path.insert(0, str(_ROOT))
sys.path.insert(1, str(_QFT))

from models.quantization import calibrate_activations, quantized_forward, static_quantized_forward
from models.eval_utils import _pbar, MerlinEncoder
from models.merlin_utils import (
    ResizedDataset as _ResizedDataset,
    DEFAULT_DATA_DIR as _DEFAULT_DATA_DIR,
    DEFAULT_REPORTS as _DEFAULT_REPORTS,
    DEFAULT_LABELS as _DEFAULT_LABELS,
    DEFAULT_METADATA as _DEFAULT_METADATA,
)
from downstream.dataset import MerlinDataset

warnings.filterwarnings("ignore", category=UserWarning)

_QUANT_CONFIGS = {"w8a8": (8, 8), "w4a8": (4, 8), "w4a4": (4, 4)}
_CONFIG_ORDER  = ["w8a8", "w4a8", "w4a4"]


# ---------------------------------------------------------------------------
# Feature extraction (with optional static calibration)
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_features(model, loader, device, use_amp, amp_dtype,
                     w_bits, a_bits, weight_granularity, calib_stats):
    """Returns (features_np, labels_np) under static (calibrated) quantization."""
    model.eval()
    all_f, all_y = [], []

    for x, y in _pbar(loader, desc="    extracting", leave=False):
        x = x.to(device)
        ctx = torch.amp.autocast("cuda", dtype=amp_dtype) if use_amp else nullcontext()
        with ctx:
            with static_quantized_forward([model.image_encoder], w_bits, a_bits,
                                          calib_stats, weight_granularity):
                f = model(x)
        all_f.append(f.cpu().float())
        all_y.append(y)

    return torch.cat(all_f).numpy(), torch.cat(all_y).numpy()


# ---------------------------------------------------------------------------
# AUROC probe (mirrors run_qit_merlin.py's run_auroc_probe)
# ---------------------------------------------------------------------------

def run_auroc_probe(train_feats, train_labels, val_feats, val_labels,
                    test_feats, test_labels,
                    label_names, device, epochs=300, lr=1e-3, batch_size=256, n_seeds=5):
    """Masked-BCE linear probe for multi-label AUROC, averaged over n_seeds probes.

    Returns (mean_auroc, per_condition_auroc_dict, std_auroc, per_condition_std_dict, seed_aucs).
    """
    from sklearn.metrics import roc_auc_score

    X_tr  = torch.tensor(train_feats, dtype=torch.float32)
    y_tr  = torch.tensor(np.clip(train_labels, 0, 1).astype(np.float32))
    m_tr  = torch.tensor((train_labels >= 0).astype(np.float32))
    X_val = torch.tensor(val_feats, dtype=torch.float32)
    y_val = torch.tensor(np.clip(val_labels, 0, 1).astype(np.float32))
    m_val = torch.tensor((val_labels >= 0).astype(np.float32))
    X_te  = torch.tensor(test_feats,  dtype=torch.float32)

    mu  = X_tr.mean(0)
    std = X_tr.std(0).clamp(min=1e-8)
    X_tr  = (X_tr  - mu) / std
    X_val = (X_val - mu) / std
    X_te  = (X_te  - mu) / std

    n_cond = y_tr.shape[1]

    valid_conds = [
        c for c in range(n_cond)
        if (test_labels[:, c] >= 0).sum() >= 10
        and np.ptp(test_labels[test_labels[:, c] >= 0, c]) > 0
    ]

    per_cond_runs = {label_names[c]: [] for c in valid_conds}
    mean_aucs = []

    X_val_dev = X_val.to(device)
    y_val_dev = y_val.to(device)
    m_val_dev = m_val.to(device)

    for seed in range(n_seeds):
        torch.manual_seed(seed)
        head = nn.Linear(X_tr.shape[1], n_cond).to(device)
        opt  = torch.optim.Adam(head.parameters(), lr=lr)
        sch  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=0)

        ds     = torch.utils.data.TensorDataset(X_tr.to(device), y_tr.to(device), m_tr.to(device))
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

        best_val_loss, no_imp = float("inf"), 0
        best_state = {k: v.clone() for k, v in head.state_dict().items()}
        for _ in range(epochs):
            head.train()
            for xb, yb, mb in loader:
                opt.zero_grad()
                raw  = F.binary_cross_entropy_with_logits(head(xb), yb, reduction="none")
                loss = (raw * mb).sum() / mb.sum().clamp(min=1)
                loss.backward()
                opt.step()
            sch.step()

            head.eval()
            with torch.no_grad():
                raw_val  = F.binary_cross_entropy_with_logits(head(X_val_dev), y_val_dev, reduction="none")
                val_loss = ((raw_val * m_val_dev).sum() / m_val_dev.sum().clamp(min=1)).item()
            if best_val_loss - val_loss > 1e-5:
                best_val_loss, no_imp = val_loss, 0
                best_state = {k: v.clone() for k, v in head.state_dict().items()}
            else:
                no_imp += 1
            if no_imp >= 30:
                break

        head.load_state_dict(best_state)
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

    per_cond     = {name: float(np.mean(vals)) for name, vals in per_cond_runs.items() if vals}
    per_cond_std = {name: float(np.std(vals))  for name, vals in per_cond_runs.items() if vals}
    mean_auroc   = float(np.mean(mean_aucs)) if mean_aucs else float("nan")
    std_auroc    = float(np.std(mean_aucs))  if mean_aucs else float("nan")
    return mean_auroc, per_cond, std_auroc, per_cond_std, mean_aucs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",  default=_DEFAULT_DATA_DIR)
    parser.add_argument("--reports",   default=_DEFAULT_REPORTS)
    parser.add_argument("--labels",    default=_DEFAULT_LABELS)
    parser.add_argument("--metadata",  default=_DEFAULT_METADATA)
    parser.add_argument("--conditions", nargs="+",
                        default=["atelectasis", "pleural_effusion", "renal_cyst"])
    parser.add_argument("--n_train", type=int, default=2000,
                        help="Must match the QIT run's --n_train (seed=42 subset) "
                             "for a like-for-like split.")
    parser.add_argument("--n_val",   type=int, default=500,
                        help="Must match the QIT run's --n_val (seed=42 subset).")
    parser.add_argument("--target_shape", type=int, nargs=3, default=[160, 224, 224],
                        metavar=("D", "H", "W"))

    parser.add_argument("--student_ckpt", default="runs/qit_merlin_unfrozen_bn/checkpoint_best.pt",
                        help="Checkpoint (key='student') for the QIT student model.")
    parser.add_argument("--models",  nargs="+", choices=["student", "teacher"],
                        default=["student", "teacher"])
    parser.add_argument("--configs", nargs="+", choices=_CONFIG_ORDER, default=_CONFIG_ORDER)
    parser.add_argument("--modes",   nargs="+", choices=["minmax", "pct"], default=["pct"],
                        help="'minmax' = full-range static calibration, "
                             "'pct' = percentile-clipped static calibration (--percentile). "
                             "Default: pct only.")
    parser.add_argument("--percentile", type=float, default=99.99,
                        help="Percentile for activation clipping (CNNs: 99.99).")
    parser.add_argument("--n_calib_batches", type=int, default=32)
    parser.add_argument("--calib_batch_size", type=int, default=16)
    parser.add_argument("--weight_granularity", default="per_channel",
                        choices=["per_tensor", "per_channel"],
                        help="Must match the QIT run's granularity ('per_channel') "
                             "for apples-to-apples comparison with dynamic-mode results.")

    parser.add_argument("--batch_size",  type=int, default=16)
    parser.add_argument("--no_amp",      action="store_true")
    parser.add_argument("--amp_dtype",   default="bf16", choices=["fp16", "bf16"])
    parser.add_argument("--output_dir",  default="runs/ptq_comparison_merlin")
    parser.add_argument("--overwrite",   action="store_true",
                        help="Recompute (model, config, mode) combos already in results.json.")
    args = parser.parse_args()

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp   = not args.no_amp and device.type == "cuda"
    amp_dtype = torch.float16 if args.amp_dtype == "fp16" else torch.bfloat16

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    results_path = out / "results.json"

    pct_label = f"pct{args.percentile:.4g}"
    mode_labels = []
    if "minmax" in args.modes:
        mode_labels.append("minmax")
    if "pct" in args.modes:
        mode_labels.append(pct_label)

    print("=" * 70)
    print("PTQ calibration comparison — Merlin QIT")
    print(f"  models  : {args.models}")
    print(f"  configs : {args.configs}")
    print(f"  modes   : {mode_labels}")
    print(f"  calib   : {args.n_calib_batches} batches x bs={args.calib_batch_size}, "
          f"percentile={args.percentile}, granularity={args.weight_granularity}")
    print(f"  output  : {results_path}")
    print("=" * 70)

    # ── load / init results ─────────────────────────────────────────────────
    if results_path.exists():
        with open(results_path) as f:
            results = json.load(f)
    else:
        results = {
            "model":              "merlin",
            "student_ckpt":       args.student_ckpt,
            "n_train":            args.n_train,
            "n_val":              args.n_val,
            "weight_granularity": args.weight_granularity,
            "n_calib_batches":    args.n_calib_batches,
            "calib_batch_size":   args.calib_batch_size,
            "percentile":         args.percentile,
            "results":            {},
        }

    def _needs(model_key, cfg_name, mode_name):
        return args.overwrite or mode_name not in results["results"].get(f"{model_key}_{cfg_name}", {})

    pending = [(m, c, mo) for m in args.models for c in args.configs for mo in mode_labels
               if _needs(m, c, mo)]
    if not pending:
        print("\nNothing to do — all requested (model, config, mode) combos already in results.json.")
        return

    # ── datasets ─────────────────────────────────────────────────────────────
    print("\nIndexing dataset...")
    meta_file = args.metadata if Path(args.metadata).exists() else None
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

    target = tuple(args.target_shape)
    train_ds = _ResizedDataset(train_ds, target)
    val_ds   = _ResizedDataset(val_ds,   target)
    test_ds  = _ResizedDataset(test_ds,  target)

    print(f"  train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}")
    print(f"  conditions ({len(label_names)}): {', '.join(label_names)}\n")

    # num_workers=0 runs loading in the main process — avoids /tmp IPC socket
    # exhaustion on shared nodes with many concurrent jobs.
    eval_loader_kw = dict(batch_size=args.batch_size, shuffle=False,
                          num_workers=0, pin_memory=False)
    train_loader = DataLoader(train_ds, **eval_loader_kw)
    val_loader   = DataLoader(val_ds,   **eval_loader_kw)
    test_loader  = DataLoader(test_ds,  **eval_loader_kw)

    calib_loader = DataLoader(train_ds, batch_size=args.calib_batch_size,
                              shuffle=False, num_workers=0, pin_memory=False)

    # ── evaluate each model ────────────────────────────────────────────────
    for model_key in args.models:
        configs_needed = [c for c in args.configs if any(_needs(model_key, c, mo) for mo in mode_labels)]
        if not configs_needed:
            print(f"\n[{model_key}] all requested modes already computed — skipping.")
            continue

        print(f"\nLoading {model_key} model...")
        model = MerlinEncoder().to(device)
        for p in model.parameters():
            p.requires_grad_(False)
        if model_key == "student":
            ckpt = torch.load(args.student_ckpt, map_location=device)
            model.load_state_dict(ckpt["student"])
        model.eval()

        stats = {}
        if "minmax" in mode_labels:
            print(f"\n  [{model_key}] calibrating minmax ({args.n_calib_batches} batches)...", flush=True)
            stats["minmax"] = calibrate_activations(model, calib_loader, args.n_calib_batches,
                                                      device, percentile=100.0, use_amp=use_amp)
        if pct_label in mode_labels:
            print(f"  [{model_key}] calibrating {pct_label} ({args.n_calib_batches} batches)...", flush=True)
            stats[pct_label] = calibrate_activations(model, calib_loader, args.n_calib_batches,
                                                       device, percentile=args.percentile, use_amp=use_amp)

        for cfg_name in configs_needed:
            wb, ab = _QUANT_CONFIGS[cfg_name]
            key = f"{model_key}_{cfg_name}"
            results["results"].setdefault(key, {})

            for mode_name in mode_labels:
                if not _needs(model_key, cfg_name, mode_name):
                    print(f"\n  [{key}] [{mode_name}] already computed — skipping.")
                    continue

                calib_stats = stats[mode_name]
                print(f"\n  [{key}] [{mode_name}] computing train features...", flush=True)
                tr_f, tr_l = extract_features(model, train_loader, device, use_amp, amp_dtype,
                                              wb, ab, args.weight_granularity, calib_stats)
                print(f"  [{key}] [{mode_name}] computing val features...", flush=True)
                va_f, va_l = extract_features(model, val_loader, device, use_amp, amp_dtype,
                                              wb, ab, args.weight_granularity, calib_stats)
                print(f"  [{key}] [{mode_name}] computing test features...", flush=True)
                te_f, te_l = extract_features(model, test_loader, device, use_amp, amp_dtype,
                                              wb, ab, args.weight_granularity, calib_stats)

                print(f"  [{key}] [{mode_name}] fitting probes...", flush=True)
                mean_auc, per_cond, std_auc, per_cond_std, seed_aucs = run_auroc_probe(
                    tr_f, tr_l, va_f, va_l, te_f, te_l, label_names, device)
                results["results"][key][mode_name] = {
                    "mean_auroc": round(mean_auc, 4),
                    "std_auroc": round(std_auc, 4),
                    "per_condition": per_cond,
                    "per_condition_std": {k: round(v, 4) for k, v in per_cond_std.items()},
                    "seed_aucs": [round(v, 4) for v in seed_aucs],
                }
                print(f"    mean AUROC = {mean_auc:.4f} ± {std_auc:.4f}")

                with open(results_path, "w") as f:
                    json.dump(results, f, indent=2)

    print(f"\nResults → {results_path}")


if __name__ == "__main__":
    main()
