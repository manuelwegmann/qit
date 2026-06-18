"""
Linear-probe AUROC for all Merlin conditions.

Expects a feature directory produced by cache_merlin_features.py:
    {feat_dir}/
        label_names.json
        {split}_labels.npy          (train / val / test)
        {cfg}_{split}_feats.npy     one file per config × split

Training regime
----------------
No early stopping. Early stopping on val loss was dropped because several
conditions have very few labeled val examples (see docs/merlin_label_stats.md),
so per-epoch val-loss comparisons are noisy and can trigger spurious stops.

Instead: train for a fixed number of epochs (cosine LR to 0) at each of
several weight-decay strengths, and pick the weight decay with the lowest
mean val loss across seeds. This is the standard SSL linear-probe recipe
(SimCLR/MoCo-style) — regularization strength is the only thing tuned
against val, as a single coarse decision rather than hundreds of per-epoch
ones.

Class imbalance: many conditions are heavily skewed (see
docs/merlin_label_stats.md, e.g. free_air 27 pos / 641 neg in train). Each
condition's BCE term is weighted by pos_weight = n_neg / n_pos, computed
from the train split, matching torch.nn.BCEWithLogitsLoss convention. This
is applied in both the training loss and the val loss used for weight-decay
selection, so the selection criterion matches what's being optimized.

Condition selection: a condition is only reported for a given config if
min(test_pos, test_neg) >= min_test_per_class (default 20). AUROC precision
is bottlenecked by the smaller class, not the total labeled count, so this
is checked per-class rather than on pos+neg combined. The threshold is fixed
before looking at any results to avoid selection bias. See
docs/merlin_label_stats.md for full counts; at threshold 20 this keeps
11/30 conditions.

Results are written incrementally so the job can be resumed.

Usage:
    python scripts/run_probe_merlin.py \\
        --feat_dir runs/qit_merlin_fulldata/features \\
        --output_dir runs/qit_merlin_fulldata/probe_results

    python scripts/run_probe_merlin.py \\
        --feat_dir runs/teacher_features \\
        --output_dir runs/teacher_features/probe_results
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

_CONFIG_ORDER          = ["fp", "w8a8", "w4a8", "w4a4"]
_DEFAULT_WEIGHT_DECAYS = [0.0, 1e-4, 1e-3, 1e-2, 1e-1]


def _compute_pos_weight(train_labels):
    """Per-condition pos_weight = n_neg / n_pos from labeled (non -1) train entries,
    matching torch.nn.BCEWithLogitsLoss convention. Conditions with zero train
    positives get weight 1.0 (no positives to weight; excluded by the test-set
    filter anyway)."""
    pos_weight = np.ones(train_labels.shape[1], dtype=np.float32)
    for c in range(train_labels.shape[1]):
        col = train_labels[:, c]
        n_pos = (col == 1).sum()
        n_neg = (col == 0).sum()
        if n_pos > 0:
            pos_weight[c] = n_neg / n_pos
    return pos_weight


def _train_one(X_tr, y_tr, m_tr, X_val_dev, y_val_dev, m_val_dev, X_te_dev,
               device, epochs, lr, batch_size, weight_decay, seed, pos_weight):
    """Train a linear head for a fixed number of epochs (no early stopping).
    Returns (val_loss, test_preds) from the final epoch.
    """
    torch.manual_seed(seed)
    head = nn.Linear(X_tr.shape[1], y_tr.shape[1]).to(device)
    opt  = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=weight_decay)
    sch  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=0)
    dl   = DataLoader(
        TensorDataset(X_tr.to(device), y_tr.to(device), m_tr.to(device)),
        batch_size=batch_size, shuffle=True)

    for _ in range(epochs):
        head.train()
        for xb, yb, mb in dl:
            opt.zero_grad()
            raw  = F.binary_cross_entropy_with_logits(
                head(xb), yb, pos_weight=pos_weight, reduction="none")
            loss = (raw * mb).sum() / mb.sum().clamp(min=1)
            loss.backward()
            opt.step()
        sch.step()

    head.eval()
    with torch.no_grad():
        raw_val  = F.binary_cross_entropy_with_logits(
            head(X_val_dev), y_val_dev, pos_weight=pos_weight, reduction="none")
        val_loss = ((raw_val * m_val_dev).sum() / m_val_dev.sum().clamp(min=1)).item()
        preds    = torch.sigmoid(head(X_te_dev)).cpu().numpy()

    return val_loss, preds


def run_auroc_probe(train_feats, train_labels, val_feats, val_labels,
                    test_feats, test_labels, label_names, device,
                    epochs=300, lr=1e-3, batch_size=256, n_seeds=5,
                    min_test_per_class=20, weight_decays=_DEFAULT_WEIGHT_DECAYS):
    from sklearn.metrics import roc_auc_score

    X_tr  = torch.tensor(train_feats, dtype=torch.float32)
    y_tr  = torch.tensor(np.clip(train_labels, 0, 1).astype(np.float32))
    m_tr  = torch.tensor((train_labels >= 0).astype(np.float32))
    X_val = torch.tensor(val_feats,   dtype=torch.float32)
    y_val = torch.tensor(np.clip(val_labels,   0, 1).astype(np.float32))
    m_val = torch.tensor((val_labels   >= 0).astype(np.float32))
    X_te  = torch.tensor(test_feats,  dtype=torch.float32)

    mu, std = X_tr.mean(0), X_tr.std(0).clamp(min=1e-8)
    X_tr  = (X_tr  - mu) / std
    X_val = (X_val - mu) / std
    X_te  = (X_te  - mu) / std

    valid_conds = [
        c for c, _ in enumerate(label_names)
        if min((test_labels[:, c] == 1).sum(), (test_labels[:, c] == 0).sum())
        >= min_test_per_class
    ]
    print(f"    {len(valid_conds)}/{len(label_names)} conditions pass "
          f"min(test_pos,test_neg)>={min_test_per_class} filter", flush=True)

    X_val_dev = X_val.to(device)
    y_val_dev = y_val.to(device)
    m_val_dev = m_val.to(device)
    X_te_dev  = X_te.to(device)

    pos_weight = torch.tensor(_compute_pos_weight(train_labels), device=device)

    wd_runs = {}   # weight_decay -> {"val_losses": [...], "per_cond_runs": {...}, "mean_aucs": [...]}
    for wd in weight_decays:
        per_cond_runs = {label_names[c]: [] for c in valid_conds}
        mean_aucs, val_losses = [], []

        for seed in range(n_seeds):
            val_loss, preds = _train_one(
                X_tr, y_tr, m_tr, X_val_dev, y_val_dev, m_val_dev, X_te_dev,
                device, epochs, lr, batch_size, wd, seed, pos_weight)
            val_losses.append(val_loss)

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
                mean_aucs.append(float(np.mean(seed_aucs)))

        mean_val_loss = float(np.mean(val_losses))
        print(f"      wd={wd:<8g}  mean_val_loss={mean_val_loss:.4f}  "
              f"mean_test_auc={np.mean(mean_aucs):.4f}", flush=True)
        wd_runs[wd] = {
            "mean_val_loss": mean_val_loss,
            "per_cond_runs": per_cond_runs,
            "mean_aucs":     mean_aucs,
        }

    best_wd = min(wd_runs, key=lambda w: wd_runs[w]["mean_val_loss"])
    chosen  = wd_runs[best_wd]
    print(f"    selected weight_decay={best_wd:g} (lowest mean val loss)", flush=True)

    per_cond     = {n: float(np.mean(v)) for n, v in chosen["per_cond_runs"].items() if v}
    per_cond_std = {n: float(np.std(v))  for n, v in chosen["per_cond_runs"].items() if v}
    mean_aucs    = chosen["mean_aucs"]
    mean_auc     = float(np.mean(mean_aucs)) if mean_aucs else float("nan")
    std_auc      = float(np.std(mean_aucs))  if mean_aucs else float("nan")

    wd_summary = {str(w): round(r["mean_val_loss"], 4) for w, r in wd_runs.items()}
    return mean_auc, per_cond, std_auc, per_cond_std, mean_aucs, best_wd, wd_summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feat_dir", required=True,
                        help="Directory produced by cache_merlin_features.py.")
    parser.add_argument("--configs", nargs="+", choices=_CONFIG_ORDER,
                        default=_CONFIG_ORDER)
    parser.add_argument("--min_test_per_class", type=int, default=20,
                        help="Minimum of (test_pos, test_neg) required to report a "
                             "condition. Decided a priori, not from observed results.")
    parser.add_argument("--probe_epochs",  type=int,   default=200,
                        help="Fixed epoch budget (no early stopping).")
    parser.add_argument("--probe_seeds",   type=int,   default=5)
    parser.add_argument("--probe_lr",      type=float, default=1e-3)
    parser.add_argument("--weight_decays", type=float, nargs="+",
                        default=_DEFAULT_WEIGHT_DECAYS,
                        help="Grid of weight-decay values; best chosen by mean val loss.")
    parser.add_argument("--output_dir", default=None,
                        help="Where to write results.json. Defaults to {feat_dir}/probe_results.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Recompute configs already in results.json.")
    args = parser.parse_args()

    feat_dir = Path(args.feat_dir)
    if not feat_dir.exists():
        raise FileNotFoundError(f"Feature directory not found: {feat_dir}")

    out = Path(args.output_dir) if args.output_dir else feat_dir / "probe_results"
    out.mkdir(parents=True, exist_ok=True)
    results_path = out / "results.json"

    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    label_names = json.load(open(feat_dir / "label_names.json"))

    print("=" * 70)
    print(f"Linear-probe AUROC — {feat_dir.name}")
    print(f"  configs          : {args.configs}")
    print(f"  probe            : {args.probe_epochs} fixed epochs (no early stopping), "
          f"{args.probe_seeds} seeds, lr={args.probe_lr}")
    print(f"  weight_decays    : {args.weight_decays}")
    print(f"  min_test_per_class : {args.min_test_per_class}")
    print(f"  output           : {results_path}")
    print("=" * 70)

    if results_path.exists():
        with open(results_path) as f:
            results = json.load(f)
    else:
        results = {
            "feat_dir":         str(feat_dir),
            "probe_epochs":     args.probe_epochs,
            "probe_seeds":      args.probe_seeds,
            "weight_decays":      args.weight_decays,
            "min_test_per_class": args.min_test_per_class,
            "results":            {},
        }

    train_labels = np.load(feat_dir / "train_labels.npy")
    val_labels   = np.load(feat_dir / "val_labels.npy")
    test_labels  = np.load(feat_dir / "test_labels.npy")

    for cfg_name in args.configs:
        if not args.overwrite and cfg_name in results["results"]:
            print(f"\n[{cfg_name}] already done — skipping.")
            continue

        print(f"\n[{cfg_name}] loading features...", flush=True)
        train_feats = np.load(feat_dir / f"{cfg_name}_train_feats.npy")
        val_feats   = np.load(feat_dir / f"{cfg_name}_val_feats.npy")
        test_feats  = np.load(feat_dir / f"{cfg_name}_test_feats.npy")
        print(f"  train={train_feats.shape}  val={val_feats.shape}  test={test_feats.shape}")

        print(f"[{cfg_name}] fitting probes "
              f"({len(args.weight_decays)} weight decays × {args.probe_seeds} seeds × "
              f"{args.probe_epochs} epochs)...", flush=True)
        mean_auc, per_cond, std_auc, per_cond_std, seed_aucs, best_wd, wd_summary = run_auroc_probe(
            train_feats, train_labels,
            val_feats,   val_labels,
            test_feats,  test_labels,
            label_names, device,
            epochs=args.probe_epochs, lr=args.probe_lr,
            batch_size=256, n_seeds=args.probe_seeds,
            min_test_per_class=args.min_test_per_class,
            weight_decays=args.weight_decays)

        results["results"][cfg_name] = {
            "mean_auroc":        round(mean_auc, 4),
            "std_auroc":         round(std_auc, 4),
            "selected_weight_decay": best_wd,
            "weight_decay_val_loss": wd_summary,
            "per_condition":     {k: round(v, 4) for k, v in per_cond.items()},
            "per_condition_std": {k: round(v, 4) for k, v in per_cond_std.items()},
            "seed_aucs":         [round(v, 4) for v in seed_aucs],
        }
        print(f"  mean AUROC = {mean_auc:.4f} ± {std_auc:.4f}  ({len(per_cond)} conditions)  "
              f"[weight_decay={best_wd:g}]")
        for cond, auc in sorted(per_cond.items(), key=lambda x: -x[1]):
            print(f"    {cond:<38} {auc:.4f} ± {per_cond_std.get(cond, 0):.4f}")

        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

    print(f"\nResults → {results_path}")


if __name__ == "__main__":
    main()
