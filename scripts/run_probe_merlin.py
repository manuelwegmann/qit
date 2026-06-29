"""
Linear-probe AUROC for Merlin conditions (per-condition early stopping).

Each condition gets its own independent single-output linear probe
(nn.Linear(d, 1)), trained on frozen pre-cached features and early-stopped on
*that condition's own* validation AUROC. The reported "joint mean AUROC" is the
mean of these independently early-stopped per-condition AUROCs. Early stopping
is the regularizer (no weight-decay grid search).

Expects a feature directory produced by cache_merlin_features.py:
    {feat_dir}/
        label_names.json
        {split}_labels.npy          (train / val / test)
        {cfg}_{split}_feats.npy     one file per config × split

Cost: linear probes on top of frozen, pre-cached features (no backbone forward
pass) — d=2048, ~3.5k train rows, ~2k params per probe. Looping over the
reported conditions × n_seeds is GPU-seconds, not minutes.

Class imbalance: many conditions are heavily skewed (see
docs/merlin_label_stats.md). Each condition's BCE term is weighted by
pos_weight = n_neg / n_pos from the train split (torch.nn.BCEWithLogitsLoss
convention).

Condition selection: a condition is only reported for a given config if
min(test_pos, test_neg) >= min_test_per_class (default 20), fixed a priori.
AUROC precision is bottlenecked by the smaller class, so this is checked
per-class. See docs/merlin_label_stats.md; at threshold 20 this keeps 11/30
conditions.

Caveat: several conditions have very few labeled val examples, so per-epoch val
AUROC can be noisy for those and early stopping inherits that noise.

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
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

_CONFIG_ORDER = ["fp", "w8a8", "w6a6", "w5a5", "w4a8", "w4a4", "w3a6", "w2a8", "w2a4"]


def _compute_pos_weight(train_labels):
    """Per-condition pos_weight = n_neg / n_pos from labeled (non -1) train entries,
    matching torch.nn.BCEWithLogitsLoss convention."""
    pos_weight = np.ones(train_labels.shape[1], dtype=np.float32)
    for c in range(train_labels.shape[1]):
        col = train_labels[:, c]
        n_pos, n_neg = (col == 1).sum(), (col == 0).sum()
        if n_pos > 0:
            pos_weight[c] = n_neg / n_pos
    return pos_weight


def _train_one_condition(X_tr, y_tr, m_tr, X_val_dev, y_val_np, m_val_np, X_te_dev,
                         device, max_epochs, lr, batch_size, weight_decay,
                         seed, pos_weight, es_patience, es_tol):
    """Independent single-output linear probe for one condition, early-stopped on
    that condition's own validation AUROC. Returns (best_val_auc, test_preds)."""
    torch.manual_seed(seed)
    head = nn.Linear(X_tr.shape[1], 1).to(device)
    opt  = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=weight_decay)
    dl   = DataLoader(TensorDataset(X_tr.to(device), y_tr.to(device), m_tr.to(device)),
                      batch_size=batch_size, shuffle=True)

    best_val_auc, no_improve = -1.0, 0
    best_state = {k: v.clone() for k, v in head.state_dict().items()}

    for _ in range(max_epochs):
        head.train()
        for xb, yb, mb in dl:
            opt.zero_grad()
            raw  = F.binary_cross_entropy_with_logits(head(xb), yb, pos_weight=pos_weight,
                                                       reduction="none")
            loss = (raw * mb).sum() / mb.sum().clamp(min=1)
            loss.backward()
            opt.step()

        head.eval()
        with torch.no_grad():
            val_scores = torch.sigmoid(head(X_val_dev)).cpu().numpy().reshape(-1)
        try:
            val_auc = roc_auc_score(y_val_np[m_val_np], val_scores[m_val_np])
        except ValueError:
            val_auc = float("nan")  # degenerate val fold (single class for this condition)

        if val_auc - best_val_auc > es_tol:
            best_val_auc, no_improve = val_auc, 0
            best_state = {k: v.clone() for k, v in head.state_dict().items()}
        else:
            no_improve += 1
        if no_improve >= es_patience:
            break

    # If val AUROC was never defined (e.g. condition has only one class in val
    # for this seed's shuffling), best_state is still the random init — keep
    # the final-epoch weights instead of falling back to that.
    if best_val_auc > -1.0:
        head.load_state_dict(best_state)
    head.eval()
    with torch.no_grad():
        test_preds = torch.sigmoid(head(X_te_dev)).cpu().numpy().reshape(-1)
    return best_val_auc, test_preds


def run_auroc_probe(train_feats, train_labels, val_feats, val_labels,
                    test_feats, test_labels, label_names, device,
                    max_epochs=300, lr=1e-3, batch_size=256, n_seeds=5,
                    weight_decay=0.0, es_patience=20, es_tol=1e-3,
                    min_test_per_class=20):
    X_tr  = torch.tensor(train_feats, dtype=torch.float32)
    X_val = torch.tensor(val_feats,   dtype=torch.float32)
    X_te  = torch.tensor(test_feats,  dtype=torch.float32)
    mu, std = X_tr.mean(0), X_tr.std(0).clamp(min=1e-8)
    X_tr, X_val, X_te = (X_tr - mu) / std, (X_val - mu) / std, (X_te - mu) / std
    X_val_dev, X_te_dev = X_val.to(device), X_te.to(device)

    valid_conds = [
        c for c, _ in enumerate(label_names)
        if min((test_labels[:, c] == 1).sum(), (test_labels[:, c] == 0).sum())
        >= min_test_per_class
    ]
    print(f"    {len(valid_conds)}/{len(label_names)} conditions pass "
          f"min(test_pos,test_neg)>={min_test_per_class} filter", flush=True)

    pos_weight_all = _compute_pos_weight(train_labels)

    per_cond_runs = {label_names[c]: [] for c in valid_conds}
    per_cond_val  = {label_names[c]: [] for c in valid_conds}
    test_auc_matrix = []  # one row per condition, one column per seed

    for c in valid_conds:
        y_tr_c   = torch.tensor(np.clip(train_labels[:, c], 0, 1).reshape(-1, 1), dtype=torch.float32)
        m_tr_c   = torch.tensor((train_labels[:, c] >= 0).astype(np.float32).reshape(-1, 1))
        y_val_np = np.clip(val_labels[:, c], 0, 1).astype(np.float32)
        m_val_np = val_labels[:, c] >= 0
        pos_weight = torch.tensor([pos_weight_all[c]], device=device)

        seed_test_aucs = []
        for seed in range(n_seeds):
            best_val_auc, preds = _train_one_condition(
                X_tr, y_tr_c, m_tr_c, X_val_dev, y_val_np, m_val_np, X_te_dev,
                device, max_epochs, lr, batch_size, weight_decay, seed,
                pos_weight, es_patience, es_tol)

            mask = test_labels[:, c] >= 0
            try:
                test_auc = float(roc_auc_score(test_labels[mask, c], preds[mask]))
            except ValueError:
                continue
            per_cond_runs[label_names[c]].append(test_auc)
            per_cond_val[label_names[c]].append(best_val_auc)
            seed_test_aucs.append(test_auc)

        test_auc_matrix.append(seed_test_aucs)
        if seed_test_aucs:
            print(f"      {label_names[c]:<38} test_auc={np.mean(seed_test_aucs):.4f} "
                  f"± {np.std(seed_test_aucs):.4f}  (val_auc={np.mean(per_cond_val[label_names[c]]):.4f})",
                  flush=True)

    # Joint mean AUROC per seed = mean across conditions for that seed slot.
    n_seeds_completed = min((len(row) for row in test_auc_matrix), default=0)
    mean_aucs = [
        float(np.mean([row[s] for row in test_auc_matrix]))
        for s in range(n_seeds_completed)
    ]

    per_cond          = {n: float(np.mean(v)) for n, v in per_cond_runs.items() if v}
    per_cond_std       = {n: float(np.std(v))  for n, v in per_cond_runs.items() if v}
    per_cond_val_mean = {n: float(np.mean(v)) for n, v in per_cond_val.items() if v}
    mean_auc = float(np.mean(mean_aucs)) if mean_aucs else float("nan")
    std_auc  = float(np.std(mean_aucs))  if mean_aucs else float("nan")

    return mean_auc, per_cond, std_auc, per_cond_std, mean_aucs, per_cond_val_mean


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feat_dir", required=True,
                        help="Directory produced by cache_merlin_features.py.")
    parser.add_argument("--configs", nargs="+", choices=_CONFIG_ORDER,
                        default=_CONFIG_ORDER)
    parser.add_argument("--min_test_per_class", type=int, default=20,
                        help="Minimum of (test_pos, test_neg) required to report a "
                             "condition. Decided a priori, not from observed results.")
    parser.add_argument("--probe_max_epochs", type=int, default=300,
                        help="Upper bound on epochs; early stopping should trigger well "
                             "before this for most conditions.")
    parser.add_argument("--probe_seeds", type=int, default=5)
    parser.add_argument("--probe_lr",    type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0,
                        help="Fixed weight decay. Early stopping is the primary "
                             "regularizer here, so this isn't grid-searched.")
    parser.add_argument("--es_patience", type=int, default=20,
                        help="Epochs without val-AUROC improvement before stopping "
                             "that condition's probe.")
    parser.add_argument("--es_tol", type=float, default=1e-3,
                        help="Minimum val-AUROC improvement (AUROC scale) to reset patience.")
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
    print(f"Linear-probe AUROC (per-condition early stopping) — {feat_dir.name}")
    print(f"  configs            : {args.configs}")
    print(f"  probe              : up to {args.probe_max_epochs} epochs, "
          f"es_patience={args.es_patience}, es_tol={args.es_tol}, "
          f"{args.probe_seeds} seeds, lr={args.probe_lr}, weight_decay={args.weight_decay}")
    print(f"  min_test_per_class : {args.min_test_per_class}")
    print(f"  output             : {results_path}")
    print("=" * 70)

    if results_path.exists():
        with open(results_path) as f:
            results = json.load(f)
    else:
        results = {
            "feat_dir":           str(feat_dir),
            "probe_max_epochs":   args.probe_max_epochs,
            "probe_seeds":        args.probe_seeds,
            "weight_decay":       args.weight_decay,
            "es_patience":        args.es_patience,
            "es_tol":             args.es_tol,
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

        print(f"[{cfg_name}] fitting per-condition probes "
              f"({args.probe_seeds} seeds, early-stopped independently per condition)...",
              flush=True)
        mean_auc, per_cond, std_auc, per_cond_std, seed_aucs, per_cond_val = run_auroc_probe(
            train_feats, train_labels,
            val_feats,   val_labels,
            test_feats,  test_labels,
            label_names, device,
            max_epochs=args.probe_max_epochs, lr=args.probe_lr,
            batch_size=256, n_seeds=args.probe_seeds,
            weight_decay=args.weight_decay,
            es_patience=args.es_patience, es_tol=args.es_tol,
            min_test_per_class=args.min_test_per_class)

        results["results"][cfg_name] = {
            "mean_auroc":               round(mean_auc, 4),
            "std_auroc":                round(std_auc, 4),
            "per_condition":            {k: round(v, 4) for k, v in per_cond.items()},
            "per_condition_std":        {k: round(v, 4) for k, v in per_cond_std.items()},
            "per_condition_val_auroc":  {k: round(v, 4) for k, v in per_cond_val.items()},
            "seed_aucs":                [round(v, 4) for v in seed_aucs],
        }
        print(f"  mean AUROC = {mean_auc:.4f} ± {std_auc:.4f}  ({len(per_cond)} conditions)")
        for cond, auc in sorted(per_cond.items(), key=lambda x: -x[1]):
            print(f"    {cond:<38} {auc:.4f} ± {per_cond_std.get(cond, 0):.4f}")

        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

    print(f"\nResults → {results_path}")


if __name__ == "__main__":
    main()
