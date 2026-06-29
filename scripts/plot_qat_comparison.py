"""
QAT convergence comparison plot.

Accepts one or more results.json files from run_qat.py and produces a
two-panel figure per dataset/backbone/quantisation combination:

  Left  : train (solid) + val (dashed) loss over epochs
  Right : train (solid) + val (dashed) accuracy over epochs
          ★ = test accuracy at final epoch (best-val checkpoint)

Title, subtitle, and legend labels are generated automatically from the JSON
metadata (dataset, backbone, bit-widths, epochs).  Override labels with
--labels when comparing runs that differ beyond their init checkpoint.

Usage:
    python scripts/plot_qat_comparison.py \\
        runs/qat_stl10_vit_b_16_qit_w4a4/results.json \\
        runs/qat_stl10_vit_b_16_pretrained_w4a4/results.json \\
        --labels "QIT init" "Pretrained init" \\
        --out plots/qat_stl10_vit_b_16_w4a4.png
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np


# ---------------------------------------------------------------------------
# Display-name maps (extend as new backbones / datasets are added)
# ---------------------------------------------------------------------------

from _plotstyle import (
    DATASET_LABELS as _DATASET_LABELS, BACKBONE_LABELS as _BACKBONE_LABELS,
    QAT_COLORS as _COLORS,
)


# ---------------------------------------------------------------------------
# Label helpers
# ---------------------------------------------------------------------------

def _init_label(init_str: str) -> str:
    """Turn the init field into a short human-readable label."""
    if init_str == "pretrained":
        return "Pretrained init"
    p = Path(init_str)
    # e.g. runs/qit_stl10_vit_b_16/checkpoint_best.pt → "qit_stl10_vit_b_16"
    return p.parent.name.replace("_", " ")


def _meta_title(runs: list) -> tuple[str, str]:
    """Build (title, two-line subtitle) from the first run's metadata."""
    d = runs[0]
    dataset  = _DATASET_LABELS.get(d.get("dataset",  ""), d.get("dataset",  ""))
    backbone = _BACKBONE_LABELS.get(d.get("backbone", ""), d.get("backbone", ""))
    wb, ab   = d.get("w_bits", "?"), d.get("a_bits", "?")
    quant    = f"W{wb}A{ab}"
    n_epochs = max(len(d.get("epoch_log", [])) for d in runs)
    lr       = d.get("lr", "")
    lr_sched = d.get("lr_schedule", "")

    # Early stopping + test-acc clarification
    patience = d.get("patience", None)
    if patience is None or patience == 0:
        es_str = "full training · ★ = best val checkpoint"
    else:
        es_str = f"early stopping: patience={patience} · ★ = best val checkpoint"

    # Dataset sizes — fall back gracefully for existing JSONs without these fields
    n_train = d.get("n_train")
    n_val   = d.get("n_val")
    n_test  = d.get("n_test")
    if all(v is not None for v in [n_train, n_val, n_test]):
        size_str = f"train {n_train:,} · val {n_val:,} · test {n_test:,}"
    else:
        size_str = ""

    title    = f"QAT convergence — {dataset} / {backbone} / {quant}"
    line1    = f"{n_epochs} epochs · lr={lr} · {lr_sched} LR · {es_str}"
    line2    = size_str
    subtitle = f"{line1}\n{line2}" if line2 else line1
    return title, subtitle


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="+",
                        help="One or more results.json files from run_qat.py.")
    parser.add_argument("--labels", nargs="*", default=None,
                        help="Legend labels, one per results file. "
                             "Defaults to auto-generated from the init field.")
    parser.add_argument("--out", default=None,
                        help="Output path. Defaults to <first result dir>/qat_comparison.png.")
    args = parser.parse_args()

    # ── load ──────────────────────────────────────────────────────────────
    runs = []
    for path in args.results:
        with open(path) as f:
            d = json.load(f)
        d["_path"] = path
        runs.append(d)

    labels = args.labels or [_init_label(d.get("init", "")) for d in runs]
    if len(labels) < len(runs):
        labels += [_init_label(d.get("init", "")) for d in runs[len(labels):]]

    title, subtitle = _meta_title(runs)

    # ── figure ────────────────────────────────────────────────────────────
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01)
    ax_loss.set_title(subtitle, fontsize=9, color="#6B7280", pad=6)

    for d, label, color in zip(runs, labels, _COLORS):
        log    = d["epoch_log"]
        epochs = [e["epoch"]     for e in log]
        tr_l   = [e["train_loss"] for e in log]
        va_l   = [e["val_loss"]   for e in log]
        tr_a   = [e["train_acc"]  for e in log]
        va_a   = [e["val_acc"]    for e in log]

        # Loss panel
        ax_loss.plot(epochs, tr_l, color=color, linewidth=1.8, label=f"{label} — train")
        ax_loss.plot(epochs, va_l, color=color, linewidth=1.8, linestyle="--",
                     label=f"{label} — val")

        # Accuracy panel — curves
        ax_acc.plot(epochs, tr_a, color=color, linewidth=1.8, label=f"{label} — train")
        ax_acc.plot(epochs, va_a, color=color, linewidth=1.8, linestyle="--",
                    label=f"{label} — val")

        # Test accuracy star — placed at the best-val epoch
        test_acc = d.get("test_acc")
        if test_acc is not None:
            best_ep = d.get("best_val_epoch") or max(
                d["epoch_log"], key=lambda e: e["val_acc"]
            )["epoch"]
            ax_acc.plot(best_ep, test_acc, marker="*", markersize=13,
                        color=color, zorder=5, linestyle="none",
                        label=f"{label} — test: {test_acc:.4f} (ep. {best_ep})")

    # ── formatting ────────────────────────────────────────────────────────
    ax_loss.set_xlabel("Epoch", fontsize=11)
    ax_loss.set_ylabel("Cross-entropy loss", fontsize=11)
    ax_loss.set_xlim(left=1)
    ax_loss.set_ylim(bottom=0)
    ax_loss.grid(alpha=0.3)
    ax_loss.set_axisbelow(True)

    ax_acc.set_xlabel("Epoch", fontsize=11)
    ax_acc.set_ylabel("Accuracy", fontsize=11)
    ax_acc.set_xlim(left=1)
    ax_acc.set_ylim(0, 1.05)
    ax_acc.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2f}"))
    ax_acc.grid(alpha=0.3)
    ax_acc.set_axisbelow(True)

    # Shared legend: line style legend in loss panel, full legend in acc panel
    solid_patch  = mlines.Line2D([], [], color="black", linewidth=1.5,           label="train")
    dashed_patch = mlines.Line2D([], [], color="black", linewidth=1.5,
                                 linestyle="--",                                  label="val")
    star_patch   = mlines.Line2D([], [], color="black", marker="*", markersize=10,
                                 linestyle="none",                                label="test ★")
    ax_loss.legend(handles=[solid_patch, dashed_patch, star_patch],
                   fontsize=9, loc="upper right")
    ax_acc.legend(fontsize=9, loc="lower right")

    plt.tight_layout()

    # ── save ──────────────────────────────────────────────────────────────
    if args.out:
        out_path = Path(args.out)
    else:
        first_dir = Path(runs[0]["_path"]).parent
        stem = "qat_comparison"
        out_path = first_dir / f"{stem}.png"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Figure saved → {out_path}")


if __name__ == "__main__":
    main()
