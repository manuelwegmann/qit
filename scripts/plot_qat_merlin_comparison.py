"""
QAT (Merlin) convergence comparison plot.

Accepts one or more results.json files from run_qat_merlin.py and produces a
two-panel figure:

  Left  : train (solid) + val (dashed) BCE loss over epochs
  Right : train (solid) + val (dashed) AUROC over epochs
          stars = test AUROC at best-val epoch (quantized + FP)
          dotted gray line = chance (AUROC=0.5)

Usage:
    python scripts/plot_qat_merlin_comparison.py \\
        runs/qat_merlin_renal_cyst_qit_w4a4/results.json \\
        runs/qat_merlin_renal_cyst_pretrained_w4a4/results.json \\
        --labels "QIT init" "Pretrained init" \\
        --out plots/qat_merlin_renal_cyst_w4a4.png
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines


_COLORS = ["#2563EB", "#DC2626", "#16A34A", "#9333EA", "#EA580C"]


def _init_label(init_str: str) -> str:
    if init_str == "pretrained":
        return "Pretrained init"
    p = Path(init_str)
    return p.parent.name.replace("_", " ")


def _meta_title(runs: list) -> tuple[str, str]:
    d = runs[0]
    wb, ab = d.get("w_bits", "?"), d.get("a_bits", "?")
    quant  = f"W{wb}A{ab}"
    cond   = d.get("condition", "")
    n_epochs = max(len(r.get("epoch_log", [])) for r in runs)
    lr       = d.get("lr", "")
    lr_sched = d.get("lr_schedule", "")

    patience = d.get("patience", None)
    es_str = "full training" if not patience else f"early stopping: patience={patience}"

    n_train, n_val, n_test = d.get("n_train"), d.get("n_val"), d.get("n_test")
    size_str = ""
    if all(v is not None for v in [n_train, n_val, n_test]):
        size_str = f"train {n_train:,} · val {n_val:,} · test {n_test:,}"

    title = f"QAT convergence — Merlin / {cond} / {quant}"
    line1 = f"{n_epochs} epochs · lr={lr} · {lr_sched} LR · {es_str} · ★ = test @ best-val epoch"
    line2 = size_str
    subtitle = f"{line1}\n{line2}" if line2 else line1
    return title, subtitle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="+",
                        help="One or more results.json files from run_qat_merlin.py.")
    parser.add_argument("--labels", nargs="*", default=None,
                        help="Legend labels, one per results file. "
                             "Defaults to auto-generated from the init field.")
    parser.add_argument("--out", default=None,
                        help="Output path. Defaults to <first result dir>/qat_comparison.png.")
    args = parser.parse_args()

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

    fig, (ax_loss, ax_auroc) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01)
    ax_loss.set_title(subtitle, fontsize=9, color="#6B7280", pad=6)

    for d, label, color in zip(runs, labels, _COLORS):
        log    = d["epoch_log"]
        epochs = [e["epoch"]       for e in log]
        tr_l   = [e["train_loss"]  for e in log]
        va_l   = [e["val_loss"]    for e in log]
        tr_a   = [e["train_auroc"] for e in log]
        va_a   = [e["val_auroc"]   for e in log]

        ax_loss.plot(epochs, tr_l, color=color, linewidth=1.8, label=f"{label} — train")
        ax_loss.plot(epochs, va_l, color=color, linewidth=1.8, linestyle="--",
                      label=f"{label} — val")

        ax_auroc.plot(epochs, tr_a, color=color, linewidth=1.8, label=f"{label} — train")
        ax_auroc.plot(epochs, va_a, color=color, linewidth=1.8, linestyle="--",
                       label=f"{label} — val")

        best_ep = d.get("best_val_epoch") or max(log, key=lambda e: e["val_auroc"])["epoch"]
        test_auroc = d.get("test_auroc", {})
        if "quantized" in test_auroc:
            ax_auroc.plot(best_ep, test_auroc["quantized"], marker="*", markersize=13,
                          color=color, zorder=5, linestyle="none",
                          label=f"{label} — test W{d.get('w_bits')}A{d.get('a_bits')}: "
                                f"{test_auroc['quantized']:.4f} (ep. {best_ep})")
        if "fp" in test_auroc:
            ax_auroc.plot(best_ep, test_auroc["fp"], marker="D", markersize=8,
                          color=color, zorder=5, linestyle="none", alpha=0.7,
                          label=f"{label} — test FP: {test_auroc['fp']:.4f} (ep. {best_ep})")

    ax_auroc.axhline(0.5, color="gray", linestyle=":", linewidth=1.2, alpha=0.7,
                      label="chance (AUROC=0.5)")

    ax_loss.set_xlabel("Epoch", fontsize=11)
    ax_loss.set_ylabel("BCE loss", fontsize=11)
    ax_loss.set_xlim(left=1)
    ax_loss.set_ylim(bottom=0)
    ax_loss.grid(alpha=0.3)
    ax_loss.set_axisbelow(True)

    ax_auroc.set_xlabel("Epoch", fontsize=11)
    ax_auroc.set_ylabel("AUROC", fontsize=11)
    ax_auroc.set_xlim(left=1)
    ax_auroc.grid(alpha=0.3)
    ax_auroc.set_axisbelow(True)

    solid_patch  = mlines.Line2D([], [], color="black", linewidth=1.5, label="train")
    dashed_patch = mlines.Line2D([], [], color="black", linewidth=1.5,
                                  linestyle="--", label="val")
    ax_loss.legend(handles=[solid_patch, dashed_patch], fontsize=9, loc="upper right")
    ax_auroc.legend(fontsize=8, loc="best")

    plt.tight_layout()

    if args.out:
        out_path = Path(args.out)
    else:
        first_dir = Path(runs[0]["_path"]).parent
        out_path = first_dir / "qat_comparison.png"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Figure saved → {out_path}")


if __name__ == "__main__":
    main()
