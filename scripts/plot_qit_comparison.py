"""
QIT comparison plot: teacher vs student accuracy across quantization regimes.

Produces a grouped bar chart showing how much each quantization configuration
degrades accuracy for the teacher (baseline) vs the QIT-trained student, with
the improvement delta annotated above each pair.

Usage:
    python scripts/plot_qit_comparison.py --results runs/qit_stl10_v2/results.json
    python scripts/plot_qit_comparison.py --results runs/qit_stl10_v2/results.json --out plots/qit_comparison.png
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


# Display names and ordering (least → most aggressive quantization)
_CONFIG_LABELS = {
    "fp":   "FP",
    "w8a8": "W8A8",
    "w6a6": "W6A6",
    "w4a6": "W4A6",
    "w4a4": "W4A4",
    "w2a4": "W2A4",
}
_CONFIG_ORDER = list(_CONFIG_LABELS.keys())

_TEACHER_COLOR = "#6B7280"   # neutral grey
_STUDENT_COLOR = "#2563EB"   # blue
_DELTA_POS     = "#16A34A"   # green
_DELTA_NEG     = "#DC2626"   # red


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True,
                        help="Path to results.json from a run_qit_cifar.py run.")
    parser.add_argument("--out", default=None,
                        help="Output path (default: same dir as results.json).")
    parser.add_argument("--metric", default="acc",
                        choices=["acc", "auroc"],
                        help="Metric key suffix in results.json.")
    args = parser.parse_args()

    results_path = Path(args.results)
    with open(results_path) as f:
        data = json.load(f)

    probe = data["results"]["probe"]
    teacher_key = f"teacher_{args.metric}"
    student_key = f"student_{args.metric}"

    # Collect in display order, skip missing configs
    configs, teacher_vals, student_vals = [], [], []
    for key in _CONFIG_ORDER:
        if key not in probe:
            continue
        t = probe[key].get(teacher_key)
        s = probe[key].get(student_key)
        if t is None or s is None:
            continue
        configs.append(key)
        teacher_vals.append(t)
        student_vals.append(s)

    n      = len(configs)
    x      = np.arange(n)
    width  = 0.35
    labels = [_CONFIG_LABELS.get(c, c) for c in configs]

    fig, ax = plt.subplots(figsize=(max(8, n * 1.4), 5))

    bars_t = ax.bar(x - width / 2, teacher_vals, width,
                    color=_TEACHER_COLOR, label="Teacher (PTQ baseline)",
                    zorder=3)
    bars_s = ax.bar(x + width / 2, student_vals, width,
                    color=_STUDENT_COLOR, label="Student (QIT)",
                    zorder=3)

    # Annotate deltas above each pair
    top = max(max(teacher_vals), max(student_vals))
    for i, (t, s) in enumerate(zip(teacher_vals, student_vals)):
        delta  = s - t
        colour = _DELTA_POS if delta >= 0 else _DELTA_NEG
        sign   = "+" if delta >= 0 else ""
        bar_top = max(t, s) + 0.008 * top
        ax.text(x[i], bar_top, f"{sign}{delta*100:.1f}pp",
                ha="center", va="bottom", fontsize=9,
                color=colour, fontweight="bold")

    # Formatting
    y_metric = "Accuracy" if args.metric == "acc" else "AUROC"
    ax.set_ylabel(y_metric, fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_xlabel("Quantization configuration  (→ more aggressive)", fontsize=11)
    ax.set_ylim(0, min(1.0, top * 1.18))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2f}"))
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.set_axisbelow(True)

    # Legend
    teacher_patch = mpatches.Patch(color=_TEACHER_COLOR, label="Teacher (PTQ baseline)")
    student_patch = mpatches.Patch(color=_STUDENT_COLOR, label="Student (QIT-trained)")
    ax.legend(handles=[teacher_patch, student_patch], fontsize=10,
              loc="lower left")

    # Dataset / model info if available in results
    dataset = data.get("dataset", "")
    title_parts = ["QIT: teacher vs student across quantization regimes"]
    if dataset:
        title_parts.append(f"({dataset.upper()} / ResNet18)")
    ax.set_title("  ".join(title_parts), fontsize=12, pad=12)

    plt.tight_layout()

    out_path = (Path(args.out) if args.out
                else results_path.parent / "qit_comparison.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Figure saved → {out_path}")


if __name__ == "__main__":
    main()
