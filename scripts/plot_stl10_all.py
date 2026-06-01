"""
Produces a single PDF with one subplot per backbone (all 5 STL-10 QIT runs).
Usage:
    python scripts/plot_stl10_all.py
    python scripts/plot_stl10_all.py --out plots/stl10_all.pdf
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

_BACKBONES = [
    ("resnet18",           "ResNet-18"),
    ("efficientnet_b0",    "EfficientNet-B0"),
    ("mobilenet_v3_small", "MobileNet-V3 Small"),
    ("vit_b_16",           "ViT-B/16"),
    ("swin_t",             "Swin-T"),
]

_CONFIG_LABELS = {
    "fp":   "FP",
    "w8a8": "W8A8",
    "w6a6": "W6A6",
    "w4a6": "W4A6",
    "w4a4": "W4A4",
    "w2a4": "W2A4",
}
_CONFIG_ORDER = list(_CONFIG_LABELS.keys())

_TEACHER_COLOR = "#6B7280"
_STUDENT_COLOR = "#2563EB"
_DELTA_POS     = "#16A34A"
_DELTA_NEG     = "#DC2626"


def plot_backbone(ax, probe: dict, title: str):
    configs, teacher_vals, student_vals = [], [], []
    for key in _CONFIG_ORDER:
        if key not in probe:
            continue
        t = probe[key].get("teacher_acc")
        s = probe[key].get("student_acc")
        if t is None or s is None:
            continue
        configs.append(key)
        teacher_vals.append(t)
        student_vals.append(s)

    n     = len(configs)
    x     = np.arange(n)
    width = 0.35
    labels = [_CONFIG_LABELS.get(c, c) for c in configs]

    ax.bar(x - width / 2, teacher_vals, width, color=_TEACHER_COLOR,
           label="Teacher (PTQ baseline)", zorder=3)
    ax.bar(x + width / 2, student_vals, width, color=_STUDENT_COLOR,
           label="Student (QIT)", zorder=3)

    top = max(max(teacher_vals), max(student_vals))
    for i, (t, s) in enumerate(zip(teacher_vals, student_vals)):
        delta  = s - t
        colour = _DELTA_POS if delta >= 0 else _DELTA_NEG
        sign   = "+" if delta >= 0 else ""
        bar_top = max(t, s) + 0.008 * top
        ax.text(x[i], bar_top, f"{sign}{delta*100:.1f}pp",
                ha="center", va="bottom", fontsize=8,
                color=colour, fontweight="bold")

    ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
    ax.set_ylabel("Accuracy", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_xlabel("Quantization config  (→ more aggressive)", fontsize=9)
    ax.set_ylim(0, min(1.0, top * 1.20))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2f}"))
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.set_axisbelow(True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs_dir", default="runs",
                        help="Directory containing qit_stl10_<backbone> subdirs.")
    parser.add_argument("--out", default="plots/stl10_all_backbones.pdf")
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes_flat = axes.flatten()

    for i, (backbone_key, backbone_label) in enumerate(_BACKBONES):
        results_path = runs_dir / f"qit_stl10_{backbone_key}" / "results.json"
        with open(results_path) as f:
            data = json.load(f)
        plot_backbone(axes_flat[i], data["results"]["probe"],
                      f"STL-10 / {backbone_label}")

    # Hide the unused 6th subplot slot
    axes_flat[5].set_visible(False)

    # Shared legend below the plots
    teacher_patch = mpatches.Patch(color=_TEACHER_COLOR, label="Teacher (PTQ baseline)")
    student_patch = mpatches.Patch(color=_STUDENT_COLOR, label="Student (QIT-trained)")
    fig.legend(handles=[teacher_patch, student_patch], fontsize=11,
               loc="lower right", bbox_to_anchor=(0.97, 0.05))

    fig.suptitle("QIT: teacher vs student across quantization regimes — STL-10",
                 fontsize=14, fontweight="bold", y=1.01)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
