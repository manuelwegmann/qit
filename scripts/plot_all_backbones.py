"""
Produces a single PDF with one subplot per backbone (all 5 QIT runs for a dataset).

Usage:
    python scripts/plot_all_backbones.py --dataset cifar10
    python scripts/plot_all_backbones.py --dataset stl10
    python scripts/plot_all_backbones.py --dataset stl10 --out plots/stl10_all.pdf
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

# Per-dataset settings. run_override points a backbone at a non-default run dir
# (default dir: qit_<dataset>_<backbone_key>).
_DATASETS = {
    "cifar10": {
        "label": "CIFAR-10",
        "run_override": {
            "resnet18":        "qit_cifar10_resnet18_freeze_bn",
            "efficientnet_b0": "qit_cifar10_efficientnet_b0_freeze_bn",
        },
    },
    "stl10": {
        "label": "STL-10",
        "run_override": {},
    },
}

from _plotstyle import (
    CONFIG_LABELS as _CONFIG_LABELS, CONFIG_ORDER as _CONFIG_ORDER,
    TEACHER_COLOR as _TEACHER_COLOR, STUDENT_COLOR as _STUDENT_COLOR,
    DELTA_POS as _DELTA_POS, DELTA_NEG as _DELTA_NEG,
)


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
    parser.add_argument("--dataset", required=True, choices=list(_DATASETS),
                        help="Which dataset's QIT runs to summarise.")
    parser.add_argument("--runs_dir", default="runs",
                        help="Directory containing qit_<dataset>_<backbone> subdirs.")
    parser.add_argument("--out", default=None,
                        help="Output PDF. Default: plots/<dataset>_all_backbones.pdf")
    args = parser.parse_args()

    cfg      = _DATASETS[args.dataset]
    label    = cfg["label"]
    override = cfg["run_override"]

    runs_dir = Path(args.runs_dir)
    out_path = Path(args.out or f"plots/{args.dataset}_all_backbones.pdf")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes_flat = axes.flatten()

    for i, (backbone_key, backbone_label) in enumerate(_BACKBONES):
        run_dir_name = override.get(backbone_key, f"qit_{args.dataset}_{backbone_key}")
        results_path = runs_dir / run_dir_name / "results.json"
        with open(results_path) as f:
            data = json.load(f)
        plot_backbone(axes_flat[i], data["results"]["probe"],
                      f"{label} / {backbone_label}")

    # Hide the unused 6th subplot slot
    axes_flat[5].set_visible(False)

    # Shared legend below the plots
    teacher_patch = mpatches.Patch(color=_TEACHER_COLOR, label="Teacher (PTQ baseline)")
    student_patch = mpatches.Patch(color=_STUDENT_COLOR, label="Student (QIT-trained)")
    fig.legend(handles=[teacher_patch, student_patch], fontsize=11,
               loc="lower right", bbox_to_anchor=(0.97, 0.05))

    fig.suptitle(f"QIT: teacher vs student across quantization regimes — {label}",
                 fontsize=14, fontweight="bold", y=1.01)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
