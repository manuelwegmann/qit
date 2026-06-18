"""
Merlin QIT AUROC plot: teacher vs student across quantization regimes,
with error bars showing the across-seed std (5 probe seeds per config).

Usage:
    python scripts/plot_merlin_auroc.py --results runs/qit_merlin_unfrozen_bn_eval/results.json
    python scripts/plot_merlin_auroc.py --results runs/qit_merlin_unfrozen_bn_eval/results.json --out plots/merlin_auroc.png
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
    "w4a8": "W4A8",
    "w4a4": "W4A4",
}
_CONFIG_ORDER = list(_CONFIG_LABELS.keys())

_TEACHER_COLOR = "#F4A261"   # warm amber
_STUDENT_COLOR = "#2A9D8F"   # teal
_ERR_COLOR     = "#264653"   # dark slate
_DELTA_POS     = "#06A77D"   # green
_DELTA_NEG     = "#E63946"   # red


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True,
                        help="Path to results.json from a Merlin QIT run.")
    parser.add_argument("--out", default=None,
                        help="Output path (default: alongside results.json).")
    args = parser.parse_args()

    results_path = Path(args.results)
    with open(results_path) as f:
        data = json.load(f)

    if "auroc" in data:
        auroc = data["auroc"]
        title = "Merlin QIT: teacher vs student AUROC across quantization regimes"
    else:
        # PTQ comparison format: results[f"{model}_{cfg}"]["pct<percentile>"]
        pct = data.get("percentile", 99.9)
        pct_key = f"pct{pct:.4g}"
        auroc = {k: v[pct_key] for k, v in data["results"].items() if pct_key in v}
        title = (f"Merlin PTQ comparison: teacher vs student AUROC across quantization regimes\n"
                 f"(calibration: percentile-clipped, {pct_key})")

    configs = []
    teacher_vals, teacher_errs, teacher_seeds = [], [], []
    student_vals, student_errs, student_seeds = [], [], []
    for key in _CONFIG_ORDER:
        t = auroc.get(f"teacher_{key}")
        s = auroc.get(f"student_{key}")
        if t is None or s is None:
            continue
        configs.append(key)
        teacher_vals.append(t["mean_auroc"])
        teacher_errs.append(t.get("std_auroc", 0.0))
        teacher_seeds.append(t.get("seed_aucs", []))
        student_vals.append(s["mean_auroc"])
        student_errs.append(s.get("std_auroc", 0.0))
        student_seeds.append(s.get("seed_aucs", []))

    n      = len(configs)
    x      = np.arange(n)
    width  = 0.35
    labels = [_CONFIG_LABELS.get(c, c) for c in configs]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(max(8, n * 1.6), 5.5))

    err_kw = dict(ecolor=_ERR_COLOR, elinewidth=2, capsize=6, capthick=2)

    bars_t = ax.bar(x - width / 2, teacher_vals, width, yerr=teacher_errs,
                     color=_TEACHER_COLOR, edgecolor="white", linewidth=1.3,
                     label="Teacher (PTQ baseline)", error_kw=err_kw, zorder=3)
    bars_s = ax.bar(x + width / 2, student_vals, width, yerr=student_errs,
                     color=_STUDENT_COLOR, edgecolor="white", linewidth=1.3,
                     label="Student (QIT)", error_kw=err_kw, zorder=3)

    # Value labels inside each bar
    for bars, vals in ((bars_t, teacher_vals), (bars_s, student_vals)):
        for rect, v in zip(bars, vals):
            ax.text(rect.get_x() + rect.get_width() / 2, 0.03, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=9, color="white",
                    fontweight="bold", rotation=90, zorder=4)

    # Individual probe seeds as jittered dots, offset toward each bar's outer
    # edge so they don't sit on top of the error bar (centered on the bar).
    rng = np.random.default_rng(0)
    dot_kw = dict(s=22, color="white", edgecolor=_ERR_COLOR, linewidth=0.8,
                   alpha=0.9, zorder=5)
    dot_offset, dot_jitter = 0.10, 0.04
    for i, (t_seeds, s_seeds) in enumerate(zip(teacher_seeds, student_seeds)):
        for center, sign, seeds in ((x[i] - width / 2, -1, t_seeds),
                                     (x[i] + width / 2, +1, s_seeds)):
            if not seeds:
                continue
            offsets = sign * dot_offset + rng.uniform(-dot_jitter, dot_jitter, size=len(seeds))
            ax.scatter(center + offsets, seeds, **dot_kw)

    # Annotate deltas above each pair
    top = max(max(teacher_vals), max(student_vals))
    for i, (t, s, te, se) in enumerate(zip(teacher_vals, student_vals, teacher_errs, student_errs)):
        delta  = s - t
        colour = _DELTA_POS if delta >= 0 else _DELTA_NEG
        sign   = "+" if delta >= 0 else ""
        bar_top = max(t + te, s + se) + 0.015 * top
        ax.text(x[i], bar_top, f"{sign}{delta*100:.1f}pp",
                ha="center", va="bottom", fontsize=10,
                color=colour, fontweight="bold")

    # Formatting
    ax.set_ylabel("AUROC", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_xlabel("Quantization configuration  (→ more aggressive)", fontsize=11)
    ax.set_ylim(0, min(1.0, top * 1.2))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2f}"))
    ax.grid(axis="y", alpha=0.4, zorder=0)
    ax.grid(axis="x", visible=False)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    teacher_patch = mpatches.Patch(facecolor=_TEACHER_COLOR, edgecolor="white", label="Teacher (PTQ baseline)")
    student_patch = mpatches.Patch(facecolor=_STUDENT_COLOR, edgecolor="white", label="Student (QIT-trained)")
    seed_dot = plt.Line2D([0], [0], marker="o", color="none", label="Individual probe seeds",
                           markerfacecolor="white", markeredgecolor=_ERR_COLOR,
                           markeredgewidth=0.8, markersize=6)
    ax.legend(handles=[teacher_patch, student_patch, seed_dot], fontsize=10,
              loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)

    n_seeds = 5
    ax.set_title(
        f"{title}\n(error bars: ±1 std over {n_seeds} probe seeds)",
        fontsize=13, fontweight="bold", pad=14,
    )

    plt.tight_layout()

    if args.out:
        out_path = Path(args.out)
    else:
        out_path = results_path.parent / "merlin_auroc.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Figure saved → {out_path}")


if __name__ == "__main__":
    main()
