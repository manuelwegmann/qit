"""
PTQ calibration comparison plot.

For each results.json produced by run_ptq_comparison.py, plots linear probe
accuracy vs. quantisation config for both models across all three calibration
modes (dynamic / minmax / percentile).

Usage:
    python scripts/plot_ptq_comparison.py \\
        runs/ptq_comparison_stl10_vit_b_16/results.json \\
        --out plots/ptq_stl10_vit_b_16.png
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np


from _plotstyle import (
    CONFIG_LABELS as _CONFIG_LABELS, CONFIG_ORDER as _CONFIG_ORDER,
    DATASET_LABELS as _DATASET_LABELS, BACKBONE_LABELS as _BACKBONE_LABELS,
)

# Two models → two colour families
_COLORS = ["#2563EB", "#DC2626"]   # blue = model_a, red = model_b
# Three calibration modes → three line styles
_MODE_STYLES = {
    "dynamic": dict(linestyle="-",  linewidth=2.0, marker="o", markersize=5),
    "minmax":  dict(linestyle="--", linewidth=1.8, marker="s", markersize=4),
}


def _pct_style(label: str):
    return dict(linestyle=":", linewidth=1.8, marker="^", markersize=4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="+",
                        help="One or more results.json from run_ptq_comparison.py.")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    # Support multiple result files (one subplot each) or just one
    all_data = []
    for path in args.results:
        with open(path) as f:
            d = json.load(f)
        d["_path"] = path
        all_data.append(d)

    n = len(all_data)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 5.5), squeeze=False)

    for ax, d in zip(axes[0], all_data):
        ds  = _DATASET_LABELS.get(d.get("dataset", ""), d.get("dataset", ""))
        bb  = _BACKBONE_LABELS.get(d.get("backbone", ""), d.get("backbone", ""))
        pct = d.get("percentile", 99.9)
        pct_key = f"pct{pct:.4g}"

        ax.set_title(f"{ds} / {bb}", fontsize=12, fontweight="bold")

        configs = [c for c in _CONFIG_ORDER if c in next(iter(d["results"].values()))]
        x = np.arange(len(configs))

        legend_handles = []
        for (label, model_res), color in zip(d["results"].items(), _COLORS):
            for mode_key, style_base in [
                ("dynamic", _MODE_STYLES["dynamic"]),
                ("minmax",  _MODE_STYLES["minmax"]),
                (pct_key,   _pct_style(pct_key)),
            ]:
                y = [model_res.get(cfg, {}).get(mode_key, float("nan"))
                     for cfg in configs]
                mode_label = mode_key if mode_key != pct_key else f"pct{pct:.4g}"
                line, = ax.plot(x, y, color=color, **style_base,
                                label=f"{label} — {mode_label}")
                legend_handles.append(line)

        ax.set_xticks(x)
        ax.set_xticklabels([_CONFIG_LABELS.get(c, c) for c in configs], fontsize=10)
        ax.set_xlabel("Quantisation config", fontsize=11)
        ax.set_ylabel("Linear probe accuracy", fontsize=11)
        ax.set_ylim(0, 1.05)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2f}"))
        ax.grid(alpha=0.3)
        ax.set_axisbelow(True)

        # Subtitle with calibration info
        calib_info = (f"calib: {d.get('n_calib_batches', '?')} batches  "
                      f"percentile={pct}  "
                      f"weight_gran={d.get('weight_granularity', '?')}")
        ax.set_title(ax.get_title() + f"\n{calib_info}", fontsize=10,
                     fontweight="normal", color="#6B7280")
        ax.set_title(f"PTQ --- {ds} / {bb}", fontsize=12, fontweight="bold", pad=22)

        # Legend — split into model colours and line style meaning
        model_patches = [
            mlines.Line2D([], [], color=c, linewidth=2, label=lbl)
            for (lbl, _), c in zip(d["results"].items(), _COLORS)
        ]
        style_patches = [
            mlines.Line2D([], [], color="grey", linestyle="-",  linewidth=1.5,
                          marker="o", markersize=4, label="dynamic"),
            mlines.Line2D([], [], color="grey", linestyle="--", linewidth=1.5,
                          marker="s", markersize=4, label="minmax"),
            mlines.Line2D([], [], color="grey", linestyle=":",  linewidth=1.5,
                          marker="^", markersize=4, label=f"pct{pct:.4g}"),
        ]
        leg1 = ax.legend(handles=model_patches, fontsize=9,
                         loc="upper left", bbox_to_anchor=(1.02, 1.0),
                         frameon=False, title="Model")
        ax.add_artist(leg1)
        ax.legend(handles=style_patches, fontsize=9,
                  loc="upper left", bbox_to_anchor=(1.02, 0.55),
                  frameon=False, title="Calibration")

    plt.tight_layout()

    if args.out:
        out_path = Path(args.out)
    else:
        first_dir = Path(all_data[0]["_path"]).parent
        out_path = first_dir / "ptq_comparison.png"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Figure saved → {out_path}")


if __name__ == "__main__":
    main()
