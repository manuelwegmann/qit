"""
Layer-wise CKA training dynamics plot.

Reads one or more results.json files produced by run_qit.py with --log_layer_cka
and plots per-layer CKA between student and teacher over training epochs.

Two modes:
  line plot (default) — one line per layer with training loss on a secondary axis
  heatmap (--heatmap) — layers × epochs colour matrix, one subplot per model

Usage:
    python scripts/plot_layer_cka.py runs/qit_stl10_vit_b_16_cka/results.json
    python scripts/plot_layer_cka.py run1/results.json run2/results.json --heatmap --out cka.png
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


# ── cosmetics ──────────────────────────────────────────────────────────────

from _plotstyle import (
    DATASET_LABELS as _DATASET_LABELS, BACKBONE_LABELS as _BACKBONE_LABELS,
)

# Palette: light-to-dark blue for shallow→deep layers
_LAYER_COLORS = ["#93C5FD", "#3B82F6", "#1D4ED8", "#1E3A5F"]
_LOSS_COLOR   = "#9CA3AF"   # light grey — background reference line


def _short_layer_name(full_name: str) -> str:
    """'encoder.layers.encoder_layer_11' → 'Layer 11'."""
    part = full_name.rsplit(".", 1)[-1]          # e.g. 'encoder_layer_11' or 'layer4'
    part = part.replace("encoder_layer_", "")    # '11'
    part = part.replace("layer",          "")    # '4' for resnet
    try:
        idx = int(part)
        return f"Layer {idx}"
    except ValueError:
        return full_name.split(".")[-1].replace("_", " ").title()


def _plot_one(ax_cka, ax_loss, data: dict) -> None:
    epochs_data = data["results"]["epochs"]
    # Filter to only epochs that have layer_cka logged
    epochs_data = [e for e in epochs_data if "layer_cka" in e]
    if not epochs_data:
        raise ValueError("No layer_cka entries found — was --log_layer_cka used?")

    epochs       = [e["epoch"]   for e in epochs_data]
    loss_vals    = [e["loss"]    for e in epochs_data]
    fp_sim_vals  = [e["fp_sim"]  for e in epochs_data]

    layer_names  = list(epochs_data[0]["layer_cka"].keys())
    cka_series   = {
        ln: [e["layer_cka"][ln] for e in epochs_data]
        for ln in layer_names
    }

    # ── loss on secondary axis (grey, behind) ────────────────────────────
    ax_loss.plot(epochs, loss_vals, color=_LOSS_COLOR, linewidth=1.2,
                 linestyle="--", alpha=0.6, zorder=1, label="Train loss")
    ax_loss.set_ylabel("Training loss", color=_LOSS_COLOR, fontsize=9)
    ax_loss.tick_params(axis="y", labelcolor=_LOSS_COLOR, labelsize=8)
    ax_loss.spines["right"].set_color(_LOSS_COLOR)
    ax_loss.set_ylim(bottom=0)

    # ── CKA per layer ────────────────────────────────────────────────────
    for (ln, vals), color in zip(cka_series.items(), _LAYER_COLORS):
        label = _short_layer_name(ln)
        ax_cka.plot(epochs, vals, color=color, linewidth=2.0,
                    marker="o", markersize=2.5, zorder=3, label=label)

    # fp_sim as a dashed reference (represents final-layer cosine sim)
    ax_cka.plot(epochs, fp_sim_vals, color="#6EE7B7", linewidth=1.5,
                linestyle=":", zorder=2, label="FP feature sim (cosine)")

    ax_cka.set_ylim(0.80, 1.01)
    ax_cka.yaxis.set_major_locator(mticker.MultipleLocator(0.05))
    ax_cka.yaxis.set_minor_locator(mticker.MultipleLocator(0.01))
    ax_cka.set_ylabel("CKA (student vs teacher)", fontsize=10)
    ax_cka.set_xlabel("Epoch", fontsize=10)
    ax_cka.grid(axis="y", alpha=0.25, zorder=0)
    ax_cka.grid(axis="x", alpha=0.12, zorder=0)
    ax_cka.set_axisbelow(True)

    # Annotate final CKA values on the right margin
    for (ln, vals), color in zip(cka_series.items(), _LAYER_COLORS):
        ax_cka.annotate(
            f"{vals[-1]:.3f}",
            xy=(epochs[-1], vals[-1]),
            xytext=(4, 0), textcoords="offset points",
            color=color, fontsize=8, va="center",
        )

    # Title
    bb  = _BACKBONE_LABELS.get(data.get("backbone", ""), data.get("backbone", ""))
    ds  = _DATASET_LABELS.get(data.get("dataset",   ""), data.get("dataset",   ""))
    bits = data.get("results", {}).get("epochs", [{}])[0]  # unused here but available
    ax_cka.set_title(
        f"Layer-wise CKA during QIT — {ds} / {bb}",
        fontsize=11, fontweight="bold", pad=10,
    )

    # Legend: CKA layers + loss, in one box
    handles_cka, labels_cka = ax_cka.get_legend_handles_labels()
    handles_loss, labels_loss = ax_loss.get_legend_handles_labels()
    ax_cka.legend(
        handles_cka + handles_loss,
        labels_cka  + labels_loss,
        fontsize=8, loc="lower right",
        framealpha=0.9, ncol=2,
    )


def _plot_heatmap(ax, data: dict) -> None:
    """Heatmap: layers (Y) × epochs (X), colour = CKA value."""
    import matplotlib.colors as mcolors

    epochs_data = [e for e in data["results"]["epochs"] if "layer_cka" in e]
    if not epochs_data:
        raise ValueError("No layer_cka entries found.")

    layer_names = list(epochs_data[0]["layer_cka"].keys())
    epochs      = [e["epoch"] for e in epochs_data]

    # Build matrix: rows = layers (shallow → deep), cols = epochs
    matrix = np.array([
        [e["layer_cka"][ln] for e in epochs_data]
        for ln in layer_names
    ])

    # Colour range: tight around the actual data so differences are visible
    vmin = max(0.0, matrix.min() - 0.02)
    vmax = 1.0
    cmap = plt.cm.RdYlGn   # red = low similarity, green = high

    im = ax.imshow(matrix, aspect="auto", cmap=cmap,
                   vmin=vmin, vmax=vmax, origin="upper",
                   interpolation="nearest")

    # Axes
    ax.set_yticks(range(len(layer_names)))
    ax.set_yticklabels([_short_layer_name(ln) for ln in layer_names], fontsize=9)

    # X-axis: show epoch ticks at reasonable intervals
    step = max(1, len(epochs) // 10)
    tick_pos = list(range(0, len(epochs), step))
    ax.set_xticks(tick_pos)
    ax.set_xticklabels([epochs[i] for i in tick_pos], fontsize=8)
    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylabel("Layer (shallow → deep)", fontsize=10)

    # Colourbar
    cbar = plt.colorbar(im, ax=ax, pad=0.02, fraction=0.03)
    cbar.set_label("CKA (student vs teacher)", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    # Annotate cells every ~5 epochs to keep it readable
    annot_step = max(1, len(epochs) // 10)
    for row_idx, ln in enumerate(layer_names):
        for col_idx in range(0, len(epochs), annot_step):
            val = matrix[row_idx, col_idx]
            text_color = "black" if val > (vmin + vmax) / 2 else "white"
            ax.text(col_idx, row_idx, f"{val:.2f}",
                    ha="center", va="center", fontsize=6.5,
                    color=text_color, fontweight="bold")

    bb = _BACKBONE_LABELS.get(data.get("backbone", ""), data.get("backbone", ""))
    ds = _DATASET_LABELS.get(data.get("dataset",   ""), data.get("dataset",   ""))
    ax.set_title(f"{ds} / {bb}", fontsize=11, fontweight="bold", pad=8)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="+",
                        help="One or more results.json from run_qit.py --log_layer_cka.")
    parser.add_argument("--heatmap", action="store_true",
                        help="Plot a layers × epochs heatmap instead of line curves.")
    parser.add_argument("--out", default=None,
                        help="Output path. Defaults to <results_dir>/layer_cka[_heatmap].png.")
    args = parser.parse_args()

    datasets = []
    for path in args.results:
        with open(path) as f:
            d = json.load(f)
        d["_path"] = path
        # backbone is saved at top level; dataset is not in the JSON so infer from path
        d.setdefault("backbone", "")
        p = str(path)
        d["dataset"] = "cifar10" if "cifar10" in p else ("stl10" if "stl10" in p else "")
        datasets.append(d)

    n = len(datasets)

    if args.heatmap:
        fig, axes = plt.subplots(1, n, figsize=(6 * n, 3.5), squeeze=False)
        for ax, d in zip(axes[0], datasets):
            _plot_heatmap(ax, d)
        default_stem = "layer_cka_heatmap.png"
    else:
        fig, axes = plt.subplots(1, n, figsize=(7 * n, 4.5), squeeze=False)
        for ax, d in zip(axes[0], datasets):
            ax_loss = ax.twinx()
            _plot_one(ax, ax_loss, d)
        default_stem = "layer_cka.png"

    plt.tight_layout()

    if args.out:
        out_path = Path(args.out)
    else:
        out_path = Path(datasets[0]["_path"]).parent / default_stem

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
