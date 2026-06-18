"""
Plot linear-probe AUROC results from run_probe_merlin.py: student (QIT) vs
teacher (pretrained) across quantization configs.

Produces three figures:
  1. Mean AUROC — grouped bars, teacher vs student, across fp/w8a8/w4a8/w4a4.
  2. Per-condition AUROC vs config — student, one line per eligible condition.
  3. Per-condition AUROC vs config — teacher, one line per eligible condition.

Methodology (quantization + probe settings) is pulled from the results.json
files where available and stated in the figure subtitles, since this is a
PTQ + linear-probe comparison, not an end-to-end fine-tune.

Usage:
    python scripts/plot_probe_merlin.py \\
        --student runs/qit_merlin_fulldata/features/probe_results/results.json \\
        --teacher runs/teacher_features/probe_results/results.json \\
        --out_dir plots/merlin
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

_CONFIG_LABELS = {"fp": "FP", "w8a8": "W8A8", "w4a8": "W4A8", "w4a4": "W4A4"}
_CONFIG_ORDER  = list(_CONFIG_LABELS.keys())

_TEACHER_COLOR = "#F4A261"
_STUDENT_COLOR = "#2A9D8F"
_ERR_COLOR     = "#264653"
_DELTA_POS     = "#06A77D"
_DELTA_NEG     = "#E63946"

# Quantization is static PTQ — percentile-clipped activation calibration —
# applied identically when caching both teacher and student features
# (see cache_merlin_features.py). Pulled from the cache job logs since the
# probe results.json doesn't carry calibration metadata.
_QUANT_METHOD = ("Static PTQ · percentile-clipped activation calibration "
                 "(pct99.99, 32 batches × bs16) · per-channel weight granularity")


def _probe_method_str(data):
    epochs = data.get("probe_epochs", "?")
    seeds  = data.get("probe_seeds", "?")
    wds    = data.get("weight_decays", [])
    wd_str = ", ".join(f"{w:g}" for w in wds) if wds else "?"
    min_pc = data.get("min_test_per_class", "?")
    return (f"Linear probe: {epochs} fixed epochs (no early stopping) · {seeds} seeds · "
            f"weight decay ∈ {{{wd_str}}} selected by val loss · "
            f"class-balanced via pos_weight · min(test_pos,test_neg)≥{min_pc}")


def plot_mean_auroc(student, teacher, out_path):
    configs = [c for c in _CONFIG_ORDER
              if c in student["results"] and c in teacher["results"]]
    labels  = [_CONFIG_LABELS[c] for c in configs]
    x, width = np.arange(len(configs)), 0.35

    t_vals  = [teacher["results"][c]["mean_auroc"] for c in configs]
    t_errs  = [teacher["results"][c]["std_auroc"]  for c in configs]
    t_seeds = [teacher["results"][c]["seed_aucs"]  for c in configs]
    s_vals  = [student["results"][c]["mean_auroc"] for c in configs]
    s_errs  = [student["results"][c]["std_auroc"]  for c in configs]
    s_seeds = [student["results"][c]["seed_aucs"]  for c in configs]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(max(8, len(configs) * 1.8), 5.5))
    err_kw = dict(ecolor=_ERR_COLOR, elinewidth=2, capsize=6, capthick=2)

    bars_t = ax.bar(x - width / 2, t_vals, width, yerr=t_errs, color=_TEACHER_COLOR,
                    edgecolor="white", linewidth=1.3, label="Teacher (pretrained)",
                    error_kw=err_kw, zorder=3)
    bars_s = ax.bar(x + width / 2, s_vals, width, yerr=s_errs, color=_STUDENT_COLOR,
                    edgecolor="white", linewidth=1.3, label="Student (QIT)",
                    error_kw=err_kw, zorder=3)

    for bars, vals in ((bars_t, t_vals), (bars_s, s_vals)):
        for rect, v in zip(bars, vals):
            ax.text(rect.get_x() + rect.get_width() / 2, 0.03, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=9, color="white",
                    fontweight="bold", rotation=90, zorder=4)

    rng = np.random.default_rng(0)
    dot_kw = dict(s=22, color="white", edgecolor=_ERR_COLOR, linewidth=0.8,
                 alpha=0.9, zorder=5)
    for i, (ts, ss) in enumerate(zip(t_seeds, s_seeds)):
        for center, sign, seeds in ((x[i] - width / 2, -1, ts), (x[i] + width / 2, +1, ss)):
            offsets = sign * 0.10 + rng.uniform(-0.04, 0.04, size=len(seeds))
            ax.scatter(center + offsets, seeds, **dot_kw)

    top = max(max(t_vals), max(s_vals))
    for i, (t, s, te, se) in enumerate(zip(t_vals, s_vals, t_errs, s_errs)):
        delta  = s - t
        colour = _DELTA_POS if delta >= 0 else _DELTA_NEG
        sign   = "+" if delta >= 0 else ""
        bar_top = max(t + te, s + se) + 0.015 * top
        ax.text(x[i], bar_top, f"{sign}{delta*100:.1f}pp",
                ha="center", va="bottom", fontsize=10, color=colour, fontweight="bold")

    ax.set_ylabel("Mean AUROC", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_xlabel("Quantization configuration  (→ more aggressive)", fontsize=11)
    ax.set_ylim(0, min(1.0, top * 1.2))
    ax.grid(axis="y", alpha=0.4, zorder=0)
    ax.grid(axis="x", visible=False)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    teacher_patch = mpatches.Patch(facecolor=_TEACHER_COLOR, edgecolor="white", label="Teacher (pretrained)")
    student_patch = mpatches.Patch(facecolor=_STUDENT_COLOR, edgecolor="white", label="Student (QIT)")
    seed_dot = plt.Line2D([0], [0], marker="o", color="none", label="Individual probe seeds",
                          markerfacecolor="white", markeredgecolor=_ERR_COLOR,
                          markeredgewidth=0.8, markersize=6)
    ax.legend(handles=[teacher_patch, student_patch, seed_dot], fontsize=10,
              loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)

    n_cond = len(student["results"][configs[0]]["per_condition"])
    title = f"Merlin linear-probe AUROC — pretrained vs QIT across quantization\n(mean over {n_cond} eligible conditions, error bars: ±1 std over 5 probe seeds)"
    ax.set_title(title, fontsize=13, fontweight="bold", pad=14)
    fig.text(0.01, -0.02, f"{_QUANT_METHOD}\n{_probe_method_str(student)}",
             fontsize=8, color="#6B7280", ha="left", va="top")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Figure saved → {out_path}")
    plt.close(fig)


def plot_per_condition(data, model_name, color, out_path):
    configs = [c for c in _CONFIG_ORDER if c in data["results"]]
    labels  = [_CONFIG_LABELS[c] for c in configs]
    conditions = sorted(data["results"][configs[0]]["per_condition"].keys())

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(9, 6))

    cmap = plt.get_cmap("tab20")
    x = np.arange(len(configs))
    for i, cond in enumerate(conditions):
        vals = [data["results"][c]["per_condition"][cond] for c in configs]
        errs = [data["results"][c]["per_condition_std"].get(cond, 0.0) for c in configs]
        ax.errorbar(x, vals, yerr=errs, marker="o", markersize=5, linewidth=1.8,
                    capsize=3, color=cmap(i % 20), label=cond.replace("_", " "))

    ax.axhline(0.5, color="gray", linestyle=":", linewidth=1.2, alpha=0.7, label="chance (0.5)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_xlabel("Quantization configuration  (→ more aggressive)", fontsize=11)
    ax.set_ylabel("AUROC", fontsize=12)
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis="y", alpha=0.4)
    ax.grid(axis="x", visible=False)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False,
              title="Condition (error bars: ±1 std/5 seeds)", title_fontsize=8)

    title = f"Merlin linear-probe AUROC per condition — {model_name} — across quantization"
    ax.set_title(title, fontsize=13, fontweight="bold", pad=14)
    fig.text(0.01, -0.02, f"{_QUANT_METHOD}\n{_probe_method_str(data)}",
             fontsize=8, color="#6B7280", ha="left", va="top")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Figure saved → {out_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--student", required=True,
                        help="results.json from run_probe_merlin.py for the QIT student.")
    parser.add_argument("--teacher", required=True,
                        help="results.json from run_probe_merlin.py for the pretrained teacher.")
    parser.add_argument("--out_dir", default="plots/merlin")
    args = parser.parse_args()

    with open(args.student) as f:
        student = json.load(f)
    with open(args.teacher) as f:
        teacher = json.load(f)

    out_dir = Path(args.out_dir)
    plot_mean_auroc(student, teacher, out_dir / "probe_merlin_mean_auroc.png")
    plot_per_condition(student, "Student (QIT)", _STUDENT_COLOR,
                       out_dir / "probe_merlin_per_condition_student.png")
    plot_per_condition(teacher, "Teacher (pretrained)", _TEACHER_COLOR,
                       out_dir / "probe_merlin_per_condition_teacher.png")


if __name__ == "__main__":
    main()
