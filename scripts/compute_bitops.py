"""
Theoretical compute savings from quantization, reported as BitOps (bit-operations:
MACs x w_bits x a_bits) relative to an unquantized baseline.

quantized_forward() / static_quantized_forward() (models/quantization.py) apply a
single (w_bits, a_bits) pair uniformly to every nn.Linear/Conv2d/Conv3d layer in the
model. This script counts MACs for exactly those layers via forward hooks
(batch_size=1) and scales by (w_bits * a_bits) to get BitOps for a config; dividing
by the baseline BitOps (w_bits=a_bits=--baseline_bits) gives a hardware-agnostic
"% of baseline compute" figure. Because the scaling is uniform across layers, this
ratio reduces to (w_bits * a_bits) / baseline_bits**2 regardless of architecture --
the script also reports the absolute GMACs/GFLOPs per backbone so that ratio means
something concrete.

--baseline_bits defaults to 16, NOT 32: both run_qit.py and
run_qit_merlin.py run the "fp" (unquantized) pass under
torch.amp.autocast(dtype=torch.bfloat16) by default, so the "fp" entry in
results.json is itself a bf16 (16x16) computation, not true FP32. Pass
--baseline_bits 32 to compare against a true-FP32 reference instead.

Note this is a *theoretical* compute metric, not a measured speedup: real hardware
doesn't always scale linearly with bit-width (e.g. no native 4-bit kernels on most
GPUs).

Optionally pass --results to also report the accuracy/AUROC retained at each config
(from a run_qit*/run_ptq_comparison* results.json), e.g.:
    "W4A8 retains 98.6% of FP performance at 50% of FP16 BitOps"

Usage:
    python scripts/compute_bitops.py --backbone vit_b_16
    python scripts/compute_bitops.py --backbone merlin \\
        --results runs/qit_merlin_fulldata/results.json
    python scripts/compute_bitops.py --all
    python scripts/compute_bitops.py --backbone vit_b_16 --baseline_bits 32
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from models.eval_utils import build_backbone, QUANT_CONFIGS, MerlinEncoder

_CIFAR_BACKBONES = ["resnet18", "efficientnet_b0", "mobilenet_v3_small", "vit_b_16", "swin_t"]
_ALL_BACKBONES = _CIFAR_BACKBONES + ["merlin"]

# Default Merlin configs to report BitOps for (a representative subset).
_MERLIN_CONFIGS = [
    ("fp",   None, None),
    ("w8a8", 8,    8),
    ("w4a8", 4,    8),
    ("w4a4", 4,    4),
]


# ---------------------------------------------------------------------------
# MAC counting
# ---------------------------------------------------------------------------

def count_macs(model: nn.Module, input_tensor: torch.Tensor) -> dict:
    """MACs for each nn.Linear/Conv2d/Conv3d layer, for one forward pass.

    Linear:        MACs = out.numel() * in_features
    Conv2d/Conv3d: MACs = out.numel() * (in_channels // groups) * prod(kernel_size)
    """
    macs: dict = {}
    handles = []

    def _make_hook(name, module):
        def _hook(_module, _inp, out):
            if isinstance(module, nn.Linear):
                m = out.numel() * module.in_features
            else:
                k = 1
                for s in module.kernel_size:
                    k *= s
                m = out.numel() * (module.in_channels // module.groups) * k
            macs[name] = macs.get(name, 0) + m
        return _hook

    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv3d)):
            handles.append(module.register_forward_hook(_make_hook(name, module)))

    model.eval()
    with torch.no_grad():
        model(input_tensor)

    for h in handles:
        h.remove()

    return macs


def build_model_and_input(backbone: str, device: torch.device):
    if backbone == "merlin":
        model = MerlinEncoder().to(device)
        x = torch.zeros(1, 1, 160, 224, 224, device=device)
        configs = _MERLIN_CONFIGS
    else:
        model = build_backbone(backbone).to(device)
        x = torch.zeros(1, 3, 224, 224, device=device)
        configs = QUANT_CONFIGS
    return model, x, configs


# ---------------------------------------------------------------------------
# Optional: accuracy/AUROC retained per config, from a results.json
# ---------------------------------------------------------------------------

def load_performance(path: str, configs: list) -> tuple:
    """Returns ({config_name: metric_value}, metric_label)."""
    with open(path) as f:
        data = json.load(f)

    perf = {}
    if "auroc" in data:
        for name, _, _ in configs:
            entry = data["auroc"].get(f"student_{name}")
            if entry is not None:
                perf[name] = entry["mean_auroc"]
        return perf, "AUROC"

    if "results" in data and "probe" in data["results"]:
        probe = data["results"]["probe"]
        for name, _, _ in configs:
            entry = probe.get(name)
            if entry is not None and "student_acc" in entry:
                perf[name] = entry["student_acc"]
        return perf, "accuracy"

    if "results" in data:
        # run_probe_merlin.py format: results[cfg]["mean_auroc"], one model per file.
        probe = data["results"]
        for name, _, _ in configs:
            entry = probe.get(name)
            if entry is not None and "mean_auroc" in entry:
                perf[name] = entry["mean_auroc"]
        if perf:
            return perf, "AUROC"

    raise ValueError(f"Unrecognized results.json format: {path}")


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def report(backbone: str, macs_by_layer: dict, configs: list,
           baseline_bits: int = 16, perf: dict = None, metric_label: str = None,
           config_filter: list = None) -> dict:
    total_macs = sum(macs_by_layer.values())
    gmacs  = total_macs / 1e9
    gflops = 2 * gmacs  # one mult + one add per MAC

    print(f"\n{backbone}")
    print(f"  quantized layers (Linear/Conv2d/Conv3d): {len(macs_by_layer)}")
    print(f"  total: {gmacs:.3f} GMACs  ({gflops:.3f} GFLOPs)")
    print(f"  baseline (\"fp\"): {baseline_bits}-bit  ->  {gflops:.3f} GFLOPs is 100%")

    header = f"  {'config':<8} {'w':>3} {'a':>3} {'BitOps':>10} {'% of FP' + str(baseline_bits):>10}"
    if perf is not None:
        header += f"  {metric_label:>10} {'% of FP':>9}"
    print(header)

    fp_perf = perf.get("fp") if perf else None
    out_configs = {}
    for name, w, a in configs:
        if config_filter and name not in config_filter:
            continue
        wb = w if w is not None else baseline_bits
        ab = a if a is not None else baseline_bits
        ratio = (wb * ab) / (baseline_bits * baseline_bits)
        bitops_gflops = gflops * ratio
        line = f"  {name:<8} {wb:>3} {ab:>3} {bitops_gflops:>9.4f}G {ratio*100:>9.2f}%"
        entry = {"w_bits": wb, "a_bits": ab, "bitops_ratio": ratio,
                 "bitops_gflops": bitops_gflops}
        if perf is not None:
            val = perf.get(name)
            if val is not None and fp_perf:
                line += f"  {val:>10.4f} {val/fp_perf*100:>8.2f}%"
                entry["metric"] = val
                entry["pct_of_fp"] = val / fp_perf * 100
            else:
                line += f"  {'--':>10} {'--':>9}"
        print(line)
        out_configs[name] = entry

    return {"backbone": backbone, "gmacs": gmacs, "gflops": gflops, "configs": out_configs}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", choices=_ALL_BACKBONES)
    parser.add_argument("--all", action="store_true",
                        help="Run for every backbone in " + str(_ALL_BACKBONES))
    parser.add_argument("--configs", nargs="+", default=None,
                        help="Subset of config names to report (default: all).")
    parser.add_argument("--results", default=None,
                        help="Path to a results.json — adds an accuracy/AUROC "
                             "retained column (only valid with a single --backbone).")
    parser.add_argument("--baseline_bits", type=int, default=16,
                        help="Bit-width of the unquantized ('fp') reference. "
                             "Default 16, matching the bf16 autocast used by "
                             "default in run_qit.py / run_qit_merlin.py "
                             "for the 'fp' eval pass. Use 32 for a true-FP32 "
                             "reference.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", default=None, help="Optional path to dump JSON summary.")
    args = parser.parse_args()

    if not args.backbone and not args.all:
        parser.error("specify --backbone <name> or --all")
    if args.results and (args.all or not args.backbone):
        parser.error("--results requires a single --backbone")

    backbones = _ALL_BACKBONES if args.all else [args.backbone]
    device = torch.device(args.device)

    summary = {}
    for backbone in backbones:
        print(f"\nLoading {backbone}...", flush=True)
        model, x, configs = build_model_and_input(backbone, device)
        macs_by_layer = count_macs(model, x)

        perf, metric_label = (None, None)
        if args.results:
            perf, metric_label = load_performance(args.results, configs)

        summary[backbone] = report(backbone, macs_by_layer, configs,
                                    baseline_bits=args.baseline_bits,
                                    perf=perf, metric_label=metric_label,
                                    config_filter=args.configs)
        del model

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSummary saved → {out_path}")


if __name__ == "__main__":
    main()
