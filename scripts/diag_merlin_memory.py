"""
Peak-VRAM diagnostic for Merlin QIT training.

Tests each batch size for:
  - Training step  : student forward + backward (teacher features pre-cached, as in training loop)
  - Validation step: teacher forward + student forward on same batch (no grad)

Run via:
  sbatch --gres=gpu:l40s:1 scripts/run_slurm_diag_merlin.sbatch
"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

_ROOT = Path(__file__).parent.parent
_QFT  = _ROOT.parent / "quantized_ft"
sys.path.insert(0, str(_ROOT))
sys.path.insert(1, str(_QFT))

from models.eval_utils import MerlinEncoder
from models.quantization import quantized_forward
from downstream.dataset import MerlinDataset

_CT_DATA   = _ROOT.parent / "CT-CLIP" / "data"
DATA_DIR   = str(_CT_DATA / "merlin_data")
REPORTS    = str(_CT_DATA / "reports_final.xlsx")
LABELS     = str(_CT_DATA / "zero_shot_findings_disease_cls.csv")
METADATA   = str(_CT_DATA / "metadata.csv")
CONDITIONS = ["atelectasis", "pleural_effusion", "renal_cyst"]

BATCH_SIZES = [1, 2, 3, 4, 6, 8, 12, 16, 20]
W_BITS, A_BITS = 4, 8  # mid-range QIT config


def _qit_loss(s, t):
    s = F.normalize(s.float(), dim=-1)
    t = F.normalize(t.float(), dim=-1)
    return (1.0 - (s * t).sum(-1)).mean()


def _peak_mb(device):
    return torch.cuda.max_memory_allocated(device) / 1024 ** 2


def _run(fn, device):
    """Run fn(), return (peak_mb, error_str). Resets peak stats before running."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    try:
        fn()
        torch.cuda.synchronize(device)
        return _peak_mb(device), None
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return None, "OOM"
    except Exception as e:
        torch.cuda.empty_cache()
        return None, str(e)[:50]


def main():
    device    = torch.device("cuda")
    total_mb  = torch.cuda.get_device_properties(device).total_memory / 1024 ** 2
    print(f"GPU : {torch.cuda.get_device_name(device)}  ({total_mb:.0f} MB total)\n")

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_shape", type=int, nargs=3, default=None,
                        metavar=("D", "H", "W"),
                        help="Resize volumes to (D H W). Use '160 224 224' for Merlin training resolution.")
    cfg = parser.parse_args()

    # Get real sample shape (fast — indexing only, no full load)
    print("Getting input shape from dataset...")
    meta = METADATA if Path(METADATA).exists() else None
    ds   = MerlinDataset(data_folder=DATA_DIR, reports_file=REPORTS, labels_file=LABELS,
                         meta_file=meta, label_cols=CONDITIONS, split="train")
    x0, _ = ds[0]
    native_shape = tuple(x0.shape)   # (C, D, H, W)
    del ds, x0

    if cfg.target_shape is not None:
        shape = (native_shape[0], *cfg.target_shape)   # (1, D, H, W)
        print(f"  Native shape: {native_shape}  →  resized to: {shape}\n")
    else:
        shape = native_shape
        print(f"  Input shape (per sample): {shape}  → batched: (B, {', '.join(map(str, shape))})\n")

    # Load models once
    print("Loading teacher + student models...")
    teacher = MerlinEncoder().to(device)
    for p in teacher.parameters():
        p.requires_grad_(False)
    teacher.eval()
    student = MerlinEncoder().to(device)

    model_mb = torch.cuda.memory_allocated(device) / 1024 ** 2
    print(f"  Model weights: {model_mb:.0f} MB  ({model_mb / total_mb * 100:.1f}% of GPU)\n")

    print(f"{'BS':>4}  {'Train bwd (MB)':>16}  {'%':>6}  "
          f"{'Val fwd (MB)':>14}  {'%':>6}  {'Verdict':>10}")
    print("─" * 66)

    for bs in BATCH_SIZES:
        # Pre-build batch on CPU; move to GPU inside the measured function
        x_cpu = torch.randn(bs, *shape)
        t_cache = torch.randn(bs, MerlinEncoder.FEAT_DIM)  # synthetic cached teacher feats

        # ── Training: student forward + backward (teacher feats from cache) ──
        def train_step():
            x = x_cpu.to(device)
            t_f = t_cache.to(device)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                with quantized_forward([student.image_encoder], W_BITS, A_BITS, "per_channel"):
                    s_f = student(x)
            _qit_loss(s_f, t_f).backward()
            # No optimizer step needed — we only care about peak during backward

        # ── Validation: teacher forward then student forward, both no-grad ──
        def val_step():
            x = x_cpu.to(device)
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
                _t = teacher(x)
                with quantized_forward([student.image_encoder], 8, 8, "per_channel"):
                    _s = student(x)

        tr_mb,  tr_err  = _run(train_step, device)
        val_mb, val_err = _run(val_step,   device)

        def _fmt(mb, err):
            if err:
                return f"{'OOM':>14}", f"{'—':>6}"
            pct = mb / total_mb * 100
            return f"{mb:>14.0f}", f"{pct:>5.1f}%"

        tr_s,  tr_p  = _fmt(tr_mb,  tr_err)
        val_s, val_p = _fmt(val_mb, val_err)

        if tr_mb and val_mb:
            worst = max(tr_mb, val_mb)
            verdict = "OK" if worst < total_mb * 0.88 else ("tight" if worst < total_mb * 0.95 else "risky")
        else:
            verdict = "FAIL"

        print(f"{bs:>4}  {tr_s}  {tr_p}  {val_s}  {val_p}  {verdict:>10}")

        if tr_err and val_err:
            print("       Both OOM — stopping early.")
            break

    print("\nKey:")
    print("  OK    = >12% headroom (safe)")
    print("  tight = 5–12% headroom (may OOM under fragmentation)")
    print("  risky = <5% headroom (likely to OOM)")
    print("  FAIL  = OOM during test")


if __name__ == "__main__":
    main()
