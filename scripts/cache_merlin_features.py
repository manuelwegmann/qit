"""
Extract and cache Merlin encoder features for all quantisation configs.

Run once per checkpoint; downstream probe and KNN scripts read directly
from the feature directory with no model inference.

Output layout:
    {output_dir}/
        label_names.json           list of condition names (30 conditions)
        {split}_labels.npy         (train/val/test) — saved before any GPU work
        {cfg}_{split}_feats.npy    one file per config × split

Alignment guarantee
-------------------
Labels are saved from ds._ds.samples (the dataset index) before any GPU
work. DataLoader(shuffle=False) preserves that exact ordering, so
feats[i] and labels[i] always correspond to the same scan. A shape
assertion is printed for every (config, split) to confirm.

Usage:
    # Teacher — pretrained weights, no fine-tuning:
    python scripts/cache_merlin_features.py \\
        --model      teacher \\
        --output_dir runs/teacher_features

    # Student — QIT checkpoint:
    python scripts/cache_merlin_features.py \\
        --model       student \\
        --student_ckpt runs/qit_merlin_fulldata/checkpoint_final.pt \\
        --output_dir   runs/qit_merlin_fulldata/features

    sbatch scripts/run_slurm_cache_teacher_features.sbatch
    sbatch scripts/run_slurm_cache_student_features.sbatch
"""

import argparse
import json
import sys
import warnings
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

_ROOT = Path(__file__).parent.parent
_QFT  = _ROOT.parent / "quantized_ft"
sys.path.insert(0, str(_ROOT))
sys.path.insert(1, str(_QFT))

from models.quantization import calibrate_activations, static_quantized_forward
from models.eval_utils import _pbar, MerlinEncoder
from downstream.dataset import MerlinDataset

warnings.filterwarnings("ignore", category=UserWarning)

_CT_DATA          = _ROOT.parent / "CT-CLIP" / "data"
_DEFAULT_DATA_DIR = str(_CT_DATA / "merlin_data")
_DEFAULT_REPORTS  = str(_CT_DATA / "reports_final.xlsx")
_DEFAULT_LABELS   = str(_CT_DATA / "zero_shot_findings_disease_cls.csv")
_DEFAULT_METADATA = str(_CT_DATA / "metadata.csv")

_QUANT_CONFIGS = {"w8a8": (8, 8), "w4a8": (4, 8), "w4a4": (4, 4)}
_CONFIG_ORDER  = ["fp", "w8a8", "w4a8", "w4a4"]
_SPLITS        = ["train", "val", "test"]


class _ResizedDataset(torch.utils.data.Dataset):
    def __init__(self, ds, target_shape):
        self._ds     = ds
        self._target = tuple(target_shape)

    def __len__(self):
        return len(self._ds)

    def __getitem__(self, i):
        x, y = self._ds[i]
        if tuple(x.shape[1:]) != self._target:
            x = F.interpolate(x.unsqueeze(0), size=self._target,
                              mode="trilinear", align_corners=False).squeeze(0)
        return x, y


@torch.no_grad()
def _extract(model, loader, device, use_amp, amp_dtype,
             w_bits, a_bits, weight_granularity, calib_stats):
    model.eval()
    all_f = []
    for x, _ in _pbar(loader, desc="    extracting", leave=False):
        x = x.to(device)
        ctx = torch.amp.autocast("cuda", dtype=amp_dtype) if use_amp else nullcontext()
        with ctx:
            if calib_stats is not None:
                with static_quantized_forward([model.image_encoder], w_bits, a_bits,
                                              calib_stats, weight_granularity):
                    f = model(x)
            else:
                f = model(x)
        all_f.append(f.cpu().float())
    return torch.cat(all_f).numpy()


def main():
    parser = argparse.ArgumentParser(
        description="Extract and cache Merlin features for all quant configs × splits.")
    parser.add_argument("--model", choices=["student", "teacher"], required=True)
    parser.add_argument("--student_ckpt", default=None,
                        help="Path to QIT checkpoint (.pt).  Required if --model student.")
    parser.add_argument("--output_dir", required=True,
                        help="Where to write feature .npy files.")
    parser.add_argument("--data_dir",  default=_DEFAULT_DATA_DIR)
    parser.add_argument("--reports",   default=_DEFAULT_REPORTS)
    parser.add_argument("--labels",    default=_DEFAULT_LABELS)
    parser.add_argument("--metadata",  default=_DEFAULT_METADATA)
    parser.add_argument("--configs",   nargs="+", choices=_CONFIG_ORDER, default=_CONFIG_ORDER)
    parser.add_argument("--splits",    nargs="+", choices=_SPLITS,       default=_SPLITS)
    parser.add_argument("--percentile",         type=float, default=99.99)
    parser.add_argument("--n_calib_batches",    type=int,   default=32)
    parser.add_argument("--calib_batch_size",   type=int,   default=16)
    parser.add_argument("--weight_granularity", default="per_channel",
                        choices=["per_tensor", "per_channel"])
    parser.add_argument("--target_shape", type=int, nargs=3, default=[160, 224, 224],
                        metavar=("D", "H", "W"))
    parser.add_argument("--batch_size", type=int,  default=16)
    parser.add_argument("--no_amp",     action="store_true")
    parser.add_argument("--amp_dtype",  default="bf16", choices=["fp16", "bf16"])
    parser.add_argument("--overwrite",  action="store_true",
                        help="Re-extract even if .npy files already exist.")
    args = parser.parse_args()

    if args.model == "student" and not args.student_ckpt:
        parser.error("--student_ckpt is required when --model student")

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp   = not args.no_amp and device.type == "cuda"
    amp_dtype = torch.float16 if args.amp_dtype == "fp16" else torch.bfloat16

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    pct_label = f"pct{args.percentile:.4g}"

    print("=" * 70)
    print(f"Merlin feature extraction — {args.model}")
    if args.model == "student":
        print(f"  checkpoint : {args.student_ckpt}")
    else:
        print(f"  checkpoint : pretrained (no fine-tuning)")
    print(f"  configs    : {args.configs}")
    print(f"  splits     : {args.splits}")
    print(f"  calib      : {args.n_calib_batches}×bs={args.calib_batch_size}, "
          f"{pct_label}, gran={args.weight_granularity}")
    print(f"  output_dir : {out}")
    print("=" * 70)

    # ── datasets ───────────────────────────────────────────────────────────
    print("\nIndexing dataset...")
    meta_file = args.metadata if Path(args.metadata).exists() else None
    common = dict(data_folder=args.data_dir, reports_file=args.reports,
                  labels_file=args.labels, meta_file=meta_file, label_cols=None)
    target = tuple(args.target_shape)

    split_datasets = {}
    for split in args.splits:
        base_ds = MerlinDataset(**common, split=split)
        split_datasets[split] = _ResizedDataset(base_ds, target)
        print(f"  {split}: {len(base_ds)} samples")

    # ── save label names and labels (model-independent, no GPU needed) ─────
    print("\nSaving labels from dataset index (no GPU)...")
    # Pick any split to get label names — they're identical across splits.
    any_ds       = next(iter(split_datasets.values()))
    label_names  = list(any_ds._ds.label_names)
    names_file   = out / "label_names.json"
    with open(names_file, "w") as f:
        json.dump(label_names, f, indent=2)
    print(f"  label_names.json  ({len(label_names)} conditions)")

    all_labels = {}
    for split, ds in split_datasets.items():
        label_file = out / f"{split}_labels.npy"
        labels     = np.stack([s[1] for s in ds._ds.samples])
        np.save(label_file, labels)
        all_labels[split] = labels
        print(f"  {split}_labels.npy  shape={labels.shape}")

    # ── load model ─────────────────────────────────────────────────────────
    print(f"\nLoading {args.model} model...")
    model = MerlinEncoder().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    if args.model == "student":
        ckpt = torch.load(args.student_ckpt, map_location=device)
        model.load_state_dict(ckpt["student"])
        print(f"  loaded weights from {args.student_ckpt}")
    else:
        print(f"  using pretrained weights (no checkpoint)")
    model.eval()

    # ── calibrate once for all quantised configs ───────────────────────────
    calib_stats = None
    if any(c != "fp" for c in args.configs):
        print(f"\nCalibrating activations ({pct_label}, {args.n_calib_batches} batches)...")
        if "train" in split_datasets:
            calib_ds = split_datasets["train"]
        else:
            base = MerlinDataset(**common, split="train")
            calib_ds = _ResizedDataset(base, target)
        calib_loader = DataLoader(calib_ds, batch_size=args.calib_batch_size,
                                  shuffle=False, num_workers=0, pin_memory=False)
        calib_stats = calibrate_activations(
            model, calib_loader, args.n_calib_batches,
            device, percentile=args.percentile, use_amp=use_amp)
        print("  done.")

    # ── extract ────────────────────────────────────────────────────────────
    print()
    for split, ds in split_datasets.items():
        labels = all_labels[split]
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=0, pin_memory=False)
        for cfg_name in args.configs:
            feat_file = out / f"{cfg_name}_{split}_feats.npy"
            if feat_file.exists() and not args.overwrite:
                print(f"  [{cfg_name}][{split}] already exists — skipping "
                      f"(use --overwrite to re-extract)")
                continue

            w_bits = a_bits = c_stats = None
            if cfg_name != "fp":
                w_bits, a_bits = _QUANT_CONFIGS[cfg_name]
                c_stats = calib_stats

            print(f"  [{cfg_name}][{split}] extracting...", flush=True)
            feats = _extract(model, loader, device, use_amp, amp_dtype,
                             w_bits, a_bits, args.weight_granularity, c_stats)

            assert len(feats) == len(labels), (
                f"[{cfg_name}][{split}] Feature/label mismatch: "
                f"{len(feats)} feats vs {len(labels)} labels — aborting.")

            np.save(feat_file, feats)
            print(f"    saved {feats.shape} → {feat_file.name}  "
                  f"(labels aligned: {len(labels)} rows)", flush=True)

    print(f"\nDone.  All features written to {out}/")


if __name__ == "__main__":
    main()
