"""
PTQ calibration comparison for text (SST-2 / DistilBERT).

Mirrors run_ptq_comparison.py for HuggingFace encoders.

Three quantization modes per (model, config) pair:
  dynamic  — per-sample min-max (matches run_qit_text.py eval baseline)
  minmax   — dataset-level min-max calibration (full range)
  pct<N>   — percentile calibration, clips the top/bottom (100-N)/2 % of values

Results are saved in the format expected by plot_ptq_comparison.py.

Usage:
    python scripts/run_ptq_text.py \\
        --model distilbert-base-uncased \\
        --model_a runs/qit_sst2_distilbert-base-uncased/checkpoint_best.pt \\
        --model_a_label "QIT student" \\
        --model_b pretrained \\
        --model_b_label "Teacher" \\
        --n_calib_batches 32 --percentile 99.9
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModel, AutoTokenizer

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from models.quantization import (
    CalibrationStats,
    quantized_forward,
    static_quantized_forward,
)


QUANT_CONFIGS = [
    ("fp",   None, None),
    ("w8a8", 8,    8),
    ("w6a6", 6,    6),
    ("w4a6", 4,    6),
    ("w4a4", 4,    4),
    ("w2a4", 2,    4),
]


# ── dataset ───────────────────────────────────────────────────────────────────

class _TextDataset(torch.utils.data.Dataset):
    def __init__(self, input_ids, attention_mask, labels):
        self.input_ids      = input_ids
        self.attention_mask = attention_mask
        self.labels         = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attention_mask[idx], self.labels[idx]


def tokenize_split(hf_split, tokenizer, max_length, text_field="sentence"):
    enc = tokenizer(
        list(hf_split[text_field]),
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    labels = torch.tensor(list(hf_split["label"]), dtype=torch.long)
    return _TextDataset(enc["input_ids"], enc["attention_mask"], labels)


# ── model helpers ─────────────────────────────────────────────────────────────

def get_cls(model, input_ids, attention_mask, use_amp):
    """Run encoder and return [CLS] token as feature vector."""
    if use_amp:
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = model(input_ids=input_ids, attention_mask=attention_mask)
            return out.last_hidden_state[:, 0, :].float()
    out = model(input_ids=input_ids, attention_mask=attention_mask)
    return out.last_hidden_state[:, 0, :].float()


def load_model(model_name, checkpoint, device, hf_cache):
    """Load HuggingFace encoder from a .pt checkpoint or 'pretrained'."""
    m = AutoModel.from_pretrained(model_name, cache_dir=hf_cache).to(device)
    if checkpoint != "pretrained":
        ckpt = torch.load(checkpoint, map_location=device)
        m.load_state_dict(ckpt["student"])
        print(f"    Loaded checkpoint: {checkpoint}  (epoch {ckpt.get('epoch', '?')})")
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


# ── calibration ───────────────────────────────────────────────────────────────

def calibrate_text(model, loader, n_batches, device, percentile=100.0, use_amp=False):
    """calibrate_activations adapted for (input_ids, attention_mask, labels) batches."""
    _MAX_SAMPLES = 8192
    buffers: dict = {}
    hooks = []

    for lyr in model.modules():
        if not isinstance(lyr, (nn.Linear, nn.Conv2d, nn.Conv3d)):
            continue
        key = f"{type(lyr).__name__}_{id(lyr)}"
        buffers[key] = []

        def _make_hook(k):
            def _hook(mod, inp, out):
                x = inp[0].detach().float().flatten()
                if len(x) > _MAX_SAMPLES:
                    idx = torch.randperm(len(x), device=x.device)[:_MAX_SAMPLES]
                    x = x[idx]
                buffers[k].append(x.cpu())
            return _hook

        hooks.append(lyr.register_forward_hook(_make_hook(key)))

    model.eval()
    with torch.no_grad():
        for i, (input_ids, attention_mask, _) in enumerate(loader):
            if i >= n_batches:
                break
            get_cls(model, input_ids.to(device), attention_mask.to(device), use_amp)

    for h in hooks:
        h.remove()

    stats = CalibrationStats()
    for key, chunks in buffers.items():
        if not chunks:
            continue
        vals = torch.cat(chunks)
        if percentile >= 100.0:
            lo, hi = vals.min().item(), vals.max().item()
        else:
            p = (100.0 - percentile) / 200.0
            lo = torch.quantile(vals, p).item()
            hi = torch.quantile(vals, 1.0 - p).item()
        stats.ranges[key] = (lo, hi)

    return stats


# ── feature extraction ────────────────────────────────────────────────────────

@torch.no_grad()
def extract_features(model, loader, device, use_amp,
                     w_bits=None, a_bits=None,
                     weight_granularity="per_channel", calib_stats=None):
    model.eval()
    feats, labels = [], []
    for input_ids, attention_mask, y in loader:
        input_ids      = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        if w_bits is None:
            f = get_cls(model, input_ids, attention_mask, use_amp)
        elif calib_stats is not None:
            with static_quantized_forward([model], w_bits, a_bits,
                                          calib_stats, weight_granularity):
                f = get_cls(model, input_ids, attention_mask, use_amp)
        else:
            with quantized_forward([model], w_bits, a_bits, weight_granularity):
                f = get_cls(model, input_ids, attention_mask, use_amp)
        feats.append(f.cpu())
        labels.append(y)
    return torch.cat(feats).numpy(), torch.cat(labels).numpy()


# ── probe ─────────────────────────────────────────────────────────────────────

def run_probe(train_feats, train_labels, test_feats, test_labels,
              device, epochs=500, lr=1e-3, batch_size=256,
              n_seeds=3, es_patience=30, es_tol=1e-5):
    X_tr = torch.tensor(train_feats, dtype=torch.float32)
    y_tr = torch.tensor(train_labels, dtype=torch.long)
    X_te = torch.tensor(test_feats,  dtype=torch.float32)
    y_te = torch.tensor(test_labels, dtype=torch.long)

    mu = X_tr.mean(0); std = X_tr.std(0).clamp(min=1e-8)
    X_tr = (X_tr - mu) / std; X_te = (X_te - mu) / std

    n_classes = int(y_tr.max().item()) + 1
    dataset   = TensorDataset(X_tr.to(device), y_tr.to(device))
    loader    = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    accs = []
    for seed in range(n_seeds):
        torch.manual_seed(seed)
        head = nn.Linear(X_tr.shape[1], n_classes).to(device)
        opt  = torch.optim.Adam(head.parameters(), lr=lr)
        sch  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=0)
        best_loss = float("inf"); no_improve = 0
        head.train()
        for _ in range(epochs):
            epoch_loss = 0.0
            for xb, yb in loader:
                opt.zero_grad()
                loss = F.cross_entropy(head(xb), yb)
                loss.backward(); opt.step()
                epoch_loss += loss.item()
            sch.step()
            epoch_loss /= len(loader)
            if best_loss - epoch_loss > es_tol:
                best_loss = epoch_loss; no_improve = 0
            else:
                no_improve += 1
            if no_improve >= es_patience:
                break
        head.eval()
        with torch.no_grad():
            preds = head(X_te.to(device)).argmax(1).cpu()
        accs.append((preds == y_te).float().mean().item())
    return float(np.mean(accs))


# ── per-model evaluation ──────────────────────────────────────────────────────

def evaluate_model(model, label, calib_loader,
                   probe_train_loader, probe_test_loader,
                   n_calib_batches, percentile,
                   weight_granularity, device, use_amp,
                   active_configs=None):
    print(f"\n  [{label}] calibrating ({n_calib_batches} batches, pct={percentile})...")
    stats_minmax = calibrate_text(model, calib_loader, n_calib_batches,
                                  device, percentile=100.0, use_amp=use_amp)
    stats_pct    = calibrate_text(model, calib_loader, n_calib_batches,
                                  device, percentile=percentile, use_amp=use_amp)

    pct_label     = f"pct{percentile:.4g}"
    modes         = [("dynamic", None), ("minmax", stats_minmax), (pct_label, stats_pct)]
    configs       = active_configs if active_configs is not None else QUANT_CONFIGS
    model_results = {}

    for qk, wb, ab in configs:
        print(f"\n  [{label}] [{qk}] extracting features...")
        row = {}
        for mode_name, calib_stats in modes:
            tr_f, tr_l = extract_features(model, probe_train_loader, device, use_amp,
                                          wb, ab, weight_granularity, calib_stats)
            te_f, te_l = extract_features(model, probe_test_loader,  device, use_amp,
                                          wb, ab, weight_granularity, calib_stats)
            acc = run_probe(tr_f, tr_l, te_f, te_l, device)
            row[mode_name] = round(acc, 4)
            print(f"    {mode_name:<12}  acc={acc:.4f}")
        model_results[qk] = row

    return model_results, pct_label


# ── summary table ─────────────────────────────────────────────────────────────

def print_table(results, labels, pct_label, quant_configs):
    modes = ["dynamic", "minmax", pct_label]
    col_w = 10
    print(f"\n{'Config':<8}", end="")
    for lbl in labels:
        print(f"  {lbl[:((col_w + 1) * len(modes) - 1)]:<{(col_w + 1) * len(modes) - 1}}", end="")
    print()
    print(f"{'':8}", end="")
    for _ in labels:
        for m in modes:
            print(f"  {m:<{col_w - 2}}", end="")
    print()
    print("─" * (8 + len(labels) * ((col_w + 1) * len(modes) + 2)))
    for qk, _, _ in quant_configs:
        print(f"{qk:<8}", end="")
        for lbl in labels:
            row = results.get(lbl, {}).get(qk, {})
            for m in modes:
                v = row.get(m)
                print(f"  {v:.4f}" if v is not None else f"  {'—':>{col_w - 2}}", end="")
        print()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",              default="distilbert-base-uncased")
    parser.add_argument("--dataset",            default="sst2", choices=["sst2"])
    parser.add_argument("--max_length",         type=int,   default=128)
    parser.add_argument("--model_a",            required=True,
                        help="Path to .pt checkpoint (key='student') or 'pretrained'.")
    parser.add_argument("--model_a_label",      default="model_a")
    parser.add_argument("--model_b",            required=True,
                        help="Path to .pt checkpoint (key='student') or 'pretrained'.")
    parser.add_argument("--model_b_label",      default="model_b")
    parser.add_argument("--n_calib_batches",    type=int,   default=32,
                        help="Number of train batches used for activation calibration.")
    parser.add_argument("--calib_batch_size",   type=int,   default=32)
    parser.add_argument("--percentile",         type=float, default=99.9)
    parser.add_argument("--weight_granularity", default="per_channel",
                        choices=["per_tensor", "per_channel"])
    parser.add_argument("--configs",            nargs="+",
                        choices=[qk for qk, _, _ in QUANT_CONFIGS],
                        default=None,
                        help="Quant configs to evaluate. Default: all.")
    parser.add_argument("--num_workers",        type=int,   default=4)
    parser.add_argument("--data_dir",           default=None)
    parser.add_argument("--output_dir",         default=None)
    parser.add_argument("--no_amp",             action="store_true")
    args = parser.parse_args()

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = not args.no_amp and device.type == "cuda"
    slug    = args.model.replace("/", "_")
    data_dir = Path(args.data_dir or f"data/{args.dataset}")
    data_dir.mkdir(parents=True, exist_ok=True)
    hf_cache = str(data_dir / "hf_cache")
    out = Path(args.output_dir or f"runs/ptq_comparison_{args.dataset}_{slug}")
    out.mkdir(parents=True, exist_ok=True)

    active_configs = (
        [c for c in QUANT_CONFIGS if c[0] in args.configs]
        if args.configs else QUANT_CONFIGS
    )

    print("=" * 70)
    print(f"PTQ calibration comparison — {args.dataset.upper()} / {args.model}")
    print(f"  model_a : {args.model_a_label!r:30s}  ({args.model_a})")
    print(f"  model_b : {args.model_b_label!r:30s}  ({args.model_b})")
    print(f"  calib   : {args.n_calib_batches} batches × bs={args.calib_batch_size}"
          f"  percentile={args.percentile}")
    print(f"  configs : {[c[0] for c in active_configs]}")
    print("=" * 70)

    # ── tokenise & load ───────────────────────────────────────────────────
    print("\nLoading tokenizer and dataset...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir=hf_cache)
    ds        = load_dataset("nyu-mll/glue", args.dataset, cache_dir=hf_cache)

    print("  Tokenizing...")
    train_ds      = tokenize_split(ds["train"],      tokenizer, args.max_length)
    probe_test_ds = tokenize_split(ds["validation"], tokenizer, args.max_length)
    print(f"  Train: {len(train_ds)} | Probe test: {len(probe_test_ds)}\n")

    calib_loader       = DataLoader(train_ds,      batch_size=args.calib_batch_size,
                                    shuffle=False,  num_workers=args.num_workers,
                                    pin_memory=True)
    probe_train_loader = DataLoader(train_ds,      batch_size=args.calib_batch_size,
                                    shuffle=False,  num_workers=args.num_workers,
                                    pin_memory=True)
    probe_test_loader  = DataLoader(probe_test_ds, batch_size=args.calib_batch_size,
                                    shuffle=False,  num_workers=args.num_workers,
                                    pin_memory=True)

    # ── models ────────────────────────────────────────────────────────────
    print("Loading models...")
    model_a = load_model(args.model, args.model_a, device, hf_cache)
    model_b = load_model(args.model, args.model_b, device, hf_cache)

    # ── evaluate ──────────────────────────────────────────────────────────
    all_results = {}
    pct_label   = None
    for model, label in [(model_a, args.model_a_label), (model_b, args.model_b_label)]:
        res, pct_label = evaluate_model(
            model, label, calib_loader,
            probe_train_loader, probe_test_loader,
            args.n_calib_batches, args.percentile,
            args.weight_granularity, device, use_amp,
            active_configs=active_configs,
        )
        all_results[label] = res

    # ── summary table ─────────────────────────────────────────────────────
    print(f"\n{'Results':=^70}")
    print_table(all_results, [args.model_a_label, args.model_b_label], pct_label,
                active_configs)

    # ── save ──────────────────────────────────────────────────────────────
    out_path = out / "results.json"
    with open(out_path, "w") as f:
        json.dump({
            "backbone":           slug,
            "dataset":            args.dataset,
            "model_a":            args.model_a,
            "model_a_label":      args.model_a_label,
            "model_b":            args.model_b,
            "model_b_label":      args.model_b_label,
            "n_calib_batches":    args.n_calib_batches,
            "percentile":         args.percentile,
            "weight_granularity": args.weight_granularity,
            "results":            all_results,
        }, f, indent=2)
    print(f"\nResults → {out_path}")


if __name__ == "__main__":
    main()
