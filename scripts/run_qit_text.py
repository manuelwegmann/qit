"""
QIT on SST-2 with DistilBERT (or any HuggingFace encoder).

Mirrors run_qit.py for text:
  - Teacher : frozen pretrained encoder, [CLS] token → 768-dim features
  - Student : same weights, QIT-trained under random quantization
  - Loss    : cosine | mse | kl
  - Probe   : linear head, accuracy at FP / W8A8 / W4A4 / W2A4 on SST-2 validation split

Teacher features for the full train set are cached to disk on first run.
"""

import argparse
import json
import shutil
import sys
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from datasets import load_dataset
from torch.utils.data import DataLoader, Subset, TensorDataset
from transformers import AutoModel, AutoTokenizer

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from models.quantization import quantized_forward, BitWidthSampler


QUANT_CONFIGS = [
    ("fp",   None, None),
    ("w8a8", 8,    8),
    ("w6a6", 6,    6),
    ("w4a6", 4,    6),
    ("w4a4", 4,    4),
    ("w2a4", 2,    4),
]


def _pbar(iterable, **kwargs):
    n = len(iterable) if hasattr(iterable, "__len__") else None
    extra = {"miniters": max(1, n // 20), "mininterval": 0} if n else {"mininterval": 30}
    return tqdm.tqdm(iterable, **extra, **kwargs)


# ── dataset ───────────────────────────────────────────────────────────────────

class _TextDataset(torch.utils.data.Dataset):
    """Pre-tokenized dataset returning (input_ids, attention_mask, label)."""
    def __init__(self, input_ids, attention_mask, labels):
        self.input_ids      = input_ids
        self.attention_mask = attention_mask
        self.labels         = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attention_mask[idx], self.labels[idx]


class _IndexedSubset(torch.utils.data.Dataset):
    """Subset that also returns the original full-dataset index."""
    def __init__(self, subset: Subset):
        self.subset = subset

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        input_ids, attention_mask, label = self.subset[idx]
        return input_ids, attention_mask, label, self.subset.indices[idx]


def tokenize_split(hf_split, tokenizer, max_length, text_field="sentence"):
    enc = tokenizer(
        list(hf_split[text_field]),  # cast from PyArrow array to plain list
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    labels = torch.tensor(list(hf_split["label"]), dtype=torch.long)
    return _TextDataset(enc["input_ids"], enc["attention_mask"], labels)


# ── model forward ─────────────────────────────────────────────────────────────

def get_cls(model, input_ids, attention_mask, use_amp):
    """Run encoder and return [CLS] token as feature vector."""
    if use_amp:
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = model(input_ids=input_ids, attention_mask=attention_mask)
            return out.last_hidden_state[:, 0, :].float()
    out = model(input_ids=input_ids, attention_mask=attention_mask)
    return out.last_hidden_state[:, 0, :].float()


# ── loss / probe ──────────────────────────────────────────────────────────────

def compute_qit_loss(z_student, z_teacher, loss_type):
    zt = z_teacher.detach()
    if loss_type == "cosine":
        return 1 - F.cosine_similarity(z_student, zt).mean()
    if loss_type == "mse":
        return F.mse_loss(z_student, zt)
    if loss_type == "kl":
        return F.kl_div(F.log_softmax(z_student, dim=-1),
                        F.softmax(zt, dim=-1), reduction="batchmean")
    raise ValueError(f"Unknown loss_type: {loss_type!r}")


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


# ── feature extraction ────────────────────────────────────────────────────────

@torch.no_grad()
def cache_teacher_features(teacher, loader, device, use_amp):
    teacher.eval()
    feats = []
    for input_ids, attention_mask, _ in _pbar(loader, desc="  caching", leave=False):
        feats.append(get_cls(teacher, input_ids.to(device),
                             attention_mask.to(device), use_amp).cpu())
    return torch.cat(feats)


@torch.no_grad()
def extract_features(model, loader, device, use_amp, w_bits=None, a_bits=None):
    model.eval()
    feats, labels = [], []
    for input_ids, attention_mask, y in _pbar(loader, desc="    extracting", leave=False):
        input_ids      = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        if w_bits is None:
            f = get_cls(model, input_ids, attention_mask, use_amp)
        else:
            with quantized_forward([model], w_bits, a_bits):
                f = get_cls(model, input_ids, attention_mask, use_amp)
        feats.append(f.cpu()); labels.append(y)
    return torch.cat(feats).numpy(), torch.cat(labels).numpy()


# ── training / validation ─────────────────────────────────────────────────────

def train_epoch(student, loader, optimizer, loss_type, bit_sampler,
                device, use_amp, teacher_feats_cache, fp_loss_weight=0.0):
    student.train()
    total_loss = total_fp_sim = n = 0

    for input_ids, attention_mask, _, full_idx in _pbar(loader, desc="  batches", leave=False):
        input_ids      = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        z_teacher      = teacher_feats_cache[full_idx].to(device)
        w_bits, a_bits = bit_sampler.sample()

        optimizer.zero_grad()
        with quantized_forward([student], w_bits, a_bits):
            z_student_q = get_cls(student, input_ids, attention_mask, use_amp)
        loss = compute_qit_loss(z_student_q, z_teacher, loss_type)

        if fp_loss_weight > 0.0:
            z_fp = get_cls(student, input_ids, attention_mask, use_amp)
            loss = loss + fp_loss_weight * compute_qit_loss(z_fp, z_teacher, loss_type)

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            z_fp = (z_fp.detach() if fp_loss_weight > 0.0
                    else get_cls(student, input_ids, attention_mask, use_amp))
            fp_sim = F.cosine_similarity(z_fp, z_teacher).mean().item()

        total_loss += loss.item(); total_fp_sim += fp_sim; n += 1

    return total_loss / max(n, 1), total_fp_sim / max(n, 1)


@torch.no_grad()
def validate_epoch(student, loader, loss_type, all_configs, device, use_amp, teacher_feats_cache):
    student.eval()
    total_loss = total_fp_sim = n = 0

    for i, (input_ids, attention_mask, _, full_idx) in enumerate(loader):
        input_ids      = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        z_teacher      = teacher_feats_cache[full_idx].to(device)
        w_bits, a_bits = all_configs[i % len(all_configs)]

        with quantized_forward([student], w_bits, a_bits):
            z_student = get_cls(student, input_ids, attention_mask, use_amp)
        z_fp = get_cls(student, input_ids, attention_mask, use_amp)

        total_loss   += compute_qit_loss(z_student, z_teacher, loss_type).item()
        total_fp_sim += F.cosine_similarity(z_fp, z_teacher).mean().item()
        n += 1

    return total_loss / max(n, 1), total_fp_sim / max(n, 1)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",           type=str,   default="distilbert-base-uncased")
    parser.add_argument("--dataset",         type=str,   default="sst2", choices=["sst2"])
    parser.add_argument("--max_length",      type=int,   default=128)
    parser.add_argument("--epochs",          type=int,   default=50)
    parser.add_argument("--patience",        type=int,   default=10)
    parser.add_argument("--val_window",      type=int,   default=3)
    parser.add_argument("--lr",              type=float, default=1e-4)
    parser.add_argument("--lr_schedule",     type=str,   default="cosine",
                        choices=["cosine", "constant"])
    parser.add_argument("--batch_size",      type=int,   default=32)
    parser.add_argument("--loss",            type=str,   default="cosine",
                        choices=["cosine", "mse", "kl"])
    parser.add_argument("--fp_loss_weight",  type=float, default=0.0)
    parser.add_argument("--bit_sampling",    type=str,   default="wor",
                        choices=["wor", "random"])
    parser.add_argument("--w_bits_min",      type=int,   default=2)
    parser.add_argument("--w_bits_max",      type=int,   default=4)
    parser.add_argument("--a_bits_min",      type=int,   default=4)
    parser.add_argument("--a_bits_max",      type=int,   default=8)
    parser.add_argument("--n_train",         type=int,   default=7000)
    parser.add_argument("--num_workers",     type=int,   default=4)
    parser.add_argument("--no_amp",          action="store_true")
    parser.add_argument("--data_dir",        type=str,   default=None)
    parser.add_argument("--output_dir",      type=str,   default=None)
    args = parser.parse_args()

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = not args.no_amp and device.type == "cuda"
    slug    = args.model.replace("/", "_")
    out     = Path(args.output_dir or f"runs/qit_{args.dataset}_{slug}")
    out.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir or f"data/{args.dataset}")
    data_dir.mkdir(parents=True, exist_ok=True)
    hf_cache = str(data_dir / "hf_cache")

    w_range = (args.w_bits_min, args.w_bits_max)
    a_range = (args.a_bits_min, args.a_bits_max)

    print("=" * 70)
    print(f"QIT validation — {args.dataset.upper()} / {args.model}")
    print("=" * 70)
    print(f"  model        : {args.model}")
    print(f"  dataset      : {args.dataset}")
    print(f"  n_train      : {args.n_train}")
    print(f"  epochs       : {args.epochs}  patience: {args.patience}  val_window: {args.val_window}")
    print(f"  lr           : {args.lr:.1e}  schedule: {args.lr_schedule}")
    print(f"  batch_size   : {args.batch_size}")
    print(f"  loss         : {args.loss}  fp_loss_weight: {args.fp_loss_weight}")
    print(f"  w_bits range : {w_range}  a_bits range: {a_range}")
    print(f"  amp          : {use_amp}")
    print(f"  output       : {out}\n")

    # ── tokenise & split ──────────────────────────────────────────────────
    print("Loading tokenizer and dataset...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir=hf_cache)
    ds        = load_dataset("nyu-mll/glue", args.dataset, cache_dir=hf_cache)

    print("  Tokenizing train split...")
    full_train_ds = tokenize_split(ds["train"],      tokenizer, args.max_length)
    print("  Tokenizing validation split...")
    probe_test_ds = tokenize_split(ds["validation"], tokenizer, args.max_length)

    # QIT train/val split (labels ignored during QIT)
    rng      = np.random.default_rng(42)
    all_idx  = rng.permutation(len(full_train_ds)).tolist()
    n_subset = len(full_train_ds) if args.n_train < 0 else min(args.n_train, len(full_train_ds))
    subset   = all_idx[:n_subset]
    n_qit    = int(0.9 * n_subset)
    qit_train_ds = _IndexedSubset(Subset(full_train_ds, subset[:n_qit]))
    qit_val_ds   = _IndexedSubset(Subset(full_train_ds, subset[n_qit:]))

    qit_train_loader   = DataLoader(qit_train_ds,   batch_size=args.batch_size, shuffle=True,
                                    num_workers=args.num_workers, pin_memory=True)
    qit_val_loader     = DataLoader(qit_val_ds,     batch_size=args.batch_size, shuffle=False,
                                    num_workers=args.num_workers, pin_memory=True)
    cache_loader       = DataLoader(full_train_ds,  batch_size=args.batch_size, shuffle=False,
                                    num_workers=args.num_workers, pin_memory=True)
    probe_train_loader = DataLoader(full_train_ds,  batch_size=args.batch_size, shuffle=False,
                                    num_workers=args.num_workers, pin_memory=True)
    probe_test_loader  = DataLoader(probe_test_ds,  batch_size=args.batch_size, shuffle=False,
                                    num_workers=args.num_workers, pin_memory=True)

    print(f"  QIT train: {len(qit_train_ds)} | QIT val: {len(qit_val_ds)} | "
          f"Probe train: {len(full_train_ds)} | Probe test: {len(probe_test_ds)}\n")

    # ── models ────────────────────────────────────────────────────────────
    print(f"Loading teacher ({args.model})...")
    teacher = AutoModel.from_pretrained(args.model, cache_dir=hf_cache).to(device)
    for p in teacher.parameters():
        p.requires_grad_(False)
    teacher.eval()

    print(f"Loading student (same weights)...")
    student = AutoModel.from_pretrained(args.model, cache_dir=hf_cache).to(device)

    results        = {"epochs": [], "probe": {}}
    best_ckpt_path = out / "checkpoint_best.pt"

    if args.epochs > 0:
        feat_cache_path = data_dir / f"teacher_feats_{slug}.pt"
        if feat_cache_path.exists():
            print(f"Loading cached teacher features from {feat_cache_path}...")
            teacher_feats_cache = torch.load(feat_cache_path, map_location="cpu")
        else:
            print("Computing teacher features (first run for this model)...")
            teacher_feats_cache = cache_teacher_features(teacher, cache_loader, device, use_amp)
            torch.save(teacher_feats_cache, feat_cache_path)
            print(f"  Saved to {feat_cache_path}")
        print(f"  Cache shape: {teacher_feats_cache.shape}\n")

        bit_sampler = BitWidthSampler(w_range, a_range, mode=args.bit_sampling)
        all_configs = bit_sampler.configs
        print(f"  bit-width grid : {len(all_configs)} configs  "
              f"({w_range[1]-w_range[0]+1}w × {a_range[1]-a_range[0]+1}a)\n")

        optimizer = torch.optim.Adam(student.parameters(), lr=args.lr)
        scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(
                         optimizer, T_max=args.epochs, eta_min=0)
                     if args.lr_schedule == "cosine" else None)

        val_history       = deque(maxlen=args.val_window)
        ckpt_buffer       = deque(maxlen=args.val_window)
        best_avg_val      = float("inf")
        best_mid_epoch    = -1
        epochs_no_improve = 0
    else:
        print("[epochs=0] Skipping QIT training — probe only.\n")

    # ── training ──────────────────────────────────────────────────────────
    epoch = 0
    if args.epochs > 0:
        print(f"\n{'Training':=^70}")
    for epoch in range(1, args.epochs + 1):
        loss, fp_sim = train_epoch(
            student, qit_train_loader, optimizer, args.loss,
            bit_sampler, device, use_amp, teacher_feats_cache,
            fp_loss_weight=args.fp_loss_weight,
        )
        val_loss, val_fp_sim = validate_epoch(
            student, qit_val_loader, args.loss,
            all_configs, device, use_amp, teacher_feats_cache,
        )

        val_history.append(val_loss)
        avg_val     = sum(val_history) / len(val_history)
        window_full = len(val_history) == args.val_window
        is_best     = window_full and avg_val < best_avg_val
        if is_best:
            best_avg_val      = avg_val
            epochs_no_improve = 0
        elif window_full:
            epochs_no_improve += 1

        marker = f" *best* (avg={avg_val:.4f})" if is_best else (
                 f" (avg={avg_val:.4f})" if window_full else "")
        print(f"[epoch {epoch:3d}/{args.epochs}]  "
              f"loss={loss:.4f}  fp_sim={fp_sim:.4f}  |  "
              f"val_loss={val_loss:.4f}  val_fp_sim={val_fp_sim:.4f}{marker}")
        results["epochs"].append({
            "epoch": epoch, "loss": loss, "fp_sim": fp_sim,
            "val_loss": val_loss, "val_fp_sim": val_fp_sim,
        })

        slot = (epoch - 1) % args.val_window
        ckpt = {"epoch": epoch, "student": student.state_dict()}
        torch.save(ckpt, out / f"_buf{slot}.pt")
        torch.save(ckpt, out / "checkpoint_latest.pt")
        ckpt_buffer.append((epoch, slot))
        if is_best:
            mid_epoch, mid_slot = list(ckpt_buffer)[args.val_window // 2]
            shutil.copy(out / f"_buf{mid_slot}.pt", best_ckpt_path)
            best_mid_epoch = mid_epoch

        if scheduler is not None:
            scheduler.step()

        if args.patience > 0 and window_full and epochs_no_improve >= args.patience:
            print(f"\n[early stop] {args.val_window}-epoch avg val loss did not improve "
                  f"for {args.patience} epochs. Best avg: {best_avg_val:.4f}")
            break

    if epoch > 0:
        torch.save({"epoch": epoch, "student": student.state_dict()},
                   out / "checkpoint_final.pt")
        for i in range(args.val_window):
            p = out / f"_buf{i}.pt"
            if p.exists():
                p.unlink()

    if best_ckpt_path.exists():
        print(f"\nLoading best checkpoint (epoch {best_mid_epoch}, "
              f"window avg={best_avg_val:.4f}) for evaluation...")
        student.load_state_dict(torch.load(best_ckpt_path, map_location=device)["student"])
    else:
        print("\n[epochs=0] Using initial student weights for evaluation.")

    # ── probe ─────────────────────────────────────────────────────────────
    print(f"\n{'Probe evaluation':=^70}")
    print(f"  {'Config':<8}  {'Teacher acc':>12}  {'Student acc':>12}  "
          f"{'Δ':>8}  {'feat_sim':>10}")

    for qk, wb, ab in QUANT_CONFIGS:
        print(f"\n  [{qk}] extracting features...")
        t_tr, l_tr = extract_features(teacher, probe_train_loader, device, use_amp, wb, ab)
        s_tr, _    = extract_features(student, probe_train_loader, device, use_amp, wb, ab)
        t_te, l_te = extract_features(teacher, probe_test_loader,  device, use_amp, wb, ab)
        s_te, _    = extract_features(student, probe_test_loader,  device, use_amp, wb, ab)

        feat_sim    = float(F.cosine_similarity(torch.tensor(s_tr), torch.tensor(t_tr)).mean())
        teacher_acc = run_probe(t_tr, l_tr, t_te, l_te, device)
        student_acc = run_probe(s_tr, l_tr, s_te, l_te, device)
        delta       = student_acc - teacher_acc

        print(f"  {qk:<8}  {teacher_acc:>12.4f}  {student_acc:>12.4f}  "
              f"  {delta:>+8.4f}  {feat_sim:>10.4f}")
        results["probe"][qk] = {
            "teacher_acc": teacher_acc, "student_acc": student_acc, "feat_sim": feat_sim,
        }

    out_path = out / "results.json"
    with open(out_path, "w") as f:
        json.dump({
            "model": args.model, "dataset": args.dataset,
            "epochs": args.epochs, "patience": args.patience, "val_window": args.val_window,
            "lr": args.lr, "lr_schedule": args.lr_schedule,
            "batch_size": args.batch_size, "loss": args.loss,
            "fp_loss_weight": args.fp_loss_weight,
            "w_bits_range": list(w_range), "a_bits_range": list(a_range),
            "results": results,
        }, f, indent=2)
    print(f"\nResults → {out_path}")


if __name__ == "__main__":
    main()
