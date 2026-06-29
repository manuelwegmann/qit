"""
Shared evaluation utilities: backbone construction, feature extraction, linear probe.
Used by run_qit.py, run_ptq_comparison.py, and run_qat.py.
"""

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, models, transforms

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from models.quantization import (
    CalibrationStats, quantized_forward, static_quantized_forward,
)
# Re-exported for backwards compatibility: MerlinEncoder used to live here.
# Its real home is now models/merlin.py (keeps the heavy merlin/quantized_ft
# import stack out of the lightweight torchvision probe utilities).
from models.merlin import MerlinEncoder  # noqa: F401


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

QUANT_CONFIGS = [
    ("fp",   None, None),
    ("w8a8", 8,    8),
    ("w6a6", 6,    6),
    ("w4a6", 4,    6),
    ("w4a4", 4,    4),
    ("w2a4", 2,    4),
]

# Output feature dimension for each backbone (after head replaced with Identity)
FEATURE_DIMS = {
    "resnet18":          512,
    "efficientnet_b0":   1280,
    "mobilenet_v3_small": 576,
    "vit_b_16":          768,
    "swin_t":            768,
}

_TRANSFORM = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pbar(iterable, **kwargs):
    if not sys.stderr.isatty():
        kwargs["disable"] = True
    n = len(iterable) if hasattr(iterable, "__len__") else None
    extra = {"miniters": max(1, n // 20), "mininterval": 0} if n else {"mininterval": 30}
    return tqdm.tqdm(iterable, **extra, **kwargs)


def build_backbone(name: str) -> nn.Module:
    if name == "resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        m.fc = nn.Identity()
    elif name == "efficientnet_b0":
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        m.classifier = nn.Identity()
    elif name == "mobilenet_v3_small":
        m = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        m.classifier = nn.Identity()
    elif name == "vit_b_16":
        m = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        m.heads = nn.Identity()
    elif name == "swin_t":
        m = models.swin_t(weights=models.Swin_T_Weights.IMAGENET1K_V1)
        m.head = nn.Identity()
    else:
        raise ValueError(f"Unknown backbone: {name!r}")
    return m


def load_backbone(backbone_name: str, checkpoint: str, device: torch.device) -> nn.Module:
    """Build backbone and load weights from checkpoint path or 'pretrained'.

    checkpoint='pretrained' → ImageNet pretrained weights (torchvision default).
    checkpoint=<path>       → .pt file with a 'student' key containing the state dict.
    """
    model = build_backbone(backbone_name)
    if checkpoint != "pretrained":
        ckpt = torch.load(checkpoint, map_location=device)
        state = ckpt.get("student", ckpt)
        model.load_state_dict(state)
    return model.to(device)


def load_vision_dataset(dataset: str, split: str, data_dir: str,
                        transform=_TRANSFORM, download: bool = True):
    """Construct one torchvision split with the standard QIT transform.

    Centralizes the CIFAR10(train=...) vs STL10(split=...) constructor mapping so
    the experiment scripts compose their splits from a single place. Returns the
    same Dataset object a direct constructor call would.

      stl10   splits: 'unlabeled' | 'train' | 'test'
      cifar10 splits: 'train' | 'test'
    """
    if dataset == "stl10":
        return datasets.STL10(data_dir, split=split, download=download, transform=transform)
    if dataset == "cifar10":
        if split not in ("train", "test"):
            raise ValueError(f"cifar10 has no split {split!r} (use 'train' or 'test')")
        return datasets.CIFAR10(data_dir, train=(split == "train"),
                                download=download, transform=transform)
    raise ValueError(f"Unknown dataset: {dataset!r}")


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_features(
    backbone: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
    w_bits: Optional[int] = None,
    a_bits: Optional[int] = None,
    weight_granularity: str = "per_tensor",
    calib_stats: Optional[CalibrationStats] = None,
) -> tuple:
    """Extract (features, labels) from loader. Quantizes if w_bits/a_bits given.

    calib_stats — if provided alongside w_bits/a_bits, uses static (calibrated)
                  quantization instead of dynamic per-sample min-max.
    """
    backbone.eval()
    feats, labels = [], []

    for batch in _pbar(loader, desc="    extracting", leave=False):
        x, y = batch[0].to(device), batch[1]

        if w_bits is None:
            ctx = torch.amp.autocast("cuda", dtype=torch.bfloat16) if use_amp else _null_ctx()
            with ctx:
                f = backbone(x)
        elif calib_stats is not None:
            ctx = torch.amp.autocast("cuda", dtype=torch.bfloat16) if use_amp else _null_ctx()
            with ctx:
                with static_quantized_forward([backbone], w_bits, a_bits, calib_stats, weight_granularity):
                    f = backbone(x)
        else:
            ctx = torch.amp.autocast("cuda", dtype=torch.bfloat16) if use_amp else _null_ctx()
            with ctx:
                with quantized_forward([backbone], w_bits, a_bits, weight_granularity):
                    f = backbone(x)

        feats.append(f.cpu().float())
        labels.append(y)

    return torch.cat(feats).numpy(), torch.cat(labels).numpy()


class _null_ctx:
    def __enter__(self): return self
    def __exit__(self, *a): pass


# ---------------------------------------------------------------------------
# Linear probe
# ---------------------------------------------------------------------------

def split_train_val(n: int, val_frac: float = 0.2, seed: int = 0):
    """Deterministic train/val index split for carving a probe-validation set
    out of a probe-train feature set (used for run_probe's early stopping)."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_val = max(1, int(n * val_frac))
    return idx[n_val:], idx[:n_val]


def run_probe(
    train_feats: np.ndarray, train_labels: np.ndarray,
    val_feats:   np.ndarray, val_labels:   np.ndarray,
    test_feats:  np.ndarray, test_labels:  np.ndarray,
    device: torch.device,
    epochs: int = 500, lr: float = 1e-3, batch_size: int = 256,
    n_seeds: int = 3, es_patience: int = 30, es_tol: float = 1e-5,
) -> float:
    """Linear probe: nn.Linear + Adam + cosine LR, averaged over n_seeds.

    Early stopping and best-checkpoint selection use held-out val_feats/val_labels
    (not the training loss) to avoid overfitting the probe head, and the
    best-val-epoch weights are restored before evaluating accuracy on test_feats.
    """
    X_tr  = torch.tensor(train_feats, dtype=torch.float32)
    y_tr  = torch.tensor(train_labels, dtype=torch.long)
    X_val = torch.tensor(val_feats, dtype=torch.float32)
    y_val = torch.tensor(val_labels, dtype=torch.long)
    X_te  = torch.tensor(test_feats,  dtype=torch.float32)
    y_te  = torch.tensor(test_labels, dtype=torch.long)

    mu  = X_tr.mean(0)
    std = X_tr.std(0).clamp(min=1e-8)
    X_tr  = (X_tr  - mu) / std
    X_val = (X_val - mu) / std
    X_te  = (X_te  - mu) / std

    n_classes = int(y_tr.max().item()) + 1
    dataset   = TensorDataset(X_tr.to(device), y_tr.to(device))
    loader    = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    X_val_dev = X_val.to(device)
    y_val_dev = y_val.to(device)

    accs = []
    for seed in range(n_seeds):
        torch.manual_seed(seed)
        head = nn.Linear(X_tr.shape[1], n_classes).to(device)
        opt  = torch.optim.Adam(head.parameters(), lr=lr)
        sch  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=0)

        best_val_loss, no_improve = float("inf"), 0
        best_state = {k: v.clone() for k, v in head.state_dict().items()}
        for _ in range(epochs):
            head.train()
            for xb, yb in loader:
                opt.zero_grad()
                loss = F.cross_entropy(head(xb), yb)
                loss.backward()
                opt.step()
            sch.step()

            head.eval()
            with torch.no_grad():
                val_loss = F.cross_entropy(head(X_val_dev), y_val_dev).item()
            if best_val_loss - val_loss > es_tol:
                best_val_loss, no_improve = val_loss, 0
                best_state = {k: v.clone() for k, v in head.state_dict().items()}
            else:
                no_improve += 1
            if no_improve >= es_patience:
                break

        head.load_state_dict(best_state)
        head.eval()
        with torch.no_grad():
            preds = head(X_te.to(device)).argmax(1).cpu()
        accs.append((preds == y_te).float().mean().item())

    return float(np.mean(accs))


def probe_teacher_student(teacher, student, train_loader, test_loader,
                          w_bits, a_bits, device, use_amp,
                          weight_granularity: str = "per_tensor"):
    """Probe teacher and student at one (w_bits, a_bits) config.

    Extracts features for both models on the probe-train and probe-test loaders,
    carves a held-out val split out of probe-train (same split for both models),
    fits a linear probe for each, and returns (teacher_acc, student_acc, feat_sim)
    where feat_sim is the mean cosine similarity between student and teacher train
    features. Shared by run_qit.py and run_cross_dataset_eval.py.
    """
    t_tr, l_tr = extract_features(teacher, train_loader, device, use_amp, w_bits, a_bits, weight_granularity)
    s_tr, _    = extract_features(student, train_loader, device, use_amp, w_bits, a_bits, weight_granularity)
    t_te, l_te = extract_features(teacher, test_loader,  device, use_amp, w_bits, a_bits, weight_granularity)
    s_te, _    = extract_features(student, test_loader,  device, use_amp, w_bits, a_bits, weight_granularity)

    tr_idx, va_idx = split_train_val(len(l_tr))
    feat_sim    = float(F.cosine_similarity(torch.tensor(s_tr), torch.tensor(t_tr)).mean())
    teacher_acc = run_probe(t_tr[tr_idx], l_tr[tr_idx], t_tr[va_idx], l_tr[va_idx], t_te, l_te, device)
    student_acc = run_probe(s_tr[tr_idx], l_tr[tr_idx], s_tr[va_idx], l_tr[va_idx], s_te, l_te, device)
    return teacher_acc, student_acc, feat_sim
