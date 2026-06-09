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
from torchvision import models, transforms

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from models.quantization import (
    CalibrationStats, quantized_forward, static_quantized_forward,
)


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
# Merlin CT encoder wrapper
# ---------------------------------------------------------------------------

class MerlinEncoder(nn.Module):
    """Loads only Merlin's image encoder (I3ResNet-152) — text encoder never instantiated.

    Input  : (B, 1, D, H, W) float32 — CT volumes from MerlinDataset (D=240, H=W=480).
    Output : (B, 2048) float32 — raw ResNet-152 avgpool features (pre contrastive head).

    Why 2048 and not 512: I3ResNet with ImageEmbedding=True returns the avgpool output
    directly (line 96 of i3res.py), skipping the Conv3d contrastive projection head.
    The 512D projection is only used during CLIP training; raw 2048D is standard for
    downstream transfer.

    Gradient checkpointing is already built into I3ResNet (checkpoint.checkpoint on
    all four ResNet stages) — no external wrapping needed.

    The `image_encoder` property exposes the ImageEncoder submodule so that
    quantized_forward patches only those layers.
    """

    FEAT_DIM = 2048

    def __init__(self):
        super().__init__()
        import os
        import merlin.models.load as _mload
        from merlin.models.build import ImageEncoder
        from merlin.models.load import MODEL_CONFIGS, REPO_ID
        from merlin.utils import download_file

        chk_name = MODEL_CONFIGS["default"]["checkpoint"]
        local_dir = os.path.join(os.path.dirname(os.path.abspath(_mload.__file__)), "checkpoints")
        chk_path  = os.path.join(local_dir, chk_name)

        if not os.path.exists(chk_path):
            print(f"Downloading {chk_name}...")
            download_file(repo_id=REPO_ID, filename=chk_name, local_dir=local_dir)

        # Build image encoder only — Longformer never loaded
        self._enc = ImageEncoder(ImageEmbedding=True)

        # Full checkpoint has keys like "encode_image.i3_resnet.conv1.weight".
        # Strip prefix and load into ImageEncoder directly.
        full_sd = torch.load(chk_path, map_location="cpu")
        enc_sd = {k[len("encode_image."):]: v
                  for k, v in full_sd.items() if k.startswith("encode_image.")}
        self._enc.load_state_dict(enc_sd, strict=True)

    @property
    def image_encoder(self) -> nn.Module:
        return self._enc

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, D, H, W)
        # i3_resnet.forward() first does x.permute(0,1,4,2,3) expecting (B,C,H,W,D).
        # So we permute (B,1,D,H,W) → (B,1,H,W,D) here.
        b = x.shape[0]
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        out = self._enc(x)
        # i3_resnet returns (1, B, 2048) for ImageEmbedding=True — normalise to (B, 2048)
        return out.reshape(b, -1)


# ---------------------------------------------------------------------------
# Linear probe
# ---------------------------------------------------------------------------

def run_probe(
    train_feats: np.ndarray, train_labels: np.ndarray,
    test_feats:  np.ndarray, test_labels:  np.ndarray,
    device: torch.device,
    epochs: int = 500, lr: float = 1e-3, batch_size: int = 256,
    n_seeds: int = 3, es_patience: int = 30, es_tol: float = 1e-5,
) -> float:
    """Linear probe: nn.Linear + Adam + cosine LR, averaged over n_seeds."""
    X_tr = torch.tensor(train_feats, dtype=torch.float32)
    y_tr = torch.tensor(train_labels, dtype=torch.long)
    X_te = torch.tensor(test_feats,  dtype=torch.float32)
    y_te = torch.tensor(test_labels, dtype=torch.long)

    mu  = X_tr.mean(0)
    std = X_tr.std(0).clamp(min=1e-8)
    X_tr = (X_tr - mu) / std
    X_te = (X_te - mu) / std

    n_classes = int(y_tr.max().item()) + 1
    dataset   = TensorDataset(X_tr.to(device), y_tr.to(device))
    loader    = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    accs = []
    for seed in range(n_seeds):
        torch.manual_seed(seed)
        head = nn.Linear(X_tr.shape[1], n_classes).to(device)
        opt  = torch.optim.Adam(head.parameters(), lr=lr)
        sch  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=0)

        best_loss, no_improve = float("inf"), 0
        head.train()
        for _ in range(epochs):
            epoch_loss = 0.0
            for xb, yb in loader:
                opt.zero_grad()
                loss = F.cross_entropy(head(xb), yb)
                loss.backward()
                opt.step()
                epoch_loss += loss.item()
            sch.step()
            epoch_loss /= len(loader)
            if best_loss - epoch_loss > es_tol:
                best_loss, no_improve = epoch_loss, 0
            else:
                no_improve += 1
            if no_improve >= es_patience:
                break

        head.eval()
        with torch.no_grad():
            preds = head(X_te.to(device)).argmax(1).cpu()
        accs.append((preds == y_te).float().mean().item())

    return float(np.mean(accs))
