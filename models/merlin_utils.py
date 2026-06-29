"""
Shared helpers for the Merlin CT scripts: default data paths and the dataset
wrappers used across training, feature caching, QAT and PTQ comparison.

Kept separate from models/merlin.py (the MerlinEncoder model) so a script can
import just the dataset/path helpers without pulling in the model.
"""

from pathlib import Path

import torch
import torch.nn.functional as F

_ROOT = Path(__file__).parent.parent

# Default Merlin data paths (relative to the project root). The CT-CLIP data
# repo is expected as a sibling of this project.
CT_DATA          = _ROOT.parent / "CT-CLIP" / "data"
DEFAULT_DATA_DIR = str(CT_DATA / "merlin_data")
DEFAULT_REPORTS  = str(CT_DATA / "reports_final.xlsx")
DEFAULT_LABELS   = str(CT_DATA / "zero_shot_findings_disease_cls.csv")
DEFAULT_METADATA = str(CT_DATA / "metadata.csv")


class IndexedDataset(torch.utils.data.Dataset):
    """Wraps any dataset to return (index, x, y) — the index lets a training loop
    look the sample up in a pre-computed teacher feature cache."""
    def __init__(self, ds):
        self._ds = ds

    def __len__(self):
        return len(self._ds)

    def __getitem__(self, i):
        x, y = self._ds[i]
        return i, x, y


class ResizedDataset(torch.utils.data.Dataset):
    """Trilinearly resizes CT volumes to a fixed (D, H, W) on load.

    Used to downsample from the quantized_ft default of (240, 480, 480) to the
    original Merlin training resolution of (160, 224, 224), reducing volume size
    ~7× and allowing much larger batch sizes.
    """
    def __init__(self, ds, target_shape):
        self._ds = ds
        self._target = tuple(target_shape)  # (D, H, W)

    def __len__(self):
        return len(self._ds)

    @property
    def label_names(self):
        return self._ds.label_names

    def __getitem__(self, i):
        x, y = self._ds[i]               # x: (1, D, H, W)
        if tuple(x.shape[1:]) != self._target:
            x = F.interpolate(
                x.unsqueeze(0),          # (1, 1, D, H, W)
                size=self._target,
                mode="trilinear",
                align_corners=False,
            ).squeeze(0)                 # (1, D, H, W)
        return x, y
