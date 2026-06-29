"""
Merlin CT image-encoder wrapper.

Kept in its own module so the lightweight torchvision probe/eval utilities in
eval_utils.py don't transitively pull in the merlin / quantized_ft model stack
just to import a linear probe.
"""

import torch
import torch.nn as nn


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
