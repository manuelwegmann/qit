"""
Quantization utilities with STE (straight-through estimator).

Two weight granularities are supported:
  per_tensor  — one scale for the entire weight matrix (default, coarser)
  per_channel — one scale per output channel/filter (finer, closer to real deployment)

Activations are always quantized per-tensor: per-token would require knowing
which axis is the sequence dimension, and at runtime per-tensor is the common
choice for activations anyway.

Usage:
    w_bits, a_bits = sample_bits()
    with quantized_forward([backbone], w_bits, a_bits):                     # per-tensor
        z = backbone(x)
    with quantized_forward([backbone], w_bits, a_bits, "per_channel"):      # per-channel
        z = backbone(x)
"""

import random
from contextlib import contextmanager
from typing import List, Literal, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Fake quantizers (STE)
# ---------------------------------------------------------------------------

class _FakeQuantize(torch.autograd.Function):
    """Per-tensor min-max uniform quantizer with straight-through estimator."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, n_bits: int) -> torch.Tensor:
        x_min = x.detach().min()
        x_max = x.detach().max()
        scale = (x_max - x_min).clamp(min=1e-8) / (2 ** n_bits - 1)
        return torch.round((x - x_min) / scale) * scale + x_min

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class _FakeQuantizePerChannel(torch.autograd.Function):
    """Per-output-channel min-max uniform quantizer with straight-through estimator.

    Works for both Linear weights [out, in] and Conv2d weights [out, in, kH, kW].
    Each output channel (row / filter) gets its own scale, so one outlier channel
    does not blow up the quantization step for all other channels.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, n_bits: int) -> torch.Tensor:
        # Flatten all dims except the output-channel dim, compute per-row min/max
        x_flat = x.detach().reshape(x.shape[0], -1)
        x_min  = x_flat.min(dim=1).values.reshape(-1, *([1] * (x.ndim - 1)))
        x_max  = x_flat.max(dim=1).values.reshape(-1, *([1] * (x.ndim - 1)))
        scale  = (x_max - x_min).clamp(min=1e-8) / (2 ** n_bits - 1)
        return (torch.round((x - x_min) / scale) * scale + x_min).contiguous()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


def fake_quantize(x: torch.Tensor, n_bits: int) -> torch.Tensor:
    return _FakeQuantize.apply(x, n_bits)


def fake_quantize_per_channel(x: torch.Tensor, n_bits: int) -> torch.Tensor:
    return _FakeQuantizePerChannel.apply(x, n_bits)


# ---------------------------------------------------------------------------
# Bit-width sampler
# ---------------------------------------------------------------------------

def sample_bits(
    w_bits_range: Tuple[int, int] = (2, 8),
    a_bits_range: Tuple[int, int] = (4, 8),
) -> Tuple[int, int]:
    """Sample random bit-widths for weights and activations independently."""
    return random.randint(*w_bits_range), random.randint(*a_bits_range)


class BitWidthSampler:
    """Sampler over the full (w_bits, a_bits) Cartesian grid.

    mode="wor"    — without-replacement: exhausts all pairs before reshuffling.
                    Guarantees every config appears at least once per round.
    mode="random" — uniform random with replacement (same as plain randint).

    The `configs` property returns all pairs in sorted order — use it to drive
    deterministic validation (cycle through configs[i % len(configs)]).
    """

    def __init__(self, w_bits_range: Tuple[int, int], a_bits_range: Tuple[int, int],
                 mode: str = "wor"):
        if mode not in ("wor", "random"):
            raise ValueError(f"mode must be 'wor' or 'random', got {mode!r}")
        from itertools import product as _product
        self._all: List[Tuple[int, int]] = list(_product(
            range(w_bits_range[0], w_bits_range[1] + 1),
            range(a_bits_range[0], a_bits_range[1] + 1),
        ))
        self._mode = mode
        self._pool: List[Tuple[int, int]] = []

    def sample(self) -> Tuple[int, int]:
        if self._mode == "random":
            return random.choice(self._all)
        if not self._pool:
            self._pool = self._all.copy()
            random.shuffle(self._pool)
        return self._pool.pop()

    @property
    def configs(self) -> List[Tuple[int, int]]:
        return list(self._all)


# ---------------------------------------------------------------------------
# Context manager: quantized forward pass
# ---------------------------------------------------------------------------

WeightGranularity = Literal["per_tensor", "per_channel"]


@contextmanager
def quantized_forward(
    modules: List[nn.Module],
    w_bits: int,
    a_bits: int,
    weight_granularity: WeightGranularity = "per_tensor",
):
    """
    Temporarily replace the forward() of all nn.Linear and nn.Conv2d layers
    inside `modules` so that weights are fake-quantized to w_bits and input
    activations to a_bits.  Restores original forwards on exit.

    weight_granularity:
        "per_tensor"  — one scale for the entire weight tensor
        "per_channel" — one scale per output channel / filter

    Activations are always quantized per-tensor.
    BatchNorm and LayerNorm are left in full precision (standard QAT practice).
    """
    w_quant = (fake_quantize_per_channel if weight_granularity == "per_channel"
               else fake_quantize)

    patched: List[Tuple[nn.Module, object]] = []

    for module in modules:
        for layer in module.modules():
            if isinstance(layer, nn.Linear):
                orig_forward = layer.forward

                def make_linear_q_forward(lyr, wb, ab):
                    def q_forward(x):
                        return F.linear(fake_quantize(x, ab),
                                        w_quant(lyr.weight, wb),
                                        lyr.bias)
                    return q_forward

                layer.forward = make_linear_q_forward(layer, w_bits, a_bits)
                patched.append((layer, orig_forward))

            elif isinstance(layer, nn.Conv2d):
                orig_forward = layer.forward

                def make_conv_q_forward(lyr, wb, ab):
                    def q_forward(x):
                        return F.conv2d(fake_quantize(x, ab),
                                        w_quant(lyr.weight, wb),
                                        lyr.bias, lyr.stride, lyr.padding,
                                        lyr.dilation, lyr.groups)
                    return q_forward

                layer.forward = make_conv_q_forward(layer, w_bits, a_bits)
                patched.append((layer, orig_forward))

    try:
        yield
    finally:
        for layer, orig in patched:
            layer.forward = orig
