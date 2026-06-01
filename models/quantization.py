"""
Quantization utilities with STE (straight-through estimator) and PTQ calibration.

Two weight granularities:
  per_tensor  — one scale for the entire weight matrix (default, coarser)
  per_channel — one scale per output channel/filter (finer, closer to real deployment)

Activations are quantized per-tensor.

Usage — dynamic (STE, for QIT training):
    w_bits, a_bits = sample_bits()
    with quantized_forward([backbone], w_bits, a_bits):
        z = backbone(x)

Usage — calibrated (PTQ inference):
    stats = calibrate_activations(model, calib_loader, n_batches=32, device=device)
    with static_quantized_forward([model], w_bits=4, a_bits=4, calib_stats=stats):
        z = model(x)
"""

import random
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple

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

    Works for Linear [out, in], Conv2d [out, in, kH, kW], and Conv3d [out, in, kD, kH, kW].
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
    Temporarily replace the forward() of all nn.Linear, nn.Conv2d, and nn.Conv3d
    layers inside `modules` so that weights are fake-quantized to w_bits and input
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

            elif isinstance(layer, nn.Conv3d):
                orig_forward = layer.forward

                def make_conv3d_q_forward(lyr, wb, ab):
                    def q_forward(x):
                        return F.conv3d(fake_quantize(x, ab),
                                        w_quant(lyr.weight, wb),
                                        lyr.bias, lyr.stride, lyr.padding,
                                        lyr.dilation, lyr.groups)
                    return q_forward

                layer.forward = make_conv3d_q_forward(layer, w_bits, a_bits)
                patched.append((layer, orig_forward))

    try:
        yield
    finally:
        for layer, orig in patched:
            layer.forward = orig


# ---------------------------------------------------------------------------
# PTQ calibration
# ---------------------------------------------------------------------------

@dataclass
class CalibrationStats:
    """Per-layer activation ranges computed from a calibration dataset.

    Keys are f'{type(layer).__name__}_{id(layer)}' — unique within a model instance.
    Values are (act_min, act_max) floats used to replace per-sample dynamic ranges.
    """
    ranges: Dict[str, Tuple[float, float]] = field(default_factory=dict)


def fake_quantize_static(x: torch.Tensor, n_bits: int,
                          x_min: float, x_max: float) -> torch.Tensor:
    """Fake-quantize using a pre-computed fixed range (inference, no STE needed)."""
    scale = max((x_max - x_min) / (2 ** n_bits - 1), 1e-8)
    return torch.round((x.clamp(x_min, x_max) - x_min) / scale) * scale + x_min


@torch.no_grad()
def calibrate_activations(
    model: nn.Module,
    loader,
    n_batches: int,
    device: torch.device,
    percentile: float = 100.0,
    use_amp: bool = False,
) -> CalibrationStats:
    """Collect per-layer input activation statistics over n_batches of calibration data.

    percentile=100  → min-max calibration (full dynamic range)
    percentile<100  → symmetric percentile clipping (e.g. 99.99 removes the top/bottom 0.005%)
                      Useful when activation outliers inflate the quantization step size.

    Returns a CalibrationStats object for use with static_quantized_forward.
    """
    _MAX_SAMPLES = 8192  # cap per-layer memory usage

    buffers: Dict[str, List[torch.Tensor]] = {}
    hooks = []

    for lyr in model.modules():
        if not isinstance(lyr, (nn.Linear, nn.Conv2d, nn.Conv3d)):
            continue
        key = f"{type(lyr).__name__}_{id(lyr)}"
        buffers[key] = []

        def _make_hook(k: str):
            def _hook(mod: nn.Module, inp: tuple, out: torch.Tensor) -> None:
                x = inp[0].detach().float().flatten()
                if len(x) > _MAX_SAMPLES:
                    idx = torch.randperm(len(x), device=x.device)[:_MAX_SAMPLES]
                    x = x[idx]
                buffers[k].append(x.cpu())
            return _hook

        hooks.append(lyr.register_forward_hook(_make_hook(key)))

    model.eval()
    for i, batch in enumerate(loader):
        if i >= n_batches:
            break
        x = batch[0].to(device)
        if use_amp:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                model(x)
        else:
            model(x)

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


@contextmanager
def static_quantized_forward(
    modules: List[nn.Module],
    w_bits: int,
    a_bits: int,
    calib_stats: Optional[CalibrationStats] = None,
    weight_granularity: WeightGranularity = "per_tensor",
):
    """Like quantized_forward but uses pre-computed activation ranges from calib_stats.

    Falls back to dynamic (per-sample min-max) for any layer not present in calib_stats,
    and for the FP case (calib_stats=None) acts identically to quantized_forward.
    """
    if calib_stats is None:
        with quantized_forward(modules, w_bits, a_bits, weight_granularity):
            yield
        return

    w_quant = (fake_quantize_per_channel if weight_granularity == "per_channel"
               else fake_quantize)
    patched: List[Tuple[nn.Module, object]] = []

    for module in modules:
        for lyr in module.modules():
            key = f"{type(lyr).__name__}_{id(lyr)}"
            rng = calib_stats.ranges.get(key)
            if rng is None:
                continue
            lo, hi = rng

            if isinstance(lyr, nn.Linear):
                orig = lyr.forward
                def _make_linear(l, wb, ab, lo, hi):
                    def _fwd(x):
                        return F.linear(fake_quantize_static(x, ab, lo, hi),
                                        w_quant(l.weight, wb), l.bias)
                    return _fwd
                lyr.forward = _make_linear(lyr, w_bits, a_bits, lo, hi)
                patched.append((lyr, orig))

            elif isinstance(lyr, nn.Conv2d):
                orig = lyr.forward
                def _make_conv(l, wb, ab, lo, hi):
                    def _fwd(x):
                        return F.conv2d(fake_quantize_static(x, ab, lo, hi),
                                        w_quant(l.weight, wb), l.bias,
                                        l.stride, l.padding, l.dilation, l.groups)
                    return _fwd
                lyr.forward = _make_conv(lyr, w_bits, a_bits, lo, hi)
                patched.append((lyr, orig))

            elif isinstance(lyr, nn.Conv3d):
                orig = lyr.forward
                def _make_conv3d(l, wb, ab, lo, hi):
                    def _fwd(x):
                        return F.conv3d(fake_quantize_static(x, ab, lo, hi),
                                        w_quant(l.weight, wb), l.bias,
                                        l.stride, l.padding, l.dilation, l.groups)
                    return _fwd
                lyr.forward = _make_conv3d(lyr, w_bits, a_bits, lo, hi)
                patched.append((lyr, orig))

    try:
        yield
    finally:
        for lyr, orig in patched:
            lyr.forward = orig
