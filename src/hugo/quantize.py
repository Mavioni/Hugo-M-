"""BitNet b1.58-style post-training ternary weight quantization.

Reference: Ma et al., "The Era of 1-bit LLMs: All Large Language Models are
in 1.58 Bits" (https://arxiv.org/abs/2402.17764).

Each weight is mapped to {-1, 0, +1} using absmean scaling:

    scale = mean(|W|)                      (per granularity, see below)
    W_ternary = round(clip(W / scale, -1, 1))
    W_dequant = W_ternary * scale

This module is post-training quantization (PTQ), not the quantization-aware
training the BitNet paper uses. PTQ on weights that were never trained to
tolerate ternary rounding loses noticeably more quality than a model trained
ternary from scratch -- expect visible degradation, worse as granularity
gets coarser.
"""
from __future__ import annotations

import dataclasses

import torch
from torch import nn


@dataclasses.dataclass
class LayerQuantStats:
    name: str
    shape: tuple[int, ...]
    granularity: str
    relative_l2_error: float
    zero_fraction: float


def _absmean_scale(weight: torch.Tensor, granularity: str, group_size: int | None) -> torch.Tensor:
    """Compute the absmean scale tensor, broadcastable against `weight`.

    weight is [out_features, in_features]. Returns a scale tensor of shape:
      - "tensor":  scalar (broadcasts to everything)
      - "channel": [out_features, 1] (one scale per output row)
      - "group":   [out_features, in_features // group_size, 1]
    """
    eps = 1e-6
    if granularity == "tensor":
        return weight.abs().mean().clamp_min(eps)
    if granularity == "channel":
        return weight.abs().mean(dim=1, keepdim=True).clamp_min(eps)
    if granularity == "group":
        if group_size is None or group_size <= 0:
            raise ValueError("group granularity requires a positive group_size")
        out_features, in_features = weight.shape
        if in_features % group_size != 0:
            raise ValueError(
                f"in_features={in_features} is not divisible by group_size={group_size}"
            )
        grouped = weight.view(out_features, in_features // group_size, group_size)
        return grouped.abs().mean(dim=2, keepdim=True).clamp_min(eps)
    raise ValueError(f"unknown granularity: {granularity!r}")


def ternarize_weight(
    weight: torch.Tensor,
    granularity: str = "channel",
    group_size: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a 2D weight tensor to ternary values.

    Returns (ternary_codes, scale):
      - ternary_codes: same shape as `weight`, dtype int8, values in {-1, 0, 1}
      - scale: the absmean scale tensor used, dtype float32
    """
    if weight.dim() != 2:
        raise ValueError(f"expected a 2D weight tensor, got shape {tuple(weight.shape)}")

    w = weight.float()
    scale = _absmean_scale(w, granularity, group_size)

    if granularity == "group":
        out_features, in_features = w.shape
        grouped = w.view(out_features, in_features // group_size, group_size)
        codes = (grouped / scale).round().clamp(-1, 1)
        codes = codes.view(out_features, in_features)
    else:
        codes = (w / scale).round().clamp(-1, 1)

    return codes.to(torch.int8), scale.float()


def dequantize_weight(codes: torch.Tensor, scale: torch.Tensor, group_size: int | None = None) -> torch.Tensor:
    """Reconstruct a float weight tensor from ternary codes + scale."""
    codes_f = codes.float()
    if scale.dim() == 3:  # group granularity
        out_features, in_features = codes.shape
        grouped = codes_f.view(out_features, in_features // group_size, group_size)
        return (grouped * scale).view(out_features, in_features)
    return codes_f * scale


def quantization_stats(name: str, original: torch.Tensor, codes: torch.Tensor, scale: torch.Tensor,
                        granularity: str, group_size: int | None = None) -> LayerQuantStats:
    dequant = dequantize_weight(codes, scale, group_size).to(original.dtype)
    diff = (original - dequant).float()
    rel_l2 = torch.linalg.norm(diff) / torch.linalg.norm(original.float()).clamp_min(1e-12)
    zero_fraction = (codes == 0).float().mean().item()
    return LayerQuantStats(
        name=name,
        shape=tuple(original.shape),
        granularity=granularity,
        relative_l2_error=rel_l2.item(),
        zero_fraction=zero_fraction,
    )


def pack_ternary_2bit(codes: torch.Tensor) -> torch.Tensor:
    """Pack ternary codes {-1,0,1} into 2 bits each, 4 values per byte.

    Encoding: -1 -> 0b00, 0 -> 0b01, 1 -> 0b10 (0b11 unused).
    This is the actual storage-compression step: 2 bits/weight instead of
    8 (int8) or 16 (fp16), i.e. the "~1.58 bit" weight representation
    rounded up to a byte-addressable 2 bits/weight in practice.
    """
    flat = codes.reshape(-1).to(torch.uint8)
    encoded = torch.zeros_like(flat)
    encoded[flat == 255] = 0  # -1 stored as 255 in uint8 view -> map to 0b00
    encoded[flat == 0] = 1  # 0 -> 0b01
    encoded[flat == 1] = 2  # 1 -> 0b10

    pad = (-flat.numel()) % 4
    if pad:
        encoded = torch.cat([encoded, torch.zeros(pad, dtype=torch.uint8)])
    encoded = encoded.view(-1, 4)
    packed = (
        (encoded[:, 0] & 0b11)
        | ((encoded[:, 1] & 0b11) << 2)
        | ((encoded[:, 2] & 0b11) << 4)
        | ((encoded[:, 3] & 0b11) << 6)
    )
    return packed.to(torch.uint8)


def unpack_ternary_2bit(packed: torch.Tensor, num_elements: int) -> torch.Tensor:
    """Inverse of pack_ternary_2bit. Returns int8 codes in {-1, 0, 1}."""
    lookup = torch.tensor([-1, 0, 1, 0], dtype=torch.int8)
    out = torch.empty(packed.numel() * 4, dtype=torch.int8)
    for i in range(4):
        bits = (packed >> (2 * i)) & 0b11
        out[i::4] = lookup[bits.long()]
    return out[:num_elements]


def should_skip(module_name: str, skip_patterns: list[str]) -> bool:
    return any(pattern in module_name for pattern in skip_patterns)


def quantize_linear_modules(
    model: nn.Module,
    granularity: str = "channel",
    group_size: int | None = None,
    skip_patterns: list[str] | None = None,
) -> tuple[list[LayerQuantStats], dict[str, tuple[torch.Tensor, torch.Tensor]]]:
    """In-place ternary-quantize every nn.Linear weight in `model`.

    Weights are overwritten with their dequantized (ternary * scale) values,
    so the model keeps its original architecture/dtype and stays loadable
    with the usual `from_pretrained` / `state_dict` machinery -- only the
    weight *values* become ternary. Biases are left untouched.

    Returns (stats, quantized) where `quantized` maps module name to the
    exact (codes, scale) pair used, computed from the *original* (pre-
    quantization) weights. Callers that also need the ternary codes (e.g.
    to build a packed sidecar) must reuse this dict rather than re-deriving
    scale/codes from the already-quantized weights: re-quantizing a
    dequantized ternary tensor shifts the absmean scale (it now excludes the
    weights that got rounded to zero) and silently produces a wrong scale.
    """
    skip_patterns = skip_patterns or []
    stats: list[LayerQuantStats] = []
    quantized: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if should_skip(name, skip_patterns):
            continue

        original = module.weight.data.clone()
        codes, scale = ternarize_weight(original, granularity=granularity, group_size=group_size)
        dequant = dequantize_weight(codes, scale, group_size).to(module.weight.dtype)
        module.weight.data.copy_(dequant)

        stats.append(quantization_stats(name, original, codes, scale, granularity, group_size))
        quantized[name] = (codes, scale)

    return stats, quantized
