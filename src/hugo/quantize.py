"""BitNet b1.58-style post-training ternary weight quantization.

Reference: Ma et al., "The Era of 1-bit LLMs: All Large Language Models are
in 1.58 Bits" (https://arxiv.org/abs/2402.17764).

Each weight is mapped to {-1, 0, +1} using absmean scaling:

    scale = mean(|W|)                      (per granularity, see below)
    W_ternary = round(clip(W / scale, -1, 1))
    W_dequant = W_ternary * scale

All pure mathematics (scale computation, rounding, packing, hashing) lives in
``hugo.pure``. This module contains only the impurity that touches
``nn.Module`` — walking the model tree and mutating Linear weights in place.
"""
from __future__ import annotations

import torch
from torch import nn

from hugo.pure import (
    LayerQuantStats,
    _absmean_scale,  # noqa: F401 — re-exported
    dequantize_weight,  # noqa: F401 — re-exported
    pack_ternary_2bit,  # noqa: F401 — re-exported
    quantization_stats,  # noqa: F401 — re-exported
    should_skip,  # noqa: F401 — re-exported
    ternarize_weight,  # noqa: F401 — re-exported
    unpack_ternary_2bit,  # noqa: F401 — re-exported
)


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
