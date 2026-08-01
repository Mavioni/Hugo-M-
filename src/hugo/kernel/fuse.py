"""Fused packed-ternary projections: QKV and gate+up in one kernel call.

q/k/v (and gate/up) all consume the *same* activation vector, so their
packed weights can be stacked along the output dimension and projected with
a single kernel launch. For the GEMV kernel this multiplies the N dimension
(and therefore the number of parallel programs) by 2-3x and streams each
packed K-block once instead of once per projection. Per transformer layer:
7 kernel calls -> 4 (qkv, o, gate_up, down).
"""
from __future__ import annotations

import torch
from torch import nn

from hugo.kernel import TernaryLinear
from hugo.kernel.ternary import ternary_matmul


class FusedTernary(nn.Module):
    """Stacked ``TernaryLinear`` projections sharing one input activation.

    ``packed``/``scale`` are concatenated along the output (N) dimension;
    ``forward`` returns a tuple of the per-projection outputs, which is
    capturable in a CUDA graph.
    """

    def __init__(
        self, linears: list[TernaryLinear], sizes: list[int], in_features: int
    ):
        super().__init__()
        if not linears or not all(isinstance(ln, TernaryLinear) for ln in linears):
            raise TypeError("FusedTernary requires TernaryLinear projections")
        if len(sizes) != len(linears) or sum(sizes) != sum(ln.out_features for ln in linears):
            raise ValueError("sizes must match the per-projection output widths")
        device = linears[0].packed.device
        self.sizes = list(sizes)
        self.in_features = in_features
        self.out_features = sum(self.sizes)
        self.register_buffer(
            "packed", torch.cat([ln.packed for ln in linears], dim=0).contiguous()
        )
        self.register_buffer(
            "scale", torch.cat([ln.scale for ln in linears], dim=0).contiguous()
        )
        biases = [ln.bias for ln in linears if ln.bias is not None]
        if biases:
            self.register_buffer("bias", torch.cat(biases).contiguous())
        else:
            self.register_buffer("bias", None)
        if device.type == "cuda":
            self.to(device)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        leading = x.shape[:-1]
        y = ternary_matmul(
            x.reshape(-1, self.in_features), self.packed, self.scale,
            self.out_features, self.in_features,
        )
        if leading:
            y = y.reshape(*leading, self.out_features)
        if self.bias is not None:
            y = y + self.bias
        return tuple(y.split(self.sizes, dim=-1))


def _fuse_projection(
    parent: nn.Module, names: list[str], attr: str
) -> bool:
    projections = [getattr(parent, n, None) for n in names]
    if any(not isinstance(p, TernaryLinear) for p in projections):
        return False
    linears = [p for p in projections if p is not None]  # type: ignore[assignment]
    fused = FusedTernary(
        linears, [ln.out_features for ln in linears], linears[0].in_features
    )
    setattr(parent, attr, fused)
    for name in names:
        if hasattr(parent, name):
            delattr(parent, name)
    return True


def fuse_attention_qkv(layer: nn.Module) -> bool:
    """Fuse self_attn q/k/v into a single ``attn.qkv`` module."""
    return _fuse_projection(layer.self_attn, ["q_proj", "k_proj", "v_proj"], "qkv")


def fuse_mlp_gate_up(layer: nn.Module) -> bool:
    """Fuse mlp gate/up into a single ``mlp.gate_up`` module."""
    return _fuse_projection(layer.mlp, ["gate_proj", "up_proj"], "gate_up")


def fuse_model_with_kernel(model: nn.Module) -> tuple[int, int]:
    """Fuse QKV and gate+up across every decoder layer (kernel-backed model).

    Prerequisite: the model's linears were already swapped for
    ``TernaryLinear`` (``replace_linears_with_kernel``). Returns the number
    of QKV and MLP groups fused.
    """
    n_qkv = n_mlp = 0
    for layer in model.model.layers:
        n_qkv += fuse_attention_qkv(layer)
        n_mlp += fuse_mlp_gate_up(layer)
    return n_qkv, n_mlp
