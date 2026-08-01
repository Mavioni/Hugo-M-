"""Kernel-backed ternary inference: TernaryLinear + sidecar loading.

The Triton kernel in ``hugo.kernel.ternary`` turns 2-bit-packed ternary weights
into real inference speedups for memory-bound decode. These helpers make it
drop-in: replace any ``nn.Linear`` with a ``TernaryLinear`` backed by the
``ternary_packed/`` sidecar, or build one directly from quantize output.
"""
from __future__ import annotations

import json
import pathlib

import torch
from torch import nn

from hugo.kernel.ternary import pack_ternary_2bit_rows, ternary_matmul


class TernaryLinear(nn.Module):
    """Drop-in ``nn.Linear`` replacement backed by packed ternary weights.

    Stores the 2-bit codes + per-channel scales instead of a full-weight
    matrix. Uses the Triton kernel on CUDA; falls back to an exact torch
    reference path elsewhere.
    """

    def __init__(
        self,
        packed: torch.Tensor,
        scale: torch.Tensor,
        in_features: int,
        out_features: int,
        bias: torch.Tensor | None = None,
    ):
        super().__init__()
        packed = packed.detach()
        scale = scale.detach().reshape(-1)
        expected_numel = out_features * ((in_features + 3) // 4)
        if packed.numel() != expected_numel:
            raise ValueError(
                f"packed numel {packed.numel()} != expected {expected_numel} "
                f"for (out={out_features}, in={in_features})"
            )
        if packed.dim() == 2 and packed.shape != (out_features, (in_features + 3) // 4):
            raise ValueError(
                f"packed shape {tuple(packed.shape)} incompatible with "
                f"(out={out_features}, in={in_features})"
            )
        if scale.shape != (out_features,):
            raise ValueError(
                f"scale shape {tuple(scale.shape)} != ({out_features},); "
                "kernel supports per-channel granularity only"
            )
        self._in_features = in_features
        self._out_features = out_features
        self.register_buffer(
            "packed", packed.reshape(out_features, (in_features + 3) // 4).contiguous()
        )
        self.register_buffer("scale", scale)
        if bias is not None:
            self.register_buffer("bias", bias.detach())
        else:
            self.register_buffer("bias", None)

    @classmethod
    def from_linear(
        cls, linear: nn.Linear, codes: torch.Tensor, scale: torch.Tensor
    ) -> TernaryLinear:
        """Build from quantize output (int8 codes in {-1, 0, 1} + scales)."""
        packed = pack_ternary_2bit_rows(codes)
        bias = linear.bias.detach() if linear.bias is not None else None
        return cls(
            packed,
            scale,
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=bias,
        )

    @property
    def out_features(self) -> int:
        return self._out_features

    @property
    def in_features(self) -> int:
        return self._in_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        leading = x.shape[:-1]
        y = ternary_matmul(
            x.reshape(-1, self.in_features), self.packed, self.scale,
            self.out_features, self.in_features,
        )
        if leading:
            y = y.reshape(*leading, self.out_features)
        if self.bias is not None:
            y = y + self.bias.to(y.dtype)
        return y


def load_layer_packed(
    pack_dir: str | pathlib.Path, layer_name: str
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    """Load (packed, scale, shape) for one layer from a ``ternary_packed/`` sidecar."""
    from safetensors.torch import load_file

    pack_dir = pathlib.Path(pack_dir)
    manifest = json.loads((pack_dir / "manifest.json").read_text())
    if manifest.get("granularity", "channel") != "channel":
        raise ValueError(
            f"kernel requires channel granularity, sidecar has {manifest.get('granularity')}"
        )
    tensors = load_file(str(pack_dir / "packed.safetensors"))
    entry = manifest["layers"][layer_name]
    packed = tensors[entry["packed_key"]]
    scale = tensors[entry["scale_key"]].view(entry["scale_shape"]).reshape(-1)
    return packed, scale, list(entry["shape"])


def replace_linears_with_kernel(model: nn.Module, pack_dir: str | pathlib.Path) -> int:
    """Swap every sidecar-covered ``nn.Linear`` in ``model`` for a ``TernaryLinear``.

    Model must already be loaded (e.g. the baked checkpoint produced by the
    same quantize run) so biases are preserved. Returns the number of replaced
    layers.
    """
    pack_dir = pathlib.Path(pack_dir)
    manifest = json.loads((pack_dir / "manifest.json").read_text())
    replaced = 0
    for name in manifest["layers"]:
        packed, scale, shape = load_layer_packed(pack_dir, name)
        parts = name.split(".")
        parent: nn.Module = model
        for part in parts[:-1]:
            parent = parent.get_submodule(part)
        linear = getattr(parent, parts[-1])
        if not isinstance(linear, nn.Linear):
            raise TypeError(f"{name} is {type(linear).__name__}, expected nn.Linear")
        kernel_linear = TernaryLinear(
            packed,
            scale,
            bias=linear.bias.detach() if linear.bias is not None else None,
            in_features=shape[1],
            out_features=shape[0],
        )
        if linear.weight.is_cuda:
            kernel_linear = kernel_linear.to(linear.weight.device)
        setattr(parent, parts[-1], kernel_linear)
        replaced += 1
    return replaced
