"""Reconstruct float weights from a ternary_packed/ sidecar written by ternarize.py --pack."""
from __future__ import annotations

import json
import pathlib

import torch
from safetensors.torch import load_file

from hugo.pure import dequantize_weight, unpack_ternary_2bit


def load_layer_weight(pack_dir: str | pathlib.Path, layer_name: str) -> torch.Tensor:
    pack_dir = pathlib.Path(pack_dir)
    manifest = json.loads((pack_dir / "manifest.json").read_text())
    tensors = load_file(str(pack_dir / "packed.safetensors"))

    entry = manifest["layers"][layer_name]
    shape = entry["shape"]
    packed = tensors[entry["packed_key"]]
    scale = tensors[entry["scale_key"]].view(entry["scale_shape"])

    num_elements = shape[0] * shape[1]
    codes = unpack_ternary_2bit(packed, num_elements).view(shape)
    return dequantize_weight(codes, scale, manifest.get("group_size"))
