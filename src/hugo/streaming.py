"""Disk-bounded, architecture-agnostic ternary quantization for large HF checkpoints.

`quantize.py` walks an already-instantiated `nn.Module`, which requires
`transformers` to know how to build the model class and requires enough
RAM/disk to hold the *entire* model (source + quantized copy) at once. That
breaks down for two reasons that show up in practice with very large or
very new checkpoints:

  1. The full model may not fit in available disk/RAM even once, let alone
     twice (source + output).
  2. Bleeding-edge architectures (hybrid attention/SSM blocks, vision
     towers, multi-token-prediction heads, ...) may not be supported yet by
     the installed `transformers`, or may need `trust_remote_code`.

This module instead works directly on the safetensors shards named in
`model.safetensors.index.json`, one shard at a time: download a shard,
quantize its 2D `*.weight` tensors, write the packed result, delete the
shard, move to the next one. Peak extra disk use is bounded by roughly one
shard's size, not the whole model -- and it never needs to instantiate the
model class, so it works on any architecture as long as weights are plain
linear-layer-shaped (2D) tensors.
"""
from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download, list_repo_files
from huggingface_hub.errors import EntryNotFoundError
from safetensors import safe_open
from safetensors.torch import save_file

from hugo.quantize import (
    LayerQuantStats,
    pack_ternary_2bit,
    quantization_stats,
    ternarize_weight,
)

# Substrings of a tensor name that mean "keep in full precision" even if the
# tensor is 2D. Embedding tables and untied LM heads behave very differently
# from a projection matrix under ternary rounding (every row is looked up
# individually rather than mixed via a matmul), so post-hoc PTQ hurts them
# disproportionately -- standard practice is to leave them unquantized.
DEFAULT_SKIP_SUBSTRINGS = ["embed_tokens", "lm_head", "pos_embed"]

# Files worth copying verbatim into the output dir alongside the quantized
# weights (config/tokenizer/processor -- anything that isn't itself a weight
# shard and is small).
AUX_FILE_SUFFIXES = (
    ".json", ".txt", ".model", ".jinja", ".md",
    # Custom architectures ship their own modeling/config/tokenizer code
    # (modeling_*.py, configuration_*.py, ...) that config.json's auto_map
    # points at. Those are exactly the models this streaming path exists to
    # support -- ones the installed transformers doesn't know natively -- so
    # dropping .py files would produce output that can't be loaded at all.
    # Same trust boundary as `trust_remote_code`: the user already chose to
    # point this tool at the repo.
    ".py",
)
WEIGHT_FILE_SUFFIXES = (".safetensors", ".bin", ".pt", ".gguf", ".h5")


def is_quantizable(name: str, shape, skip_patterns: list[str]) -> bool:
    if len(shape) != 2:
        return False
    if not name.endswith(".weight"):
        return False
    return not any(pattern in name for pattern in skip_patterns)


def resolve_weight_map(repo_id: str, revision: str | None, token) -> dict[str, str]:
    """Return {tensor_name: shard_filename}, handling both sharded and
    single-file safetensors checkpoints."""
    try:
        index_path = hf_hub_download(
            repo_id, "model.safetensors.index.json", revision=revision, token=token
        )
        index = json.loads(Path(index_path).read_text())
        return index["weight_map"]
    except EntryNotFoundError:
        pass  # no index -- this is a single-file checkpoint, not an error

    # Single-file checkpoint: read tensor names straight out of the header.
    single_path = hf_hub_download(repo_id, "model.safetensors", revision=revision, token=token)
    with safe_open(single_path, framework="pt", device="cpu") as f:
        names = list(f.keys())
    return {name: "model.safetensors" for name in names}


def copy_aux_files(repo_id: str, revision: str | None, token, output_dir: Path) -> list[str]:
    """Copy every non-weight file (config, tokenizer, processor, ...) from
    the source repo into output_dir. Returns the list of filenames copied."""
    copied = []
    for filename in list_repo_files(repo_id, revision=revision, token=token):
        if filename.endswith(WEIGHT_FILE_SUFFIXES) or filename.endswith(".index.json"):
            continue
        if not filename.endswith(AUX_FILE_SUFFIXES):
            continue
        # Guard against a repo path escaping output_dir (path traversal via
        # "../" or an absolute path in the listing). Anything that doesn't
        # resolve to somewhere under output_dir is skipped rather than
        # written outside the directory the caller asked us to fill.
        dest = (output_dir / filename).resolve()
        if not dest.is_relative_to(output_dir.resolve()):
            print(f"  skipping {filename!r}: resolves outside the output directory")
            continue

        local_path = hf_hub_download(repo_id, filename, revision=revision, token=token)
        # Nested paths (e.g. a subfolder of custom code) must keep their
        # relative layout, since config.json's auto_map references them by
        # path -- flattening them would break loading.
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(Path(local_path).read_bytes())
        copied.append(filename)
    return copied


@dataclasses.dataclass
class ShardResult:
    shard_name: str
    manifest_entries: dict
    layer_stats: list[LayerQuantStats]
    packed_file: str | None
    plain_file: str | None


def process_shard(
    repo_id: str,
    revision: str | None,
    token,
    shard_name: str,
    tensor_names: list[str],
    work_dir: Path,
    packed_dir: Path,
    plain_dir: Path,
    shard_index: int,
    granularity: str,
    group_size: int | None,
    skip_patterns: list[str],
) -> ShardResult:
    local_path = hf_hub_download(repo_id, shard_name, revision=revision, token=token, local_dir=work_dir)

    packed_tensors: dict[str, torch.Tensor] = {}
    plain_tensors: dict[str, torch.Tensor] = {}
    manifest_entries: dict = {}
    layer_stats: list[LayerQuantStats] = []

    with safe_open(local_path, framework="pt", device="cpu") as f:
        for name in tensor_names:
            tensor = f.get_tensor(name)
            if is_quantizable(name, tensor.shape, skip_patterns):
                codes, scale = ternarize_weight(tensor, granularity=granularity, group_size=group_size)
                packed = pack_ternary_2bit(codes)
                key = name.replace(".", "__")
                packed_tensors[f"{key}.packed"] = packed
                packed_tensors[f"{key}.scale"] = scale.to(torch.float32).contiguous()
                layer_stats.append(quantization_stats(name, tensor, codes, scale, granularity, group_size))
                manifest_entries[name] = {
                    "kind": "quantized",
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype),
                    "packed_key": f"{key}.packed",
                    "scale_key": f"{key}.scale",
                    "scale_shape": list(scale.shape),
                }
            else:
                plain_tensors[name] = tensor.contiguous()
                manifest_entries[name] = {
                    "kind": "plain",
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype),
                }

    packed_file = None
    plain_file = None
    if packed_tensors:
        packed_file = f"ternary_packed/packed_shard_{shard_index:05d}.safetensors"
        save_file(packed_tensors, str(packed_dir / f"packed_shard_{shard_index:05d}.safetensors"))
    if plain_tensors:
        plain_file = f"plain_tensors/plain_shard_{shard_index:05d}.safetensors"
        save_file(plain_tensors, str(plain_dir / f"plain_shard_{shard_index:05d}.safetensors"))

    os.remove(local_path)  # reclaim disk before moving on to the next shard

    return ShardResult(shard_name, manifest_entries, layer_stats, packed_file, plain_file)
