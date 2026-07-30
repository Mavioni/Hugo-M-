#!/usr/bin/env python3
"""Rebuild a standard, drop-in HF checkpoint from a stream_ternarize.py output dir.

stream_ternarize.py deliberately never materializes a full-size checkpoint
(that's the whole point -- it stays under a small disk budget by keeping
only compact 2-bit-packed weights + a small "plain tensors" sidecar). This
script does the reverse: dequantize everything back into normal-sized
per-shard safetensors files (same shard boundaries as the original model)
plus a matching `model.safetensors.index.json`, so the result loads with
plain `AutoModel*.from_pretrained(...)`.

Run this on a machine with enough disk for the *original* checkpoint size
(the dequantized ternary values are stored as regular fp16/bf16 numbers,
same size as the source model) -- not in the sandbox that produced the
packed output.

Example:
    python3 -m hugo.reconstruct \\
        --input ./out/qwen3.6-27b-ternary \\
        --output ./out/qwen3.6-27b-ternary-full
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from hugo.quantize import dequantize_weight, unpack_ternary_2bit

DTYPE_MAP = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, help="Output dir produced by stream_ternarize.py")
    p.add_argument("--output", required=True, help="Directory to write the reconstructed full checkpoint to")
    p.add_argument("--dtype", choices=list(DTYPE_MAP), default="bfloat16")
    return p.parse_args(argv)


def reconstruct_shard(input_dir: Path, shard_entry: dict, dtype: torch.dtype,
                       group_size: int | None = None) -> dict[str, torch.Tensor]:
    """Rebuild one shard's tensors from its packed + plain sidecars.

    `group_size` must be the value the quantization run used (it lives at the
    top level of manifest.json). Group-granularity scales are 3D and
    `dequantize_weight` needs the group size to reshape the codes back, so
    omitting it makes every grouped checkpoint fail to reconstruct.
    """
    tensors: dict[str, torch.Tensor] = {}

    packed_data = {}
    if shard_entry.get("packed_file"):
        packed_data = load_file(str(input_dir / shard_entry["packed_file"]))
    plain_data = {}
    if shard_entry.get("plain_file"):
        plain_data = load_file(str(input_dir / shard_entry["plain_file"]))

    for name, info in shard_entry["tensors"].items():
        if info["kind"] == "quantized":
            packed = packed_data[info["packed_key"]]
            scale = packed_data[info["scale_key"]].view(info["scale_shape"])
            shape = info["shape"]
            codes = unpack_ternary_2bit(packed, shape[0] * shape[1]).view(shape)
            tensors[name] = dequantize_weight(codes, scale, group_size).to(dtype)
        else:
            tensors[name] = plain_data[name].to(dtype)

    return tensors


def main(argv=None) -> int:
    args = parse_args(argv)
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    dtype = DTYPE_MAP[args.dtype]

    manifest = json.loads((input_dir / "manifest.json").read_text())
    shard_names = sorted(manifest["shards"])
    missing = [s for s, e in manifest["shards"].items() if e.get("status") != "done"]
    if missing:
        raise SystemExit(f"manifest has {len(missing)} unfinished shard(s): {missing[:5]} ... "
                          f"re-run stream_ternarize.py on --input first to finish quantizing")

    weight_map = {}
    total_size = 0
    for i, shard_name in enumerate(shard_names):
        print(f"[{i + 1}/{len(shard_names)}] reconstructing {shard_name} ...")
        tensors = reconstruct_shard(
            input_dir, manifest["shards"][shard_name], dtype, manifest.get("group_size")
        )
        save_file(tensors, str(output_dir / shard_name))
        for name, t in tensors.items():
            weight_map[name] = shard_name
            total_size += t.numel() * t.element_size()

    index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    (output_dir / "model.safetensors.index.json").write_text(json.dumps(index, indent=2))

    print("Copying aux files (config/tokenizer/custom code/...) ...")
    skip_names = {"manifest.json", "ternary_packed", "plain_tensors", "_shard_cache"}
    # Walk recursively rather than only the top level: custom architectures
    # can keep modeling code in subfolders that config.json's auto_map
    # references by relative path, so the layout has to survive the copy.
    for item in input_dir.rglob("*"):
        if not item.is_file():
            continue
        rel = item.relative_to(input_dir)
        if rel.parts[0] in skip_names:
            continue
        dest = output_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item, dest)

    print(f"Done. Reconstructed checkpoint written to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
