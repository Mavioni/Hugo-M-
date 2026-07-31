#!/usr/bin/env python3
"""CLI: ternary-quantize (BitNet b1.58-style) a Hugging Face causal LM.

Example:
    python3 -m hugo.ternarize \\
        --model huihui-ai/Huihui-Qwen3.6-27B-abliterated \\
        --output ./out/qwen3.6-27b-ternary \\
        --granularity channel \\
        --pack

Notes:
  - This is post-training quantization applied to weights that were never
    trained to be ternary. Expect a real quality hit; it will not match a
    model trained from scratch with ternary-aware training.
  - The saved model in --output is a normal HF checkpoint (same architecture,
    same dtype) whose Linear weights just happen to take only 3 distinct
    values per scale group -- it loads with plain `from_pretrained` and gets
    no speedup without a ternary-aware kernel (e.g. bitnet.cpp). Pass --pack
    to additionally emit a genuinely 2-bit-per-weight packed sidecar that
    demonstrates the real storage compression.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from collections.abc import Callable

import torch
from torch import nn

from hugo.pure import (
    LayerQuantStats,
    build_shard_integrity_hash,
    hash_packed_layer,
    merkle_root,
    pack_ternary_2bit,
)
from hugo.quantize import (
    quantize_linear_modules,
)

DEFAULT_SKIP = ["lm_head", "embed_tokens", "norm"]

DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def _load_model_and_tokenizer(
    model_id: str, revision: str | None, dtype: torch.dtype, trust_remote_code: bool
) -> tuple[nn.Module, object]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        dtype=dtype,
        trust_remote_code=trust_remote_code,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, revision=revision, trust_remote_code=trust_remote_code
    )
    return model, tokenizer


def _save_checkpoint(model: nn.Module, tokenizer: object, out_dir: pathlib.Path,
                     max_shard_size: str) -> None:
    model.save_pretrained(out_dir, max_shard_size=max_shard_size, safe_serialization=True)
    tokenizer.save_pretrained(out_dir)


def _save_packed_sidecar(packed_tensors: dict, manifest: dict, pack_dir: pathlib.Path) -> None:
    from safetensors.torch import save_file

    saved_path = pack_dir / "packed.safetensors"
    save_file(packed_tensors, str(saved_path))

    integrity_hash = build_shard_integrity_hash(packed_tensors)
    manifest["sha256"] = integrity_hash
    manifest["merkle_root"] = integrity_hash

    (pack_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="HF repo id or local path of the source model")
    p.add_argument("--output", required=True, help="Directory to write the quantized model + tokenizer to")
    p.add_argument("--revision", default=None, help="Optional HF revision/branch/commit for --model")
    p.add_argument("--granularity", choices=["tensor", "channel", "group"], default="channel",
                   help="Scale granularity for absmean quantization (default: channel, i.e. per output row)")
    p.add_argument("--group-size", type=int, default=None,
                   help="Group size along the input dimension, required when --granularity=group")
    p.add_argument("--skip", default=",".join(DEFAULT_SKIP),
                   help=f"Comma-separated substrings of module names to leave unquantized "
                        f"(default: {','.join(DEFAULT_SKIP)})")
    p.add_argument("--dtype", choices=list(DTYPE_MAP), default="bfloat16",
                   help="Compute/storage dtype to load the model in (default: bfloat16)")
    p.add_argument("--pack", action="store_true",
                   help="Also write a 2-bit-packed ternary sidecar under <output>/ternary_packed/")
    p.add_argument("--dry-run", action="store_true",
                   help="Only load, quantize in memory, and report stats -- do not save anything")
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--max-shard-size", default="4GB")
    return p.parse_args(argv)


def human_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.2f}{unit}"
        n /= 1024
    return f"{n:.2f}PB"


def summarize(stats: list[LayerQuantStats]) -> None:
    if not stats:
        print("No nn.Linear layers were quantized (check --skip patterns).")
        return

    total_params = sum(s.shape[0] * s.shape[1] for s in stats)
    avg_err = sum(s.relative_l2_error for s in stats) / len(stats)
    avg_zero = sum(s.zero_fraction for s in stats) / len(stats)
    worst = max(stats, key=lambda s: s.relative_l2_error)

    print(f"Quantized {len(stats)} Linear layers, {total_params:,} weight elements")
    print(f"  avg relative L2 error : {avg_err:.4f}")
    print(f"  avg zero fraction     : {avg_zero:.4f}  (share of weights rounded to 0)")
    print(
        f"  worst layer           : {worst.name} "
        f"(rel. L2 error {worst.relative_l2_error:.4f}, shape {worst.shape})"
    )

    fp16_bytes = total_params * 2
    packed_bytes = sum((s.shape[0] * s.shape[1] + 3) // 4 for s in stats)
    scale_bytes_channel = sum(s.shape[0] * 4 for s in stats)  # float32 scale per output row
    print(f"  fp16 size of quantized layers      : {human_bytes(fp16_bytes)}")
    print(f"  2-bit packed + per-channel scales  : {human_bytes(packed_bytes + scale_bytes_channel)}"
          f"  (~{fp16_bytes / max(packed_bytes + scale_bytes_channel, 1):.1f}x smaller)")


def build_packed_sidecar(model, quantized: dict, granularity: str, group_size: int | None):
    """Build the packed sidecar from the (codes, scale) pairs computed during
    the original quantization pass -- NOT by re-deriving them from the
    already-quantized model weights (see quantize_linear_modules docstring).

    Each layer's packed codes are individually SHA-256 hashed, and a Merkle
    root is computed over all layers for content-integrity verification.
    """
    packed_tensors = {}
    layer_hashes: list[str] = []
    manifest: dict = {"granularity": granularity, "group_size": group_size, "layers": {}}
    shapes = {name: module.weight.shape for name, module in model.named_modules() if name in quantized}

    for name, (codes, scale) in quantized.items():
        packed = pack_ternary_2bit(codes)
        key = name.replace(".", "__")
        packed_tensors[f"{key}.packed"] = packed
        scale_contig = scale.to(torch.float32).contiguous()
        packed_tensors[f"{key}.scale"] = scale_contig
        layer_hash = hash_packed_layer(packed, scale_contig)
        layer_hashes.append(layer_hash)
        manifest["layers"][name] = {
            "shape": list(shapes[name]),
            "packed_key": f"{key}.packed",
            "scale_key": f"{key}.scale",
            "scale_shape": list(scale.shape),
            "sha256": layer_hash,
        }

    manifest["merkle_root"] = merkle_root(layer_hashes)
    return packed_tensors, manifest


def main(
    argv: list[str] | None = None,
    *,
    _load_fn: Callable = _load_model_and_tokenizer,
    _save_fn: Callable = _save_checkpoint,
    _pack_save_fn: Callable = _save_packed_sidecar,
) -> int:
    args = parse_args(argv)

    if args.granularity == "group" and not args.group_size:
        print("error: --granularity=group requires --group-size", file=sys.stderr)
        return 2

    print(f"Loading {args.model!r} (dtype={args.dtype}) ...")
    model, tokenizer = _load_fn(
        args.model, args.revision, DTYPE_MAP[args.dtype], args.trust_remote_code
    )

    skip_patterns = [s for s in args.skip.split(",") if s]
    print(f"Quantizing Linear layers (granularity={args.granularity}, skip={skip_patterns}) ...")
    stats, quantized = quantize_linear_modules(
        model, granularity=args.granularity, group_size=args.group_size, skip_patterns=skip_patterns
    )
    summarize(stats)

    if args.dry_run:
        print("Dry run: not saving anything.")
        return 0

    out_dir = pathlib.Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Saving drop-in HF checkpoint (ternary-valued weights) to {out_dir} ...")
    _save_fn(model, tokenizer, out_dir, args.max_shard_size)

    stats_path = out_dir / "hugo_stats.json"
    stats_path.write_text(json.dumps([dataclasses_asdict(s) for s in stats], indent=2))
    print(f"Wrote per-layer quantization stats to {stats_path}")

    if args.pack:
        pack_dir = out_dir / "ternary_packed"
        pack_dir.mkdir(exist_ok=True)
        print(f"Building 2-bit packed sidecar under {pack_dir} ...")
        packed_tensors, manifest = build_packed_sidecar(
            model, quantized, args.granularity, args.group_size
        )
        _pack_save_fn(packed_tensors, manifest, pack_dir)
        print(f"Packed sidecar written ({len(packed_tensors) // 2} layers).")

    print("Done.")
    return 0


def dataclasses_asdict(stats: LayerQuantStats) -> dict:
    import dataclasses

    return dataclasses.asdict(stats)


if __name__ == "__main__":
    raise SystemExit(main())
