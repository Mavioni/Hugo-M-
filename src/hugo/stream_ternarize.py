#!/usr/bin/env python3
"""CLI: disk-bounded ternary quantization of a large HF checkpoint.

Unlike ternarize.py (which loads the whole model via `transformers` and
needs enough RAM/disk for source + output at once), this processes one
safetensors shard at a time: download, quantize, write packed output,
delete the shard, repeat. Peak extra disk use is roughly one shard's size,
so this scales to checkpoints far bigger than local disk. It also never
instantiates the model class, so it works on architectures the installed
`transformers` doesn't know about yet.

Output layout under --output:
  config.json, tokenizer.*, ...          (copied verbatim from the source repo)
  manifest.json                          (tensor -> {shard file, kind, shape, ...} + running stats)
  ternary_packed/packed_shard_NNNNN.safetensors   (2-bit packed ternary weights + scales)
  plain_tensors/plain_shard_NNNNN.safetensors     (tensors kept at full precision: embeddings, norms, ...)

Safe to interrupt and re-run: shards already recorded as "done" in
manifest.json are skipped.

Example:
    python3 -m hugo.stream_ternarize \\
        --model huihui-ai/Huihui-Qwen3.6-27B-abliterated \\
        --output ./out/qwen3.6-27b-ternary \\
        --granularity channel
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import shutil
import sys
import time
from pathlib import Path

from hugo.streaming import (
    DEFAULT_SKIP_SUBSTRINGS,
    copy_aux_files,
    process_shard,
    resolve_weight_map,
)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="HF repo id of the source model")
    p.add_argument("--output", required=True, help="Directory to write the quantized checkpoint to")
    p.add_argument("--revision", default=None)
    p.add_argument("--granularity", choices=["tensor", "channel", "group"], default="channel")
    p.add_argument("--group-size", type=int, default=None)
    p.add_argument("--skip", default=",".join(DEFAULT_SKIP_SUBSTRINGS),
                   help=f"Comma-separated substrings of tensor names to keep in full precision "
                        f"(default: {','.join(DEFAULT_SKIP_SUBSTRINGS)})")
    p.add_argument("--token", default=None, help="HF token, for gated/private repos")
    p.add_argument("--work-dir", default=None,
                   help="Scratch dir for the one shard being downloaded at a time "
                        "(default: <output>/_shard_cache, removed on success)")
    p.add_argument("--max-shards", type=int, default=None,
                   help="Only process the first N shards (for smoke-testing on a big model)")
    return p.parse_args(argv)


def human_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.2f}{unit}"
        n /= 1024
    return f"{n:.2f}PB"


def load_manifest(path: Path, model: str, revision, granularity: str, group_size, skip_patterns) -> dict:
    """Load an existing manifest for a resumed run, or start a fresh one.

    Every setting checked here changes the *contents* of what gets written.
    Resuming with any of them changed would skip the already-done shards
    (quantized under the old settings) and process the rest under the new
    ones, silently producing a checkpoint that's internally inconsistent and
    a manifest that misdescribes it. Better to refuse than to hand back a
    quietly-corrupt model.

    Validation raises SystemExit rather than using `assert` because asserts
    are stripped under `python -O`, which would turn this guard into a no-op
    exactly when someone is running optimized.
    """
    if path.exists():
        manifest = json.loads(path.read_text())
        mismatches = []
        for field, existing, current in (
            ("repo_id", manifest.get("repo_id"), model),
            ("revision", manifest.get("revision"), revision),
            ("granularity", manifest.get("granularity"), granularity),
            ("group_size", manifest.get("group_size"), group_size),
            ("skip_patterns", manifest.get("skip_patterns"), skip_patterns),
        ):
            if existing != current:
                mismatches.append(f"  {field}: manifest has {existing!r}, this run wants {current!r}")
        if mismatches:
            raise SystemExit(
                "Refusing to resume: this run's settings differ from the ones the existing\n"
                f"manifest at {path} was built with.\n"
                + "\n".join(mismatches)
                + "\n\nMixing settings across shards would produce an inconsistent checkpoint.\n"
                "Either re-run with the original settings, or use a fresh --output directory."
            )
        return manifest
    return {
        "repo_id": model,
        "revision": revision,
        "granularity": granularity,
        "group_size": group_size,
        "skip_patterns": skip_patterns,
        "shards": {},
        "stats": {},
    }


def aggregate_stats(manifest: dict) -> dict | None:
    """Aggregate quantization stats across *every* completed shard in the
    manifest, not just the ones this process happened to handle.

    A resumed run skips shards an earlier invocation finished, so computing
    totals only from the current process's results silently under-reports
    layer counts, sizes, and error averages. Reading them back out of the
    manifest keeps the reported numbers describing the whole checkpoint.
    """
    layer_stats = [
        s
        for entry in manifest["shards"].values()
        if entry.get("status") == "done"
        for s in entry.get("layer_stats", [])
    ]
    if not layer_stats:
        return None

    total_elements = sum(s["shape"][0] * s["shape"][1] for s in layer_stats)
    worst = max(layer_stats, key=lambda s: s["relative_l2_error"])
    return {
        "num_quantized_layers": len(layer_stats),
        "total_quantized_elements": total_elements,
        "avg_relative_l2_error": sum(s["relative_l2_error"] for s in layer_stats) / len(layer_stats),
        "avg_zero_fraction": sum(s["zero_fraction"] for s in layer_stats) / len(layer_stats),
        "worst_layer": worst["name"],
        "worst_layer_relative_l2_error": worst["relative_l2_error"],
        "fp16_equivalent_bytes": total_elements * 2,
        "packed_bytes": sum((s["shape"][0] * s["shape"][1] + 3) // 4 for s in layer_stats),
    }


def save_manifest(path: Path, manifest: dict) -> None:
    path.write_text(json.dumps(manifest, indent=2))


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.granularity == "group" and not args.group_size:
        print("error: --granularity=group requires --group-size", file=sys.stderr)
        return 2

    skip_patterns = [s for s in args.skip.split(",") if s]
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    packed_dir = output_dir / "ternary_packed"
    plain_dir = output_dir / "plain_tensors"
    packed_dir.mkdir(exist_ok=True)
    plain_dir.mkdir(exist_ok=True)
    work_dir = Path(args.work_dir) if args.work_dir else output_dir / "_shard_cache"
    work_dir.mkdir(exist_ok=True)

    manifest_path = output_dir / "manifest.json"
    manifest = load_manifest(manifest_path, args.model, args.revision, args.granularity, args.group_size, skip_patterns)

    print(f"Copying config/tokenizer files from {args.model} ...")
    copied = copy_aux_files(args.model, args.revision, args.token, output_dir)
    print(f"  copied {len(copied)} files: {copied}")

    print(f"Resolving weight map for {args.model} ...")
    weight_map = resolve_weight_map(args.model, args.revision, args.token)
    shard_to_names: dict[str, list[str]] = {}
    for name, shard in weight_map.items():
        shard_to_names.setdefault(shard, []).append(name)
    shard_names = sorted(shard_to_names)
    if args.max_shards:
        shard_names = shard_names[: args.max_shards]

    for shard_index, shard_name in enumerate(shard_names):
        entry = manifest["shards"].get(shard_name)
        if entry and entry.get("status") == "done":
            print(f"[{shard_index + 1}/{len(shard_names)}] {shard_name}: already done, skipping")
            continue

        t0 = time.time()
        print(f"[{shard_index + 1}/{len(shard_names)}] {shard_name}: downloading + quantizing "
              f"({len(shard_to_names[shard_name])} tensors) ...")
        result = process_shard(
            repo_id=args.model,
            revision=args.revision,
            token=args.token,
            shard_name=shard_name,
            tensor_names=shard_to_names[shard_name],
            work_dir=work_dir,
            packed_dir=packed_dir,
            plain_dir=plain_dir,
            shard_index=shard_index,
            granularity=args.granularity,
            group_size=args.group_size,
            skip_patterns=skip_patterns,
        )
        dt = time.time() - t0

        manifest["shards"][shard_name] = {
            "status": "done",
            "packed_file": result.packed_file,
            "plain_file": result.plain_file,
            "tensors": result.manifest_entries,
            # Persist this shard's stats so a later resumed run can aggregate
            # across shards it didn't process itself (see aggregate_stats).
            "layer_stats": [dataclasses.asdict(s) for s in result.layer_stats],
        }
        save_manifest(manifest_path, manifest)  # persist after every shard so a crash loses at most one shard

        print(f"    done in {dt:.1f}s, quantized {len(result.layer_stats)} layers this shard")

    stats = aggregate_stats(manifest)
    if stats:
        manifest["stats"] = stats
        save_manifest(manifest_path, manifest)

        print("\n--- summary (all completed shards, including any from earlier runs) ---")
        print(f"  quantized layers          : {stats['num_quantized_layers']}")
        print(f"  quantized weight elements : {stats['total_quantized_elements']:,}")
        print(f"  avg relative L2 error     : {stats['avg_relative_l2_error']:.4f}")
        print(f"  avg zero fraction         : {stats['avg_zero_fraction']:.4f}")
        print(f"  worst layer               : {stats['worst_layer']} "
              f"(rel. L2 error {stats['worst_layer_relative_l2_error']:.4f})")
        print(f"  fp16-equivalent size      : {human_bytes(stats['fp16_equivalent_bytes'])}")
        print(f"  packed size (2-bit)       : {human_bytes(stats['packed_bytes'])}"
              f"  (~{stats['fp16_equivalent_bytes'] / max(stats['packed_bytes'], 1):.1f}x smaller)")

    if not any(e.get("status") != "done" for e in manifest["shards"].values()) and len(manifest["shards"]) == len(shard_to_names):
        shutil.rmtree(work_dir, ignore_errors=True)
        print("\nAll shards processed; removed shard scratch cache.")
    else:
        print(f"\n{sum(1 for e in manifest['shards'].values() if e.get('status') == 'done')}/{len(shard_to_names)} "
              f"shards done overall. Re-run the same command to continue.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
