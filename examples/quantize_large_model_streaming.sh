#!/usr/bin/env bash
# Quantize a checkpoint too large to fit on local disk/RAM twice, streaming
# one safetensors shard at a time. Safe to interrupt and re-run -- shards
# already recorded as done in manifest.json are skipped.
set -euo pipefail

hugo-stream-ternarize \
    --model "${1:-huihui-ai/Huihui-Qwen3.6-27B-abliterated}" \
    --output "${2:-./out/large-model-ternary}" \
    --granularity channel

# To get a normal, loadable checkpoint back out (on a machine with enough
# disk for the *full* model -- this step is not disk-bounded):
#
#   hugo-reconstruct --input ./out/large-model-ternary --output ./out/large-model-ternary-full
