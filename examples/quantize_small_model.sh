#!/usr/bin/env bash
# Quantize a model small enough to load entirely in memory via `transformers`.
# Produces a drop-in HF checkpoint (ternary-valued weights, same dtype/shape
# as the source) plus a genuinely 2-bit-packed sidecar for real compression.
set -euo pipefail

hugo-ternarize \
    --model "${1:-yujiepan/qwen2.5-tiny-random}" \
    --output "${2:-./out/small-model-ternary}" \
    --granularity channel \
    --dtype bfloat16 \
    --pack
