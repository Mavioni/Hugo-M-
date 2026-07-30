#!/usr/bin/env bash
# End-to-end: QAT-train a model so it tolerates ternary rounding, then
# publish it to the Hugging Face Hub.
#
# Requires a GPU for any real model. Set HF_TOKEN (write-scoped) in your
# environment before the push step -- never pass it as an argument.
set -euo pipefail

MODEL="${1:-huihui-ai/Huihui-Qwen3.6-27B-abliterated}"
OUT="${2:-./out/qwen3.6-27b-qat}"
REPO_ID="${3:-}"

hugo-train-qat \
    --model "$MODEL" \
    --output "$OUT" \
    --dataset Salesforce/wikitext --dataset-config wikitext-2-raw-v1 \
    --epochs 1 --batch-size 1 --grad-accum 16 --lr 1e-5 --bf16

if [[ -n "$REPO_ID" ]]; then
    # Preview the model card and check credentials before uploading tens of GB.
    hugo-push --checkpoint "$OUT" --repo-id "$REPO_ID" --private --dry-run
    hugo-push --checkpoint "$OUT" --repo-id "$REPO_ID" --private
else
    echo "No repo id given; skipping publish. To publish:"
    echo "  export HF_TOKEN=<write-scoped token>"
    echo "  hugo-push --checkpoint $OUT --repo-id <you>/Hugo-... --private"
fi
