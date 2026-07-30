# hugo usage reference

For the project overview, install instructions, and repo layout, see the
[top-level README](../README.md). This page covers CLI flags and internals.

Post-training ternary weight quantization (BitNet b1.58-style) for Hugging
Face causal LMs. Every `nn.Linear` weight is mapped to `{-1, 0, +1}` via
absmean scaling:

```
scale = mean(|W|)                         # per output-channel by default
W_ternary = round(clip(W / scale, -1, 1))
W_dequant = W_ternary * scale
```

Reference: Ma et al., ["The Era of 1-bit LLMs"](https://arxiv.org/abs/2402.17764).

## What this does and doesn't do

- This is **post-training quantization (PTQ)**, applied to weights that were
  never trained to tolerate ternary rounding. Real BitNet b1.58 models are
  trained from scratch (or fine-tuned) with the ternary constraint in the
  loop; naive PTQ on an off-the-shelf checkpoint loses noticeably more
  quality, and quality degrades further as scale granularity gets coarser
  (`tensor` > `group` > `channel` error, worst to best).
- The saved checkpoint under `--output` is a normal HF model directory
  (same architecture/config/dtype) whose Linear weights just happen to take
  only 3 distinct values per scale group. It loads with plain
  `AutoModelForCausalLM.from_pretrained(...)` and runs on any existing
  stack -- but with **no speed or memory benefit**, since the values are
  still stored as regular fp16/bf16 numbers. Real gains need a ternary-aware
  kernel (e.g. [bitnet.cpp](https://github.com/microsoft/BitNet)).
- Passing `--pack` additionally writes a `ternary_packed/` sidecar with the
  weights genuinely packed at 2 bits each (4 values/byte) plus their scales,
  demonstrating the actual storage compression. `load_packed.py` reconstructs
  individual layer weights from it.

## Usage

```bash
hugo-ternarize \
    --model <hf-repo-id-or-local-path> \
    --output ./out/model-ternary \
    --granularity channel \
    --dtype bfloat16 \
    --pack
```

(`hugo-ternarize` is the console script installed by `pip install -e .`;
`python3 -m hugo.ternarize ...` works identically if you'd rather not rely
on `$PATH`.)

Key flags:
- `--granularity {tensor,channel,group}` -- scale granularity (default
  `channel`, one scale per output row; `tensor` matches the original BitNet
  b1.58 paper exactly but has higher error; `group` needs `--group-size`).
- `--skip` -- comma-separated substrings of module names to leave in full
  precision (default: `lm_head,embed_tokens,norm`).
- `--dry-run` -- quantize in memory and print error/compression stats
  without writing anything, useful for sizing a run before committing disk.
- `--pack` -- also emit the genuinely 2-bit-packed sidecar.

## `stream_ternarize.py`: quantizing checkpoints too big to fit on disk twice

`ternarize.py` above loads the whole model via `transformers`, which needs
enough RAM/disk for the source model *and* the output at once -- fine for
small models, not for the 27B-36B parameter (~50-70GB in bf16) checkpoints
in the `huihui-ai/qwen36-abliterated` collection. Those don't fit in this
sandbox's 30GB disk quota even once.

`stream_ternarize.py` solves that by working directly on the safetensors
shards named in `model.safetensors.index.json`, one at a time: download a
shard, quantize its `*.weight` tensors, write a compact packed sidecar,
delete the shard, move on. Peak extra disk use is roughly one shard's size
(a few GB), not the whole model. It also never instantiates the model
class, so it works regardless of whether the installed `transformers`
supports the architecture yet -- which matters here, since these models use
a bleeding-edge hybrid attention/SSM ("linear_attention") architecture
(`Qwen3_5ForConditionalGeneration`) with a vision tower and a
multi-token-prediction head bolted on, identified by inspecting the real
`config.json` / `model.safetensors.index.json` from
`huihui-ai/Huihui-Qwen3.6-27B-abliterated` (1199 tensors across 15 shards,
55.6GB total).

```bash
hugo-stream-ternarize \
    --model huihui-ai/Huihui-Qwen3.6-27B-abliterated \
    --output ./out/qwen3.6-27b-ternary \
    --granularity channel
```

(see `examples/quantize_large_model_streaming.sh`)

Output layout under `--output`:
```
config.json, tokenizer.*, ...                     # copied verbatim from the source repo
manifest.json                                      # tensor -> {shard, kind, shape, ...} + running stats
ternary_packed/packed_shard_00000.safetensors ...  # 2-bit packed ternary weights + scales
plain_tensors/plain_shard_00000.safetensors ...    # tensors kept full precision (embeddings, norms, biases, ...)
```

Safe to Ctrl-C and re-run: shards already marked `"status": "done"` in
`manifest.json` are skipped, so an interrupted run just picks up where it
left off. `--max-shards N` processes only the first N shards, for a quick
smoke test before committing to a full run.

This was run for real against
`huihui-ai/Huihui-Qwen3.6-27B-abliterated` from this sandbox (streaming
keeps it disk-safe even here) -- see the run's `manifest.json["stats"]` for
the exact measured compression ratio and quantization error on real
weights; `channel`-granularity relative L2 error on the first shard's
Linear layers alone was ~0.51, which is a real, substantial accuracy hit
from doing PTQ with no calibration -- expected, not a bug, and the reason
this is presented as a quantization *tool* rather than a ready-to-serve
model. `--granularity group --group-size 128` trades bigger scale metadata
for lower error (see `test_relative_error_decreases_with_finer_granularity`
in `test_quantize.py` for why finer granularity always reduces error).

To turn the packed output back into a normal, loadable checkpoint (same
shape/shard layout as the source, ternary values stored as regular
fp16/bf16 numbers), run `reconstruct.py` **on a machine with enough disk for
the full model** -- this step is not disk-bounded, unlike quantization:

```bash
hugo-reconstruct \
    --input ./out/qwen3.6-27b-ternary \
    --output ./out/qwen3.6-27b-ternary-full
```

## Tests

```bash
pytest -v
```

Covers the quantize/dequantize math, the 2-bit pack/unpack round-trip,
in-place quantization on a real `nn.Module` (skip patterns, per-channel vs
per-tensor error), the shard-streaming pipeline against a local fake
shard (no network needed), and reconstruction round-tripping back to the
exact dequantized values.
