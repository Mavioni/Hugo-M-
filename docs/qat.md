# Quantization-aware training (QAT) and publishing

The PTQ path ([`docs/usage.md`](usage.md)) rounds an already-trained model's
weights to ternary after the fact. It works, but it costs real accuracy —
the measured mean relative L2 weight error on a full run against
`huihui-ai/Huihui-Qwen3.6-27B-abliterated` was **0.53** across all 614 quantized layers,
because those weights were never trained to survive being rounded to
`{-1, 0, +1}`.

QAT fixes the root cause: put the ternary rounding *inside the forward pass
during training*, so gradients push the underlying weights toward values
that round well. This is what BitNet b1.58 does, and it's the difference
between "a model we rounded" and "a model that tolerates rounding."

## How it works

`hugo/qat.py` implements a **straight-through estimator (STE)**. The problem
it solves: `round()` is piecewise constant, so its true gradient is zero
almost everywhere — naive backprop through quantization learns nothing. The
STE forward pass quantizes normally, but the backward pass pretends
quantization was the identity function, clipped to where the clamp didn't
saturate (matching `clamp`'s real gradient there). Gradients therefore reach
the full-precision weight and any ordinary optimizer works unmodified.

Three pieces:

- `ternary_fake_quant(w, granularity, group_size)` — differentiable ternary
  quantization. Uses the *same* `_absmean_scale` as the PTQ path, so
  training and deployment-time rounding agree exactly (verified by
  `test_ternary_fake_quant_matches_ptq_values`).
- `BitLinear` — an `nn.Linear` subclass that fake-quantizes its weight on
  every forward pass. `BitLinear.from_linear()` reuses the *same* Parameter
  objects rather than copies, so converting a pretrained model keeps
  training its actual pretrained weights.
- `convert_to_bitlinear(model, ...)` / `bake_bitlinear_to_linear(model)` —
  swap Linear→BitLinear before training, then bake the final ternary values
  back into plain `nn.Linear` before saving, so the checkpoint is ordinary
  and drop-in loadable.

## Running it

QAT is training, not a one-shot transform — it needs a GPU for any real
model. Smoke-test throughput and loss movement first:

```bash
pip install -e ".[train]"

hugo-train-qat \
    --model yujiepan/qwen2.5-tiny-random \
    --output ./out/tiny-qat \
    --max-steps 20 --limit-samples 200 --batch-size 2 --max-length 64 --lr 1e-3
```

Then the real run on your own hardware:

```bash
hugo-train-qat \
    --model huihui-ai/Huihui-Qwen3.6-27B-abliterated \
    --output ./out/qwen3.6-27b-qat \
    --dataset Salesforce/wikitext --dataset-config wikitext-2-raw-v1 \
    --epochs 1 --batch-size 1 --grad-accum 16 --lr 1e-5 --bf16
```

Key flags: `--granularity` (must match what you'll use at deployment time),
`--skip` (default keeps `lm_head,embed_tokens,norm` full precision —
standard practice, since rounding those hurts disproportionately),
`--grad-accum` to get an effective batch size larger than what fits in
memory, `--max-steps`/`--limit-samples` for smoke tests.

### Compute expectations

A full fine-tune keeps optimizer state for every trainable parameter, so
plan for well beyond the model's own memory footprint — AdamW alone holds
two extra fp32 moments per parameter. For a 27B model that means the run
does not fit on a single consumer GPU, and expect GPU-days rather than
GPU-hours for a meaningful number of tokens. This script does plain
single-process training with gradient accumulation; it deliberately does
not implement FSDP/DeepSpeed sharding, so for multi-GPU work either wrap it
with your existing distributed setup or use `hugo/qat.py`'s
`convert_to_bitlinear` directly inside a training harness you already
trust.

## Publishing to Hugging Face

Authentication comes from the environment, never a CLI flag (so the token
can't leak into shell history, process listings, or CI logs):

```bash
export HF_TOKEN=<write-scoped token>

hugo-push \
    --checkpoint ./out/qwen3.6-27b-qat \
    --repo-id <you>/Hugo-Qwen3.6-27B-ternary \
    --private
```

`--dry-run` previews the generated model card and verifies credentials
without uploading anything. Start `--private`, confirm the checkpoint loads
and the card is accurate, then flip it public.

The model card is generated from the run metadata Hugo wrote at train time,
and it distinguishes QAT from PTQ provenance explicitly. That matters: a
QAT'd model and a naively-rounded PTQ model are byte-identical in structure
but behave very differently, so a PTQ checkpoint's card says so plainly and
reports its measured weight error rather than quietly implying the model was
trained this way.
