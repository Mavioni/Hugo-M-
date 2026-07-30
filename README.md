# Hugo

![tests](https://github.com/Mavioni/Hugo-M-/actions/workflows/tests.yml/badge.svg)
![license](https://img.shields.io/badge/license-AGPL--3.0-blue.svg)
![python](https://img.shields.io/badge/python-3.10%2B-blue.svg)

Hugo is a toolkit for post-training ternary weight quantization
("1.58-bit", BitNet b1.58-style) of Hugging Face models: every `nn.Linear`
weight is re-tuned down to `{-1, 0, +1} * scale` via absmean scaling, with
tooling to do this even for checkpoints far too large to fit on local disk
or RAM twice over.

```
scale = mean(|W|)                         # per output-channel by default
W_ternary = round(clip(W / scale, -1, 1))
W_dequant = W_ternary * scale
```

Reference: Ma et al., ["The Era of 1-bit LLMs: All Large Language Models
are in 1.58 Bits"](https://arxiv.org/abs/2402.17764).

## Why "Hugo"

This project started as a re-tune of models in the
[`huihui-ai/qwen36-abliterated`](https://huggingface.co/collections/huihui-ai/qwen36-abliterated)
collection down to ternary weights. Hugo is our name for that re-tune
line of work and the tool that produces it.

## Install

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu  # CPU wheel: much smaller/faster
pip install -e ".[dev]"
```

## Quickstart

```bash
# QAT (recommended): train the model to TOLERATE ternary rounding, then publish it.
# Needs a GPU. See docs/qat.md and examples/qat_train_and_publish.sh.
hugo-train-qat --model <hf-repo-id> --output ./out/model-qat --bf16
export HF_TOKEN=<write-scoped token>
hugo-push --checkpoint ./out/model-qat --repo-id <you>/Hugo-... --private

# PTQ: round an already-trained model's weights after the fact (fast, no GPU,
# but costs real accuracy -- 0.53 mean relative L2 error on a real 27B run).
hugo-ternarize --model <hf-repo-id> --output ./out/model-ternary --pack

# PTQ for a model too large to fit on disk twice, streamed shard-by-shard:
hugo-stream-ternarize --model <hf-repo-id> --output ./out/model-ternary
```

Two paths, and the difference matters:

| | What it does | Cost | Quality |
|---|---|---|---|
| **QAT** (`hugo-train-qat`) | Ternary rounding happens *inside the forward pass during training*, so weights learn to round well | GPU-days for a 27B model | The point of this project |
| **PTQ** (`hugo-ternarize`, `hugo-stream-ternarize`) | Rounds already-trained weights after the fact | Minutes, CPU-only | Substantial accuracy hit — weights never had to survive rounding |

Full references: [`docs/qat.md`](docs/qat.md) (training + publishing),
[`docs/usage.md`](docs/usage.md) (PTQ CLIs and internals).

## Repo layout

```
src/hugo/          quantize.py (PTQ math), qat.py (STE + BitLinear for training),
                   train_qat.py (QAT CLI), ternarize.py / stream_ternarize.py (PTQ CLIs),
                   streaming.py (shard I/O), reconstruct.py, load_packed.py,
                   push_to_hub.py (publish + model-card generation)
tests/             pytest suite -- no network access or GPU required
examples/          runnable example scripts for QAT and both PTQ paths
docs/qat.md        QAT training + publishing to Hugging Face
docs/usage.md      PTQ CLI + internals reference
.github/workflows/ CI: lint (ruff) + test (pytest) on every push/PR
```

This layout, the `pyproject.toml` packaging, and the community-health files
(`CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md`, `CHANGELOG.md`)
were modeled on the common structure found across several long-running,
widely-used open-source repositories: `psf/requests`, `pallets/flask`,
`tiangolo/fastapi`, `Textualize/rich`, `huggingface/transformers`,
`ggerganov/llama.cpp`, and `microsoft/BitNet`.

## Honest caveats

- **The PTQ path costs real accuracy.** Rounding weights that were never
  trained to survive rounding measured 0.53 mean relative L2 weight error
  across all 614 quantized layers of a full `Huihui-Qwen3.6-27B-abliterated`
  run (47.00GB fp16-equivalent → 5.87GB packed). That's expected, not a bug —
  it's why the QAT path exists. See
  [`docs/usage.md`](docs/usage.md).
- **QAT needs real compute.** `hugo-train-qat` is validated end-to-end (loss
  decreases, weights bake out genuinely ternary), but a 27B model is
  GPU-days on hardware that fits the model plus optimizer state, and this
  script does single-process training with gradient accumulation rather than
  FSDP/DeepSpeed sharding. See [`docs/qat.md`](docs/qat.md).
- **Ternary values, ordinary storage.** Checkpoints store ternary values as
  regular fp16/bf16 numbers, so they aren't smaller or faster as-is.
  Realizing the ~8x storage win (`--pack`, verified at exactly 8x on the
  27B run) and any speedup needs a ternary-aware runtime such as
  [bitnet.cpp](https://github.com/microsoft/BitNet).

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md). Security issues:
[`SECURITY.md`](SECURITY.md).
