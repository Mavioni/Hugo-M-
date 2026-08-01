# Hugo

<p align="center">
  <a href="https://github.com/Mavioni/Hugo-M-/actions/workflows/tests.yml"><img src="https://github.com/Mavioni/Hugo-M-/actions/workflows/tests.yml/badge.svg" alt="tests"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-AGPL--3.0-blue.svg" alt="license"></a>
  <a href="https://pypi.org/project/hugo/"><img src="https://img.shields.io/badge/python-3.10%2B-blue.svg" alt="python"></a>
  <img src="https://img.shields.io/badge/bitwidth-1.58--bit-8A2BE2.svg" alt="1.58-bit">
  <img src="https://img.shields.io/badge/storage-8%C3%97%20smaller-00C853.svg" alt="8x smaller">
</p>

<p align="center"><strong>Ternary weight quantization for any PyTorch LLM.<br>
1.58 bits per weight · ~8× smaller on disk · drop-in Hugging Face checkpoints.</strong></p>

<div align="center"><pre>
   ┌────────────────────────────┐         ┌────────────────────────────┐
   │  fp16 · 16 bits/weight     │         │  ternary · 1.58 bits       │
   │  [-0.83, 1.45, 0.02]       │  ────▸  │  { -1, +1, 0, -1 } × scale │
   │  [-2.10, ...]              │         │  4 codes packed per byte   │
   └────────────────────────────┘         └────────────────────────────┘
</pre></div>

> **TL;DR** — Hugo converts every `nn.Linear` weight in a Hugging Face model to
> three values — `{-1, 0, +1}` times a scale — the "1.58-bit" representation
> from [BitNet b1.58](https://arxiv.org/abs/2402.17764). It gives you ~8×
> smaller storage now, and a quantization-aware training path that teaches
> weights to *survive* the rounding.

---

## How it works

```
        W  = [-0.83,  1.45,  0.02, -2.10, ...]          fp16 · 16 bits/weight
                        │
                        ▼   scale = mean(|W|) = 1.10    absmean, per channel
                        │
        W/s = [-0.75,  1.32,  0.02, -1.91]
                        │
                        ▼   round(clip(·, -1, +1))      ternary rounding
                        │
         Ŵ/s = [  -1,   +1,    0,   -1  ]                 codes ∈ {-1, 0, +1}
                        │
                        ▼   × scale
                        │
        Ŵ  = [-1.10,  1.10,  0.00, -1.10]               ≈ W · 1.58 bits/weight
```

Modern LLMs store their knowledge as billions of 16-bit weights. Most of that
information survives with far fewer bits: it lives in *which* weights are zero
and *which direction* the rest point. Hugo keeps exactly that — three symbols
per weight instead of 65,536 — and a single scale factor per group brings the
values back to the right magnitude.

> **For practitioners:** BitNet b1.58-style absmean quantization with three
> granularities — `tensor` (one scale for the whole matrix, highest error),
> `channel` (one scale per output row, the default), and `group` (one scale per
> N input elements, lowest error, more metadata). Hugo also genuinely compresses
> codes to **2 bits each** via `--pack` (4 codes per byte), so a 45.7 GB fp16
> model becomes a **5.7 GB** packed sidecar. Realizing the speedup needs a
> ternary-aware runtime like [bitnet.cpp](https://github.com/microsoft/BitNet).

---

## Table of Contents

- [How it works](#how-it-works) — one picture, no prior knowledge needed
- [Two paths](#two-paths) — PTQ vs QAT, and when to pick which
- [Quickstart](#quickstart) — copy-paste to see it work in 60 seconds
- [Paths in detail](#paths-in-detail) — what each path does, step by step
- [Compute requirements](#compute-requirements) — what hardware QAT really needs
- [Installation](#installation)
- [CLI reference](#cli-reference)
- [Repository structure](#repository-structure)
- [Benchmarks](#benchmarks) — measured numbers, not vibes
- [Inference kernel](#inference-kernel) — run the 2-bit weights as-is
- [Caveats](#caveats) — read before you commit to this
- [Contributing](#contributing) · [License](#license)

---

## Two paths

**PTQ — post-training quantization. Round it, ship it.**

```
   ┌────────────────┐      ┌────────────────┐      ┌────────────────┐
   │  HF model      │ ───▸ │  round every   │ ───▸ │  drop-in HF    │
   │  checkpoint    │      │  weight to     │      │  checkpoint    │
   │  (as-is)       │      │  {-1,0,+1}×s   │      │  + 2-bit side  │
   └────────────────┘      └────────────────┘      └────────────────┘
      one pass, CPU           PTQ — no training       save_pretrained()
```

**QAT — quantization-aware training. Train it to survive the rounding.**

```
   ┌────────────────┐      ┌────────────────┐      ┌────────────────┐
   │  HF model      │ ───▸ │  train with    │ ───▸ │  plain model   │
   │  checkpoint    │      │  ternary fake- │      │  checkpoint    │
   │  (as-is)       │      │  quant + STE   │      │  ternary-valued│
   └────────────────┘      └────────────────┘      └────────────────┘
      convert in place        QAT fine-tune           bake + save
```

| | Path | What happens | When to use it |
|---|---|---|---|
| 🔧 | **PTQ** | Round already-trained weights to ternary in one shot | Quick experiments, disk compression, tool evaluation |
| 🏋️ | **QAT** | Train the model *with* ternary rounding inside the forward pass | Production models — weights learn to survive rounding |

---

## Quickstart

### 60-second smoke test (CPU-only, no GPU needed)

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"

# PTQ: round a tiny model's weights to ternary in seconds
hugo-ternarize \
    --model yujiepan/qwen2.5-tiny-random \
    --output ./out/tiny-ternary \
    --granularity channel \
    --pack

# Verify: the packed sidecar is ~8× smaller than fp16
ls -lh ./out/tiny-ternary/ternary_packed/
```

### Run it with the CUDA inference kernel

```bash
pip install -e ".[kernel]"          # Triton (or triton-windows on Windows)
```

```python
from transformers import AutoModelForCausalLM
from hugo.kernel import replace_linears_with_kernel

model = AutoModelForCausalLM.from_pretrained(
    "./out/tiny-ternary", torch_dtype=torch.float16
).to("cuda")

replaced = replace_linears_with_kernel(model, "./out/tiny-ternary/ternary_packed")
# weights now live as 2-bit codes in GPU memory; forward runs the Triton kernel
```

### QAT training smoke test (GPU recommended, ~2 minutes)

```bash
pip install -e ".[train]"

hugo-train-qat \
    --model yujiepan/qwen2.5-tiny-random \
    --output ./out/tiny-qat \
    --dataset Salesforce/wikitext --dataset-config wikitext-2-raw-v1 \
    --max-steps 20 --limit-samples 64 --batch-size 2 --max-length 128
```

The output is a normal Hugging Face checkpoint — load it with `AutoModelForCausalLM.from_pretrained()`.

Chat with the result:

```bash
python scripts/chat.py out/tiny-qat   # interactive REPL
```

### Quantize a model too large to fit on disk twice (streaming)

```bash
hugo-stream-ternarize \
    --model huihui-ai/Huihui-Qwen3.6-27B-abliterated \
    --output ./out/large-model-ternary \
    --granularity channel

# Reconstruct a full checkpoint on a machine with enough disk:
hugo-reconstruct \
    --input ./out/large-model-ternary \
    --output ./out/large-model-ternary-full
```

### Quantize an OpenMythos model

```bash
pip install -e ".[mythos]"
```

```python
from hugo.openmythos import load_mythos_checkpoint, quantize_mythos, MythosQATTrainer

# PTQ: one-shot ternary quantization
model = load_mythos_checkpoint("checkpoints/step_0020000.pt")
stats = quantize_mythos(model)
torch.save(model.state_dict(), "mythos-ternary.pt")

# QAT: train the model to tolerate ternary rounding
model = load_mythos_checkpoint("checkpoints/step_0020000.pt")
trainer = MythosQATTrainer(model)
trainer.train(steps=1000, lr=1e-5)
trainer.bake()
```

---

## Paths in detail

### PTQ: Post-training quantization

Every `nn.Linear` weight is processed in one pass with absmean scaling. The
resulting checkpoint has the **same architecture and dtype** as the source —
only the weight *values* change. Embeddings, normalization layers, and the LM
head are left untouched (standard practice: rounding those hurts
disproportionately).

**Cost:** minutes on CPU. **Quality:** expect measurable degradation — a 27B run
measured **0.53 mean relative L2 weight error** across 614 layers.

Use `--pack` to also produce a genuinely 2-bit-packed sidecar for storage
compression. Use `--dry-run` to see stats without writing anything.

### QAT: Quantization-aware training

QAT puts ternary rounding *inside the forward pass during training*, so
gradients push weights toward values that round well. It uses a **straight-through
estimator (STE)**: the forward pass quantizes (which has zero true gradient), but
the backward pass pretends quantization was the identity function. Standard
optimizers work unmodified.

**Cost:** GPU-days for a 27B+ model. **Quality:** the point of this project —
weights that were *trained* ternary rather than rounded after the fact.

See [`docs/qat.md`](docs/qat.md) for the full QAT workflow and
[`docs/compute-requirements.md`](docs/compute-requirements.md) for hardware sizing.

---

## Compute requirements

### QAT training: what you need

Quantization-aware training keeps **optimizer state** (AdamW moments) for every
trainable parameter in addition to the model weights and gradients. Here is the
memory breakdown for representative model sizes:

| Model size | Params | Weights (bf16) | Gradients (bf16) | Optimizer (fp32×2) | **Total (no activations)** | Recommended GPUs |
|---|---|---|---|---|---|---|
| 1B | 1 × 10⁹ | 2 GB | 2 GB | 8 GB | **~12 GB** | 1× RTX 4090 (24 GB) |
| 7B | 7 × 10⁹ | 14 GB | 14 GB | 56 GB | **~84 GB** | 2× A100 (80 GB) or 1× H100 (80 GB) with CPU offload |
| 13B | 13 × 10⁹ | 26 GB | 26 GB | 104 GB | **~156 GB** | 2× A100 or 2× H100 |
| 27B | 27 × 10⁹ | 54 GB | 54 GB | 216 GB | **~324 GB** | 4× A100 or 4× H100; or 8× A100 with FSDP |
| 70B | 70 × 10⁹ | 140 GB | 140 GB | 560 GB | **~840 GB** | 8× H100 (80 GB) minimum; FSDP/DeepSpeed required |

> **Note:** the current `hugo-train-qat` script does single-process training
> with gradient accumulation — it does not implement FSDP/DeepSpeed sharding.
> For multi-GPU work, either wrap it with your existing distributed setup or use
> `hugo.qat.convert_to_bitlinear` inside a training harness you already trust.

#### Activations add more

The table above excludes **activation memory**, which depends on batch size and
sequence length. Rule of thumb: activations add roughly `batch_size × seq_len ×
hidden_dim × num_layers × 2 bytes` for bf16. With a typical `batch_size=1,
seq_len=512` on a 27B model (~64 hidden dim ≈ 8192, ~48 layers), activations
add ~5–10 GB. Use gradient checkpointing (`torch.utils.checkpoint`) to trade
compute for memory.

#### Training time estimates

On 1× NVIDIA A100 (80 GB) with a 7B model, `batch_size=1, grad_accum=16,
seq_len=512`, expect roughly **4–8 hours per epoch** on a typical dataset
like WikiText-2. For a 27B model on 4× A100, expect **2–4 GPU-days per epoch**
with standard fine-tuning hyperparameters. These are rough estimates — run a
smoke test with `--max-steps 20` first to benchmark your specific hardware.

### PTQ: negligible

Post-training quantization runs on CPU in minutes and needs only enough RAM
for the model (e.g., ~54 GB for a 27B bf16 model). The streaming path
(`hugo-stream-ternarize`) bounds peak disk use to roughly one safetensors shard
(a few GB), so even extremely large checkpoints can be processed on machines
with limited storage.

See [`docs/compute-requirements.md`](docs/compute-requirements.md) for the
complete breakdown with activation math, gradient checkpointing strategies,
and multi-GPU scaling guidance.

---

## Installation

```bash
# CPU-only PyTorch (much smaller download, sufficient for PTQ)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Install Hugo
pip install -e ".[dev]"       # development (tests, linting)
pip install -e ".[train]"     # + QAT training dependencies (datasets)
pip install -e ".[dev,train]" # everything
```

Requires Python 3.10+. See [`pyproject.toml`](pyproject.toml) for the full
dependency list.

---

## CLI reference

```bash
# PTQ — load full model, quantize in memory, save
hugo-ternarize --model <repo-id> --output ./out --granularity channel --pack

# PTQ streaming — process shard by shard (disk-bounded)
hugo-stream-ternarize --model <repo-id> --output ./out --granularity channel

# QAT — train a model to tolerate ternary rounding
hugo-train-qat --model <repo-id> --output ./out --bf16

# Reconstruct — rebuild a full checkpoint from streaming output
hugo-reconstruct --input ./stream-out --output ./full-out

# Push — publish a checkpoint to Hugging Face Hub
export HF_TOKEN=<write-scoped token>
hugo-push --checkpoint ./out --repo-id <you>/Hugo-Model --private
```

See `--help` on each command, [`docs/usage.md`](docs/usage.md), and
[`docs/qat.md`](docs/qat.md) for full flag documentation.

---

## Repository structure

```
src/hugo/
├── __init__.py              # Public API: quantize, qat, streaming, pack/unpack
├── quantize.py              # Core math: absmean scaling, ternary rounding, 2-bit pack/unpack
├── pure.py                  # Pure-quantization math: hashing, Merkle roots, pack helpers
├── qat.py                   # STE, BitLinear layer, convert/bake helpers for QAT training
├── ternarize.py             # CLI: PTQ for models that fit in RAM (hugo-ternarize)
├── stream_ternarize.py      # CLI: PTQ for models larger than disk (hugo-stream-ternarize)
├── streaming.py             # Shard-at-a-time I/O: download, quantize, pack, delete, repeat
├── train_qat.py             # CLI: QAT fine-tuning (hugo-train-qat)
├── reconstruct.py           # CLI: rebuild full checkpoint from streaming output
├── load_packed.py           # Utility: load individual layer weights from packed sidecar
├── openmythos.py            # Bridge: PTQ/QAT for OpenMythos RDT checkpoints
├── push_to_hub.py           # CLI: publish checkpoint to HF Hub with auto-generated model card
└── kernel/                  # Triton inference kernel for 2-bit-packed weights
    ├── __init__.py          # TernaryLinear (drop-in), sidecar loading, replace_linears_with_kernel
    ├── decode.py            # GraphDecoder: CUDA-graph decode engine (static KV buffers)
    ├── fuse.py              # FusedTernary: QKV and gate+up in one kernel call
    └── ternary.py           # Packed-ternary GEMM + GEMV kernels, per-channel scales

tests/                       # pytest suite (no network/GPU required)
├── test_quantize.py         # Core math: ternarize, dequantize, pack/unpack round-trip
├── test_qat.py              # STE gradient correctness, BitLinear conversion, bake
├── test_streaming.py        # Shard processing, weight map resolution, resume logic
├── test_reconstruct.py      # Round-trip: quantize → reconstruct → verify
└── test_push_to_hub.py      # Model card generation, token resolution

docs/
├── qat.md                   # QAT training + publishing workflow
├── usage.md                 # PTQ CLI reference + internals
└── compute-requirements.md  # GPU/memory/time estimates for QAT training

scripts/                     # Standalone dev scripts (require GPU, run from repo root)
├── train_smollm_qat.py      # SmolLM-135M QAT with live metrics → out/<name>/train_log.json
├── benchmark_ternary.py     # PPL / speed / sparsity vs fp32 baseline → out/benchmark.json
├── benchmark_perf.py        # Prefill / decode / TTFT / VRAM microbenchmarks → out/perf_benchmark.json
├── benchmark_kernel.py      # Kernel vs fp16 decode speed + VRAM → out/perf_kernel.json
├── chat.py                  # Interactive chat with a ternarized checkpoint
└── test_harness_on_qat.py   # OpenMythos harness sanity-check on a QAT'd checkpoint

examples/                    # Runnable shell scripts
├── quantize_small_model.sh
├── quantize_large_model_streaming.sh
└── qat_train_and_publish.sh

.github/
└── workflows/tests.yml      # CI: ruff lint + pytest on every push/PR
```

---

## Benchmarks

### PTQ: 27B model, channel granularity

Run against `huihui-ai/Huihui-Qwen3.6-27B-abliterated` (614 linear layers):

```
   fp16    ████████████████████████████████████████████████████  45.7 GB
   packed  ██████                                                5.7 GB
```

| Metric | Value |
|---|---|
| Quantized layers | 614 |
| Weight elements | 22,833,741,824 |
| fp16-equivalent size | 45.67 GB |
| Packed size (2-bit + scales) | 5.71 GB |
| Compression ratio | **8.0×** |
| Mean relative L2 error | 0.53 |
| Mean zero fraction | 0.31 |

### QAT: tiny proxy model (smoke test)

Run on `yujiepan/qwen2.5-tiny-random` (20 steps, lr 1e-3):

| Metric | Value |
|---|---|
| Converted layers | 6 |
| Loss (start → end) | 20.31 → 19.49 |
| Baked layers (≤3 vals/row) | 6/6 verified |
| Reloads cleanly | ✓ |

Both verification tests are in the CI suite — see [`tests/`](tests/).

### QAT: SmolLM-135M on an 8 GB laptop GPU

Run with [`scripts/train_smollm_qat.py`](scripts/train_smollm_qat.py) on
`HuggingFaceTB/SmolLM-135M` (4,000 steps, TinyStories, batch 4, seq 256,
lr 1e-3 → 1e-5, ~11 minutes wall-clock on an RTX 5060 Laptop):

```
   perplexity on held-out TinyStories · lower is better
   fp32 baseline   ██████████████████████  2.05
   QAT ternary     █████████████████▎      1.75
```

| Metric | Value |
|---|---|
| Training loss (start → end) | 15.56 → 1.73 |
| Ternary check (rows with all non-zero weights equal) | 14,976/14,976 ✓ |
| Zero fraction | ~30% of weights exactly 0 |
| Perplexity, held-out TinyStories — fp32 baseline | 2.05 |
| Perplexity, held-out TinyStories — QAT ternary | **1.75** |
| Prefill speed — fp32 baseline / QAT ternary | 8,943 / 9,388 tok/s |
| Decode speed — fp32 baseline / QAT ternary | 48.5 / 47.5 tok/s |
| Peak VRAM (both) | 1.11 GB |

The QAT'd checkpoint rounds to exactly `{-1, 0, +1} × scale` per output
channel — verified directly on the saved weights — and *beats* the fp16
baseline on held-out data. Note the QAT run was also fine-tuned on the eval
domain (TinyStories), so the gap isn't pure quantization gain; the honest
headline is that **quantization cost nothing** while producing a genuinely
2-bit-representable model. Run [`scripts/benchmark_ternary.py`](scripts/benchmark_ternary.py)
and [`scripts/benchmark_perf.py`](scripts/benchmark_perf.py) to reproduce.

---

## Inference kernel

The packed sidecar is not just for storage — a [Triton](https://triton-lang.org)
kernel in `src/hugo/kernel/` consumes it **directly in GPU memory**, skipping
the fp16 materialization entirely. Decode (one token at a time) is
weight-bandwidth-bound, which is exactly where 2-bit weights pay off: the
kernel streams 1/8 the bytes and accumulates in fp32.

```
   weights in fp16            weights as 2-bit codes
   ┌────────────────┐         ┌──────────────────────┐
   │  896 × 896 fp16 │   ──▸   │  896 × 224 u8 codes  │
   │  (1.6 MB)       │         │  + 896 fp32 scales   │
   └────────────────┘         └──────────────────────┘
```

Measured on an RTX 5060 Laptop (8 GB), autoregressive decode, 128 tokens:

| Model | fp16 decode | kernel decode | Speed | Peak VRAM |
|---|---|---|---|---|
| Qwen2.5-0.5B (PTQ) | 51.0 tok/s | 45.5 tok/s | 0.89× | 1.01 GB → **0.38 GB** |
| SmolLM2-1.7B (PTQ) | 58.0 tok/s | **63.1 tok/s** | **1.09×** | 3.46 GB → **0.64 GB** |

The honest read: below ~1B the model is latency-bound and cuBLAS fp16 is hard
to beat — the kernel still wins big on memory (2.7×/5.4× less VRAM). Above
~1B, where every token streams gigabytes of weights, the kernel starts winning
on speed too, and the gap widens with model size.

**Kill the launch overhead with a CUDA graph.** `GraphDecoder` captures the
whole decode step (fixed-size KV caches + position mask, so shapes stay static
and the step is numerically identical) and replays it per token. Fuse the
shared-input projections too — q/k/v and gate/up each get one kernel call
instead of three (`fuse_model_with_kernel`), which also multiplies the GEMV
kernel's parallelism:

```python
from hugo.kernel.decode import GraphDecoder
from hugo.kernel.fuse import fuse_model_with_kernel

fuse_model_with_kernel(model)                  # QKV + gate/up, one call each
dec = GraphDecoder(model, max_len=512)         # works with nn.Linear *and* TernaryLinear
tokens = dec.generate(prompt_ids, max_new_tokens=128)
```

Same machine, in-graph decode:

| Model | fp16 + graph | kernel + graph (fused) | Speed |
|---|---|---|---|
| Qwen2.5-0.5B (PTQ) | 208 tok/s | **233 tok/s** | **1.12×** |
| SmolLM2-1.7B (PTQ) | 80 tok/s | **104 tok/s** | **1.30×** |

Graphs remove the per-call launch cost for *both* paths (fp16 triples from
52 → 208 tok/s at 0.5B), and with the overhead gone the kernel's bandwidth
advantage shows — fusion then pushes it past cuBLAS at every scale measured.

Kernel requirements: channel granularity, CUDA + Triton (falls back to an
exact torch reference elsewhere), activations fp16/bf16.

```python
from hugo.kernel import TernaryLinear, replace_linears_with_kernel

replace_linears_with_kernel(model, "out/tiny-ternary/ternary_packed")  # swap in place
# or build a layer directly from quantize output:
# TernaryLinear.from_linear(linear, codes, scale)
```

Run [`scripts/benchmark_kernel.py`](scripts/benchmark_kernel.py) (add
`--graph` for the CUDA-graph numbers) to reproduce.

---

## Caveats

- **PTQ costs accuracy.** Rounding weights that were never trained to survive
  rounding measured 0.53 mean relative L2 weight error on a 27B run. That's
  expected — it's why the QAT path exists.
- **QAT needs real compute.** Fine-tuning a 27B model takes GPU-days even on
  datacenter hardware. See [Compute requirements](#compute-requirements).
- **Ternary values, ordinary storage.** Checkpoints store ternary values as
  regular fp16/bf16 numbers — they aren't smaller or faster without a ternary-aware
  runtime. Use `--pack` for the genuine 2-bit storage compression, and the
  [inference kernel](#inference-kernel) to run packed weights directly.
- **No inference speedup at small scale.** The kernel needs the model to be
  weight-bandwidth-bound to beat cuBLAS — below ~1B it's a memory win, not a
  speed win (0.98× even in a CUDA graph). The speed gap widens with model
  size: 1.18× at 1.7B in-graph, and more beyond.

---

## Contributing

PRs are welcome. See [`CONTRIBUTING.md`](CONTRIBUTING.md) for setup, workflow,
and design notes. Security issues: [`SECURITY.md`](SECURITY.md).

## License

AGPL-3.0-or-later. See [`LICENSE`](LICENSE).

---

<p align="center">
  <sub>Built with ❤️ by <a href="https://github.com/Mavioni">Mavioni</a>.</sub>
</p>
