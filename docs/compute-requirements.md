# Compute requirements for Hugo QAT training

This document provides a detailed breakdown of the GPU memory, compute time,
and hardware needed to fine-tune a model with Hugo's quantization-aware
training (QAT). For the PTQ paths (`hugo-ternarize` /
`hugo-stream-ternarize`), see the [usage reference](usage.md) — those run
on CPU in minutes with modest RAM.

---

## Memory breakdown: what lives on GPU during QAT

During QAT fine-tuning, the following tensors must reside in GPU memory
(or be offloaded to CPU at a performance cost):

### 1. Model weights (bf16)

Every parameter that `model.parameters()` yields. For `hugo-train-qat`,
all `nn.Linear` weights are trainable plus any biases, layer norms,
embeddings, and the LM head. Hugging Face LLaMA/Qwen-style models have
roughly one linear weight per attention projection (Q, K, V, O) plus two
per MLP (gate+up, down) per layer.

```
weight_bytes = num_params × 2    (bf16 = 2 bytes/param)
```

| Model | Parameters | Weights (bf16) |
|---|---|---|
| 1B | ~1.0 × 10⁹ | 2.0 GB |
| 7B | ~7.0 × 10⁹ | 14.0 GB |
| 13B | ~13.0 × 10⁹ | 26.0 GB |
| 27B | ~27.0 × 10⁹ | 54.0 GB |
| 70B | ~70.0 × 10⁹ | 140.0 GB |

### 2. Gradients (bf16)

One gradient tensor per trainable parameter, same shape as the weight.
In PyTorch, `.grad` tensors are the same dtype as the parameter by default.

```
gradient_bytes ≈ weight_bytes   (slightly less — biases and norms are small)
```

### 3. Optimizer states (AdamW, fp32)

AdamW maintains two running moments per parameter, both in **float32**
(4 bytes each):

- **First moment (m):** exponential moving average of gradients
- **Second moment (v):** exponential moving average of squared gradients

```
optimizer_bytes = num_params × 8    (2 moments × 4 bytes each in fp32)
```

This is the dominant memory cost — **4× the weights** in bf16-equivalent
terms, or **8 bytes per parameter** absolute.

### 4. Activations

Intermediate tensors stored during the forward pass so gradients can be
computed during the backward pass. The rough formula:

```
activation_bytes ≈ batch_size × seq_len × hidden_dim × num_layers × 2 bytes
```

For a 27B Qwen-style model (hidden_dim ≈ 8192, ~48 layers):

| batch_size × seq_len | Activation memory (approx) |
|---|---|
| 1 × 256 | 2.0 GB |
| 1 × 512 | 4.0 GB |
| 1 × 1024 | 8.0 GB |
| 4 × 512 | 16.0 GB |
| 1 × 2048 | 16.0 GB |

These are approximate — the real number depends on the specific
architecture, whether flash attention is used, and whether the model
has multi-token-prediction heads or vision towers.

### 5. Total memory budget

| Model | Weights (bf16) | Gradients (bf16) | Optimizer (fp32) | Activations (1×512) | **Total** | **Min GPUs (80 GB each)** |
|---|---|---|---|---|---|---|
| 1B | 2 GB | 2 GB | 8 GB | ~0.5 GB | **~12.5 GB** | 1× RTX 4090 |
| 7B | 14 GB | 14 GB | 56 GB | ~2 GB | **~86 GB** | 2× A100 or 1× H100 with offload |
| 13B | 26 GB | 26 GB | 104 GB | ~4 GB | **~160 GB** | 2× A100 or 2× H100 |
| 27B | 54 GB | 54 GB | 216 GB | ~8 GB | **~332 GB** | 4× A100 or 4× H100 |
| 70B | 140 GB | 140 GB | 560 GB | ~16 GB | **~856 GB** | 8× H100 (FSDP required) |

**Key takeaway:** The optimizer state dominates — it's 4× larger than
the model weights in bf16. For a 27B model, you need at least ~332 GB
of aggregate GPU memory for batch_size=1. The current `hugo-train-qat`
script does single-process training with gradient accumulation; for
multi-GPU setups, wrap it with FSDP (Fully Sharded Data Parallel) or
DeepSpeed ZeRO to shard the optimizer state across devices.

---

## Memory optimization strategies

### Gradient accumulation

The `--grad-accum` flag simulates a larger batch size without increasing
activation memory:

```bash
hugo-train-qat --batch-size 1 --grad-accum 16 ...
# Effective batch size = 16, but activations only for batch_size=1
```

The optimizer steps every `grad_accum` micro-batches, so the effective
batch size is `batch_size × grad_accum`.

### Gradient checkpointing (activation recomputation)

Add this to `hugo-train-qat` (or wrap in your own training loop) to trade
compute for memory:

```python
model.gradient_checkpointing_enable()
```

This reduces activation memory by ~70–80% by recomputing intermediate
activations during the backward pass instead of storing them. On a 27B
model with seq_len=512, this cuts activations from ~8 GB to ~2 GB.

### CPU offloading (via DeepSpeed ZeRO-Offload or manual `.to('cpu')`)

Optimizer states can live on CPU and be streamed to GPU during the
`.step()`. This roughly doubles the feasible model size on a given GPU
setup but adds 20–40% overhead to training time.

### Parameter-efficient fine-tuning (future work)

Hugo currently fine-tunes all parameters. Future versions may support
LoRA or QLoRA-style adapter training, which would dramatically reduce the
memory footprint by keeping most weights frozen and only training small
adapter matrices.

---

## Training time estimates

These are rough estimates for **one epoch** on WikiText-2-like data
(~20M tokens). Actual times vary with hardware, I/O, and model
architecture.

| Model | Hardware | Config | Time per epoch |
|---|---|---|---|
| 1B | 1× A100 (80 GB) | bs=4, grad_accum=4, seq_len=512 | ~1–2 hours |
| 7B | 2× A100 (80 GB) | bs=1, grad_accum=16, seq_len=512 | ~4–8 hours |
| 13B | 2× A100 (80 GB) | bs=1, grad_accum=16, seq_len=512 | ~8–16 hours |
| 27B | 4× A100 (80 GB) | bs=1, grad_accum=16, seq_len=512 | ~2–4 GPU-days |
| 70B | 8× H100 (80 GB) | bs=1, grad_accum=16, seq_len=512 | ~1–2 GPU-weeks |

For a meaningful QAT result (loss decreases, weights bake out genuinely
ternary), even a partial epoch (~10–50M tokens) shows measurable
improvement over raw PTQ. Run a smoke test first:

```bash
hugo-train-qat \
    --model <your-model> \
    --output ./out/smoke-test \
    --max-steps 20 --limit-samples 200 --batch-size 2 --max-length 128
```

This runs in under 5 minutes on any GPU and validates that loss moves
downward and layers bake out with ≤3 distinct values per row.

---

## PTQ compute: negligible

Post-training quantization (`hugo-ternarize` / `hugo-stream-ternarize`)
runs on CPU:

- **RAM needed:** roughly 2× the model size for `hugo-ternarize` (source
  + quantized copy in memory), or ~1 shard's worth (a few GB) for
  `hugo-stream-ternarize`.
- **Time:** Minutes for models up to ~7B; 10–20 minutes for 27B on a
  modern CPU.
- **Disk:** `hugo-ternarize` writes the output model at the same size as
  the source. `hugo-stream-ternarize` peak disk usage is roughly
  one shard (≤5 GB for Hugging Face defaults).

These paths require no GPU and no training loop — they're pure
transforms.

---

## Multi-GPU guidance

The current `hugo-train-qat` CLI does **not** implement distributed
training internally. For multi-GPU setups, you have two options:

### Option A: Wrap with torchrun

```bash
torchrun --nproc_per_node=4 -m hugo.train_qat \
    --model <model> \
    --output ./out \
    --epochs 1 --batch-size 1 --grad-accum 4 --bf16
```

This requires adding DistributedDataParallel (DDP) or FSDP wrapping
inside `train_qat.py`. The QAT primitives (`convert_to_bitlinear`,
`bake_bitlinear_to_linear`) are framework-agnostic and work inside
any training loop.

### Option B: Use Hugo as a library inside your own training harness

```python
from hugo.qat import convert_to_bitlinear, bake_bitlinear_to_linear

model = AutoModelForCausalLM.from_pretrained("your/model")
convert_to_bitlinear(model, granularity="channel",
                     skip_patterns=["lm_head", "embed_tokens", "norm"])

# ... your FSDP/DeepSpeed training loop here ...

bake_bitlinear_to_linear(model)
model.save_pretrained("./out")
```

This is the recommended path for serious multi-GPU training — Hugo
provides the ternary STE and layer conversion; you bring the
distributed training infrastructure.

---

## Quick reference: hardware recommendations

| Your model size | What you need | Approximate cloud cost (on-demand, 2024) |
|---|---|---|
| ≤ 1B | 1× RTX 4090 (24 GB) | $0.50–1.00/hr (RunPod/Vast) |
| 3–7B | 1× A100 (80 GB) or 2× A6000 (48 GB) | $1.50–3.00/hr (A100 spot) |
| 13B | 2× A100 (80 GB) | $3.00–5.00/hr (A100 spot) |
| 27B | 4× A100 (80 GB) | $6.00–10.00/hr (A100 spot) |
| 70B | 8× H100 (80 GB) with FSDP | $20.00–40.00/hr (H100 on-demand) |

**Spot/preemptible instances** typically cost 50–70% less and are
well-suited to QAT runs checkpointed with `--output`. If the instance
is reclaimed, re-run the same command — Hugo saves after every epoch
(or use `--max-steps` with a specific number).
