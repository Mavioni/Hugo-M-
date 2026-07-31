# Hugo-M Session Transfer — 2026-07-31

## Repo
`github.com/Mavioni/Hugo-M-` — Post-training ternary weight quantization for HF models.
Local path: `C:\Users\massi\Hugo-M-`

## What works

### PTQ (Post-Training Quantization)
```bash
hugo-ternarize --model Qwen/Qwen2.5-0.5B --output ./out/qwen0.5b-ternary --granularity channel --pack
```
- 168 linear layers quantized, avg L2 error 0.55, 7.9x smaller packed sidecar
- Model loads via `from_pretrained()` but outputs gibberish — expected, PTQ costs accuracy

### QAT (Quantization-Aware Training)
```bash
hugo-train-qat --model Qwen/Qwen2.5-0.5B --output ./out/qwen0.5b-qat \
    --dataset Salesforce/wikitext --dataset-config wikitext-2-raw-v1 \
    --max-steps 10000 --batch-size 1 --grad-accum 2 --max-length 128 --lr 1e-3 --bf16
```
- 400-step run completed, loss 13.9 → 6.4
- 10,000-step attempt OOM at step 1540 on batch_size=2

## Hardware
- **GPU:** NVIDIA RTX 5060 Laptop, 8 GB VRAM
- **Constraint:** batch_size=1 max for QAT on this GPU
- **Install:** `pip install -e ".[dev,train]"`

## Chat script
`scripts/chat.py` — committed & pushed. Usage:
```bash
python scripts/chat.py out/qwen0.5b-qat
```

## Current state
- QAT training stopped early (OOM at step 1540 with bs=2). Restarted with bs=1, grad_accum=2. Training in progress or stopped.
- HF Hub: NOT logged in (`hf auth login` needed to push models)
- GitHub: Auth token works via `gh` CLI

## Key findings
- PTQ alone = garbled output, QAT is essential
- 8 GB VRAM is very tight — consider cloud GPU (RunPod/Lambda) for meaningful runs
- For 0.5B model: target 10K-50K steps, batch_size=1, grad_accum=2-4
- The model runs with the ternary STE inside forward pass; baked checkpoint works as normal HF model
