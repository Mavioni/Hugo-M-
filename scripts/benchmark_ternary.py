import torch, time, json
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

DEV = "cuda"
QAT_DIR = "out\\smollm-135M-qat"
BASE_MODEL = "HuggingFaceTB/SmolLM-135M"

results = {}

def measure_ternary(model):
    n_tern = n_zero = n_total = 0
    for mod in model.modules():
        if hasattr(mod, "weight") and mod.weight.dim() == 2:
            w = mod.weight.float()
            scale = w.abs().mean().clamp_min(1e-6)
            codes = (w / scale).round().clamp(-1, 1)
            n_tern += (codes == 1).sum().item() + (codes == -1).sum().item()
            n_zero += (codes == 0).sum().item()
            n_total += w.numel()
    return n_tern / n_total, n_zero / n_total

def perplexity(model, tokenizer, texts, n_batches=8, bs=4, seq=128):
    model.eval()
    total_nll, total_tokens = 0.0, 0
    with torch.no_grad():
        for i in range(n_batches):
            batch_texts = texts[i * bs:(i + 1) * bs]
            enc = tokenizer(batch_texts, return_tensors="pt", padding="max_length",
                            truncation=True, max_length=seq)
            ids = enc["input_ids"].to(DEV)
            labels = ids.clone()
            labels[enc["attention_mask"] == 0] = -100
            logits = model(ids, labels=labels).logits
            shift_logits = logits[:, :-1, :].float()
            shift_labels = labels[:, 1:]
            loss = torch.nn.functional.cross_entropy(
                shift_logits.reshape(-1, shift_logits.shape[-1]),
                shift_labels.reshape(-1), ignore_index=-100, reduction="sum")
            valid = (shift_labels != -100).sum().item()
            total_nll += loss.item()
            total_tokens += valid
    return total_nll / total_tokens, total_tokens

def tokens_per_sec(model, tokenizer, n_tokens=100, seq=256):
    model.eval()
    ids = torch.randint(0, 4000, (1, seq)).to(DEV)
    with torch.no_grad():
        # warmup
        model(ids)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            for _ in range(10):
                model(ids)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
    return (seq * 10) / dt

print("=== Loading models ===")
tok = AutoTokenizer.from_pretrained(BASE_MODEL)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

print("[1/2] fp32 baseline...")
m_fp32 = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.float32).to(DEV)
m_fp32.eval()

print("[2/2] QAT ternary...")
m_qat = AutoModelForCausalLM.from_pretrained(QAT_DIR, dtype=torch.float32).to(DEV)
m_qat.eval()

print("\n=== Data (held-out TinyStories) ===")
ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
texts = []
for i, s in enumerate(ds):
    if i < 8000:
        continue
    texts.append(s["text"])
    if len(texts) >= 64:
        break
print(f"{len(texts)} validation stories (samples 8000+)")

print("\n=== Benchmark ===")
for name, m in [("fp32 baseline", m_fp32), ("QAT ternary", m_qat)]:
    n_params = sum(p.numel() for p in m.parameters())
    ppl, n_tok = perplexity(m, tok, texts)
    tps = tokens_per_sec(m, tok)
    vram = torch.cuda.memory_allocated() / 1e9
    ternary_frac, zero_frac = measure_ternary(m)

    # Generate sample
    ids = tok.encode("Once upon a time", return_tensors="pt").to(DEV)
    with torch.no_grad():
        out = m.generate(ids, max_new_tokens=40, do_sample=True, temperature=0.7,
                         pad_token_id=tok.eos_token_id)
    sample = tok.decode(out[0], skip_special_tokens=True)

    r = {
        "params": n_params,
        "perplexity": round(ppl, 2),
        "tokens_sec": round(tps, 1),
        "vram_gb": round(vram, 2),
        "ternary_frac": round(ternary_frac, 4),
        "zero_frac": round(zero_frac, 4),
        "sample": sample[:100],
    }
    results[name] = r
    print(f"\n{name}:")
    print(f"  params: {n_params:,}")
    print(f"  perplexity: {ppl:.2f}  (lower=better)")
    print(f"  speed: {tps:.1f} tok/s")
    print(f"  VRAM: {vram:.2f} GB")
    print(f"  ternary: {ternary_frac:.1%}  zero: {zero_frac:.1%}")
    print(f"  sample: {sample[:100]}")

# Size on disk
r_fp = sum(f.stat().st_size for f in Path("C:\\Users\\massi\\.cache\\huggingface\\hub").rglob("*.safetensors") if "smollm" in str(f).lower() or "SmolLM" in str(f)) if Path("C:\\Users\\massi\\.cache\\huggingface\\hub").exists() else 0
r_qat = sum(f.stat().st_size for f in Path(QAT_DIR).rglob("*.safetensors"))
print(f"\nDisk: fp32 ~{(sum(p.numel() for p in m_fp32.parameters())*4)/1e9:.2f} GB (fp32), QAT checkpoint {r_qat/1e6:.1f} MB")

with open("out\\benchmark.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved to out/benchmark.json")
