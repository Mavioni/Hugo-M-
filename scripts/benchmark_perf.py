import torch, time, json
from transformers import AutoModelForCausalLM, AutoTokenizer

DEV = "cuda"
QAT_DIR = "out\\smollm-135M-qat"
BASE_MODEL = "HuggingFaceTB/SmolLM-135M"

results = {}

def bench_forward(m, seq_len, batch, n_runs=30):
    m.eval()
    ids = torch.randint(0, 20000, (batch, seq_len)).to(DEV)
    with torch.no_grad():
        m(ids)
        torch.cuda.synchronize()
        times = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            m(ids)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
    times.sort()
    med = times[len(times) // 2]
    return med, (batch * seq_len) / med  # tok/s

def bench_generate(m, tok, prompt_tokens=16, gen_tokens=128, n_runs=5):
    m.eval()
    ids = tok.encode("Once upon a time, there was a little girl named Lily. She loved to play", return_tensors="pt")[:, :prompt_tokens].to(DEV)
    # warmup
    with torch.no_grad():
        m.generate(ids, max_new_tokens=16, do_sample=False, pad_token_id=tok.eos_token_id)
    torch.cuda.synchronize()
    times = []
    vram_peak = 0
    for _ in range(n_runs):
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        with torch.no_grad():
            out = m.generate(ids, max_new_tokens=gen_tokens, do_sample=False, pad_token_id=tok.eos_token_id)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
        vram_peak = max(vram_peak, torch.cuda.max_memory_allocated() / 1e9)
    times.sort()
    med = times[len(times) // 2]
    return gen_tokens / med, vram_peak  # tok/s, GB

print("=== Loading models ===")
tok = AutoTokenizer.from_pretrained(BASE_MODEL)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

models = {}
models["fp32"] = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.float32).to(DEV).eval()
models["qat"] = AutoModelForCausalLM.from_pretrained(QAT_DIR, dtype=torch.float32).to(DEV).eval()

print("\n=== 1. Prefill throughput (forward pass, no generate) ===")
for name, m in models.items():
    results[name] = {"prefill": {}}
    for seq in (128, 256, 512):
        for bs in (1, 4):
            dt, tps = bench_forward(m, seq, bs)
            results[name]["prefill"][f"seq{seq}_bs{bs}"] = {"ms": round(dt * 1000, 1), "tok_s": round(tps, 0)}
            print(f"  {name:6s} seq={seq:4d} bs={bs}  {dt*1000:7.1f} ms  {tps:9,.0f} tok/s")

print("\n=== 2. Autoregressive generation (real chat usage) ===")
for name, m in models.items():
    tps, vram = bench_generate(m, tok)
    results[name]["generate"] = {"tok_s": round(tps, 1), "vram_peak_gb": round(vram, 2)}
    print(f"  {name:6s}  {tps:8.1f} tok/s  peak VRAM {vram:.2f} GB")

print("\n=== 3. Memory ===")
for name, m in models.items():
    n_params = sum(p.numel() for p in m.parameters())
    b = sum(p.numel() * p.element_size() for p in m.parameters())
    results[name]["memory"] = {"params": n_params, "bytes": b}
    print(f"  {name:6s}  params={n_params:,}  weights={b/1e6:.1f} MB")

print("\n=== 4. First-token latency (TTFT) ===")
for name, m in models.items():
    ids = torch.randint(0, 20000, (1, 256)).to(DEV)
    with torch.no_grad():
        m(ids)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        m(ids)
        torch.cuda.synchronize()
        ttft = time.perf_counter() - t0
    results[name]["ttft_ms"] = round(ttft * 1000, 1)
    print(f"  {name:6s}  {ttft*1000:7.1f} ms")

with open("out\\perf_benchmark.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved to out/perf_benchmark.json")
