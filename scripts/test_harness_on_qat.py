import torch, time, sys
sys.path.insert(0, "..\\openmythos")

torch.manual_seed(42)

from open_mythos.harness import RecoveryTrainer, TernaryHarness, ternarize_model
from transformers import AutoModelForCausalLM, AutoTokenizer

QAT_DIR = "out\\smollm-135M-qat"
DEV = "cuda"

print("[1/4] Loading QAT'd ternary model...")
tok = AutoTokenizer.from_pretrained(QAT_DIR)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
m = AutoModelForCausalLM.from_pretrained(QAT_DIR, dtype=torch.float32).to(DEV)
m.eval()

# Verify weights are ternary
n_tern = 0
n_total = 0
for mod in m.modules():
    if hasattr(mod, "weight") and mod.weight.dim() == 2:
        w = mod.weight.float()
        scale = w.abs().mean().clamp_min(1e-6)
        codes = (w / scale).round()
        n_tern += (codes.abs() <= 1).sum().item()
        n_total += w.numel()
print(f"  weights already ternary: {n_tern/n_total:.1%}")

# Baseline (already ternary, baked)
ids = tok.encode("Once upon a time", return_tensors="pt").to(DEV)
with torch.no_grad():
    out = m.generate(ids, max_new_tokens=60, do_sample=True, temperature=0.7, pad_token_id=tok.eos_token_id)
print(f"  [baked ternary] {tok.decode(out[0], skip_special_tokens=True)[:140]}")

# Ternarize (no-op but ensures {-1,0,1})
print("[2/4] Building harness...")
ternarize_model(m)
harness = TernaryHarness(m, dim=576, n_loops=4).to(DEV)
m.eval()
print(f"  rho(A)={harness.spectral_radius:.4f}  recovery params: {sum(p.numel() for p in harness.recovery_params()):,}")

# Test harness generation WITHOUT training (untrained recovery = zeros LoRA)
print("[3/4] Harness generation (no recovery training)...")
embed = m.get_input_embeddings()
for loops in (1, 4, 8):
    ids = tok.encode("Once upon a time", return_tensors="pt").to(DEV)
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(60):
            embeds = embed(ids)
            hidden = harness(embeds, n_loops=loops)
            logits = m.lm_head(hidden[:, -1, :]) / 0.7
            probs = torch.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, 1)
            if nxt.item() == tok.eos_token_id:
                break
            ids = torch.cat([ids, nxt], dim=-1)
    dt = time.perf_counter() - t0
    text = tok.decode(ids[0], skip_special_tokens=True)
    print(f"  [{loops}L] {text[:120]}")
    print(f"         ({dt:.1f}s, {dt/60:.2f}s/tok)")

# Quick recovery training (30 sec)
print("[4/4] Recovery training (30s)...")
from datasets import load_dataset
ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
texts = [s["text"] for i, s in enumerate(ds) if i < 2000]
trainer = RecoveryTrainer(harness, device=DEV, dtype=torch.float32)
trainer.train(tok, texts, minutes=0.5)

print("\n=== After recovery training ===")
ids = tok.encode("Once upon a time", return_tensors="pt").to(DEV)
with torch.no_grad():
    for _ in range(60):
        embeds = embed(ids)
        hidden = harness(embeds, n_loops=4)
        logits = m.lm_head(hidden[:, -1, :]) / 0.7
        probs = torch.softmax(logits, dim=-1)
        nxt = torch.multinomial(probs, 1)
        if nxt.item() == tok.eos_token_id:
            break
        ids = torch.cat([ids, nxt], dim=-1)
print(f"  [harness 4L + recovery] {tok.decode(ids[0], skip_special_tokens=True)[:140]}")
print("DONE")
