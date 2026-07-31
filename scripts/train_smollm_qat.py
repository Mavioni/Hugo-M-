"""
SmolLM-135M QAT — ternary-aware training with live monitoring.

Logs per-step metrics to out/smollm-135M-qat/train_log.txt
for swarm agents to analyse in real-time.
"""

import argparse, json, time, io, sys, math
from pathlib import Path
from collections import deque

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

import hugo
from hugo.qat import convert_to_bitlinear, bake_bitlinear_to_linear

# --- metrics collectors ---

class TrainingMonitor:
    """Collects per-step stats and writes to log for agent insight."""

    def __init__(self, log_path: Path, n_recovery: int, n_frozen: int):
        self.log_path = log_path
        self.n_recovery = n_recovery
        self.n_frozen = n_frozen
        self.history = []
        self._buffer = io.StringIO()
        self._step = 0
        self._start = time.perf_counter()

    def step(self, loss: float, grad_norm: float, sparsity: float,
             ternary_fidelity: float, lr: float):
        self._step += 1
        elapsed = time.perf_counter() - self._start
        entry = {
            "step": self._step,
            "loss": round(loss, 4),
            "grad_norm": round(grad_norm, 4),
            "sparsity": round(sparsity, 4),
            "ternary_fidelity": round(ternary_fidelity, 4),
            "lr": round(lr, 8),
            "elapsed_s": round(elapsed, 1),
        }
        self.history.append(entry)
        self.log_path.write_text(json.dumps(self.history, indent=2))

    def insight(self, text: str):
        tqdm.write(f"\n  [INSIGHT] {text}")


def measure_ternary_stats(model, device):
    """How many weights are already in {-1,0,+1}×scale range?"""
    total = 0
    ternary_ready = 0
    zeroed = 0
    with torch.no_grad():
        for mod in model.modules():
            if hasattr(mod, "weight") and mod.weight.dim() == 2:
                w = mod.weight.float()
                scale = w.abs().mean().clamp_min(1e-6)
                codes = (w / scale)
                in_range = (codes.abs() <= 1.0)
                near_int = ((codes - codes.round()).abs() < 0.01)
                total += w.numel()
                ternary_ready += (in_range & near_int).sum().item()
                zeroed += (codes.abs() < 0.01).sum().item()
    if total == 0:
        return 0.0, 0.0
    return zeroed / total, ternary_ready / total


def generate_test(model, tokenizer, device, max_tokens=40):
    prompt = "Once upon a time"
    ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            ids, max_new_tokens=max_tokens, do_sample=True,
            temperature=0.7, top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0], skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="./out/smollm-135M-qat")
    parser.add_argument("--max-steps", type=int, default=4000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--warmup-frac", type=float, default=0.1)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--test-every", type=int, default=500)
    args = parser.parse_args()

    device = args.device
    dtype = torch.bfloat16
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train_log.json"

    # --- Load ---
    MODEL = "HuggingFaceTB/SmolLM-135M"
    print(f"Loading {MODEL} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=dtype).to(device)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"  {n_total:,} params  dim={model.config.hidden_size}")

    # Baseline
    print(f"  [fp32 baseline] {generate_test(model, tokenizer, device)[:100]}\n")

    # --- Convert to BitLinear ---
    skip = ["lm_head", "embed_tokens", "norm"]
    replaced = convert_to_bitlinear(model, granularity="channel", skip_patterns=skip)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  {len(replaced)} Linear -> BitLinear, {n_trainable:,} trainable\n")

    # --- Pre-training ternary stats ---
    sp, fid = measure_ternary_stats(model, device)
    print(f"  [pre-train] sparsity={sp:.1%}  ternary_ready={fid:.1%}")

    # --- Dataset ---
    ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    texts = [s["text"] for i, s in enumerate(ds) if i < 8000]
    print(f"  {len(texts)} TinyStories samples loaded\n")

    # --- Training ---
    monitor = TrainingMonitor(log_path, n_recovery=n_trainable, n_frozen=n_total - n_trainable)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    warmup_steps = int(args.max_steps * args.warmup_frac)
    decay_steps = args.max_steps - warmup_steps

    def get_lr(step):
        if step < warmup_steps:
            return args.lr * step / max(warmup_steps, 1)
        frac = (step - warmup_steps) / max(decay_steps, 1)
        return args.min_lr + (args.lr - args.min_lr) * (1 - frac)

    model.train()
    grad_norms = deque(maxlen=100)
    losses = deque(maxlen=100)

    pbar = tqdm(total=args.max_steps, desc="QAT training", unit="step",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}")

    for step in range(1, args.max_steps + 1):
        # Sample batch
        idxs = torch.randint(0, len(texts), (args.batch_size,))
        enc = tokenizer(
            [texts[i] for i in idxs], return_tensors="pt",
            padding="max_length", truncation=True, max_length=args.max_length,
        )
        labels = enc["input_ids"].clone()
        labels[enc["attention_mask"] == 0] = -100
        batch = {k: v.to(device) for k, v in enc.items()}
        batch["labels"] = labels.to(device)

        loss = model(**batch).loss
        loss.backward()

        cur_lr = get_lr(step)
        for g in optimizer.param_groups:
            g["lr"] = cur_lr

        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        grad_norms.append(gn.item() if isinstance(gn, torch.Tensor) else gn)

        optimizer.step()
        optimizer.zero_grad()

        losses.append(loss.item())

        # Stats every log_every steps
        if step % args.log_every == 0:
            avg_loss = sum(losses) / len(losses)
            avg_gn = sum(grad_norms) / len(grad_norms)
            sp, fid = measure_ternary_stats(model, device)

            monitor.step(
                loss=avg_loss, grad_norm=avg_gn,
                sparsity=sp, ternary_fidelity=fid, lr=cur_lr
            )

            pbar.set_postfix_str(
                f"loss={avg_loss:.3f} gn={avg_gn:.2f} zero={sp:.1%} tern={fid:.1%} lr={cur_lr:.2e}"
            )
            pbar.update(args.log_every)

            # Insight generation at milestones
            if step <= args.log_every:
                monitor.insight(
                    f"Initial loss {avg_loss:.2f} — model sees ternary for the first time. "
                    f"Only {fid:.1%} of weights are ternary-ready. "
                    f"The STE will push weights toward {-1,0,+1}×scale over the next {args.max_steps} steps."
                )
            elif step >= args.max_steps * 0.25 and step - args.log_every < args.max_steps * 0.25:
                # 25% milestone
                loss_delta = losses[0] - avg_loss
                monitor.insight(
                    f"25% complete. Loss dropped {loss_delta:.2f}. "
                    f"Sparsity at {sp:.1%} — the model is learning where to zero out weights. "
                    f"Gradient norm {avg_gn:.2f} is {'healthy' if avg_gn < 2 else 'high'}."
                    f"\n  Ternary fidelity {fid:.1%}: weights migrating into {-1,0,+1} range."
                )
            elif step >= args.max_steps * 0.5 and step - args.log_every < args.max_steps * 0.5:
                monitor.insight(
                    f"Halfway. Loss {avg_loss:.2f} — model should start forming words. "
                    f"Sparsity {sp:.1%}: {'good zeroing' if sp > 0.05 else 'needs more zeros'}. "
                    f"Ternary fidelity {fid:.1%}: {'on track' if fid > 0.3 else 'more training needed'}."
                )
            elif step >= args.max_steps * 0.75 and step - args.log_every < args.max_steps * 0.75:
                monitor.insight(
                    f"75% done. Loss {avg_loss:.2f}. "
                    f"Ternary fidelity now {fid:.1%} — {'approaching deployment quality' if fid > 0.5 else 'converging'}. "
                    f"Gradient norm {avg_gn:.2f} shows {'stabilizing' if avg_gn < 0.5 else 'active learning'}."
                )

        # Generate test
        if step % args.test_every == 0:
            model.eval()
            text = generate_test(model, tokenizer, device, max_tokens=50)
            model.train()
            tqdm.write(f"\n  [gen @{step}] {text[:130]}\n")
            monitor.insight(
                f"Generate test at step {step}: \"{text[:80]}...\" — "
                f"{'coherent!' if len(text.split()) > 5 and not any(text.split().count(w) > 3 for w in text.split()) else 'still learning word patterns'}"
            )

        if step >= args.max_steps:
            break

    pbar.close()

    # --- Final stats ---
    sp, fid = measure_ternary_stats(model, device)
    print(f"\nFinal: loss={sum(losses)/len(losses):.4f}  sparsity={sp:.1%}  ternary_fidelity={fid:.1%}")

    # --- Bake ---
    print("Baking ternary weights ...")
    baked = bake_bitlinear_to_linear(model)
    print(f"  {len(baked)} layers baked")

    # --- Final test ---
    model.eval()
    print(f"\n[final gen] {generate_test(model, tokenizer, device, max_tokens=80)[:150]}")

    # --- Save ---
    model.save_pretrained(out_dir, safe_serialization=True)
    tokenizer.save_pretrained(out_dir)
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
