#!/usr/bin/env python3
"""CLI: quantization-aware training (QAT) so a model tolerates ternary rounding.

The PTQ path (`hugo-ternarize` / `hugo-stream-ternarize`) rounds an
already-trained model's weights to ternary after the fact, which costs real
accuracy because those weights never had to survive rounding. This script
instead fine-tunes with the ternary rounding *inside* the forward pass (see
`hugo/qat.py`), so the weights learn to be round-friendly, then bakes the
final ternary values into a plain checkpoint.

Requires a GPU for any real model -- this is training, not a one-shot
transform. Rough shape of the cost: a full fine-tune keeps optimizer state
for every trainable parameter, so plan for well beyond the model's own
memory footprint, and expect the run to be measured in GPU-hours (or
GPU-days for a 27B+ model), not minutes. Use --max-steps with a small value
plus --limit-samples first to sanity-check throughput and loss movement on
your hardware before committing to a long run.

Example (small model, smoke test):
    hugo-train-qat \\
        --model yujiepan/qwen2.5-tiny-random \\
        --output ./out/tiny-qat \\
        --dataset Salesforce/wikitext --dataset-config wikitext-2-raw-v1 \\
        --max-steps 20 --limit-samples 64 --batch-size 2 --max-length 128

Example (real run on your own GPU box):
    hugo-train-qat \\
        --model huihui-ai/Huihui-Qwen3.6-27B-abliterated \\
        --output ./out/qwen3.6-27b-qat \\
        --dataset Salesforce/wikitext --dataset-config wikitext-2-raw-v1 \\
        --epochs 1 --batch-size 1 --grad-accum 16 --lr 1e-5 --bf16
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from hugo.qat import bake_bitlinear_to_linear, convert_to_bitlinear

DEFAULT_SKIP = ["lm_head", "embed_tokens", "norm"]


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="HF repo id or local path of the model to QAT-train")
    p.add_argument("--output", required=True, help="Directory to write the QAT'd checkpoint to")
    p.add_argument("--dataset", default="Salesforce/wikitext", help="HF dataset id for the training text")
    p.add_argument("--dataset-config", default="wikitext-2-raw-v1")
    p.add_argument("--dataset-split", default="train")
    p.add_argument("--text-column", default="text")
    p.add_argument("--granularity", choices=["tensor", "channel", "group"], default="channel",
                   help="Ternary scale granularity -- must match what you'll use at deployment time")
    p.add_argument("--group-size", type=int, default=None)
    p.add_argument("--skip", default=",".join(DEFAULT_SKIP),
                   help=f"Comma-separated substrings of module names to leave in full precision "
                        f"(default: {','.join(DEFAULT_SKIP)})")
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--max-steps", type=int, default=None, help="Stop after N optimizer steps (overrides --epochs)")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=1, help="Gradient accumulation steps")
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--max-length", type=int, default=512, help="Token sequence length")
    p.add_argument("--limit-samples", type=int, default=None, help="Use only the first N dataset rows")
    p.add_argument("--bf16", action="store_true", help="Load/train in bfloat16 (recommended on modern GPUs)")
    p.add_argument("--device", default=None, help="Defaults to cuda if available, else cpu")
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--push-to-hub", default=None, metavar="REPO_ID",
                   help="After training, push the baked checkpoint to this HF repo id. "
                        "Requires HF_TOKEN in the environment (see hugo/push_to_hub.py).")
    p.add_argument("--private", action="store_true", help="With --push-to-hub, create the repo as private")
    return p.parse_args(argv)


def build_dataloader(args, tokenizer):
    from datasets import load_dataset

    split = args.dataset_split
    if args.limit_samples:
        split = f"{split}[:{args.limit_samples}]"
    ds = load_dataset(args.dataset, args.dataset_config, split=split)

    texts = [t for t in ds[args.text_column] if t and t.strip()]
    if not texts:
        raise SystemExit(f"no non-empty rows found in column {args.text_column!r} of {args.dataset}")

    def collate(batch_texts):
        enc = tokenizer(
            batch_texts, return_tensors="pt", padding="max_length",
            truncation=True, max_length=args.max_length,
        )
        # Causal LM loss: labels = input_ids, with padding masked out so the
        # model isn't trained to predict pad tokens.
        labels = enc["input_ids"].clone()
        labels[enc["attention_mask"] == 0] = -100
        enc["labels"] = labels
        return enc

    return DataLoader(texts, batch_size=args.batch_size, shuffle=True, collate_fn=collate)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.granularity == "group" and not args.group_size:
        print("error: --granularity=group requires --group-size", file=sys.stderr)
        return 2

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if args.bf16 else torch.float32
    if device == "cpu" and args.bf16:
        print("warning: --bf16 on CPU is slow and poorly supported; consider dropping it", file=sys.stderr)

    print(f"Loading {args.model!r} (device={device}, dtype={dtype}) ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=dtype, trust_remote_code=args.trust_remote_code
    )
    model.to(device)

    skip_patterns = [s for s in args.skip.split(",") if s]
    replaced = convert_to_bitlinear(model, args.granularity, args.group_size, skip_patterns)
    print(f"Converted {len(replaced)} Linear layers to BitLinear (ternary fake-quant in forward pass)")
    if not replaced:
        print("error: no layers were converted -- check --skip patterns", file=sys.stderr)
        return 1

    dataloader = build_dataloader(args, tokenizer)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    total_steps = args.max_steps or math.ceil(len(dataloader) * args.epochs / args.grad_accum)
    print(f"Training for ~{total_steps} optimizer steps "
          f"(batch_size={args.batch_size}, grad_accum={args.grad_accum}, lr={args.lr})")

    model.train()
    step = 0
    losses = []
    first_loss = None
    done = False

    while not done:
        for micro_step, batch in enumerate(dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss / args.grad_accum
            loss.backward()
            losses.append(loss.item() * args.grad_accum)
            if first_loss is None:
                first_loss = losses[0]

            if (micro_step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                step += 1

                if step % args.log_every == 0 or step == 1:
                    window = losses[-args.log_every * args.grad_accum:]
                    print(f"  step {step}/{total_steps}  loss={sum(window) / len(window):.4f}")

                if step >= total_steps:
                    done = True
                    break
        else:
            continue
        break

    mean_last = sum(losses[-20:]) / len(losses[-20:])
    print(f"Training done. first loss={first_loss:.4f}, mean of last 20 micro-batches={mean_last:.4f}")

    print("Baking ternary weights into a plain checkpoint ...")
    baked = bake_bitlinear_to_linear(model)
    print(f"  baked {len(baked)} layers")

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir, safe_serialization=True)
    tokenizer.save_pretrained(out_dir)

    (out_dir / "hugo_qat_run.json").write_text(json.dumps({
        "source_model": args.model,
        "granularity": args.granularity,
        "group_size": args.group_size,
        "skip_patterns": skip_patterns,
        "converted_layers": len(replaced),
        "baked_layers": len(baked),
        "optimizer_steps": step,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "max_length": args.max_length,
        "dataset": f"{args.dataset}/{args.dataset_config}:{args.dataset_split}",
        "first_loss": first_loss,
        "mean_final_loss": mean_last,
    }, indent=2))
    print(f"Saved QAT'd checkpoint to {out_dir}")

    if args.push_to_hub:
        from hugo.push_to_hub import push_checkpoint

        print(f"Pushing to https://huggingface.co/{args.push_to_hub} ...")
        url = push_checkpoint(out_dir, args.push_to_hub, private=args.private)
        print(f"Pushed: {url}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
