"""Benchmark: Triton packed-ternary kernel vs plain fp16 matmul on a real model.

Usage (from repo root):
    python scripts/benchmark_kernel.py --checkpoint out/qwen0.5b-ternary

Loads the baked checkpoint, swaps every sidecar-covered Linear for a
kernel-backed TernaryLinear, and measures autoregressive decode speed +
peak VRAM for both paths. Results go to out/perf_kernel.json.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from hugo.kernel import replace_linears_with_kernel


def bench_generate(model, tok, device, prompt: str, gen_tokens: int = 128, n_runs: int = 3):
    model.eval()
    ids = tok.encode(prompt, return_tensors="pt")[:, :32].to(device)
    with torch.no_grad():
        model.generate(ids, max_new_tokens=16, do_sample=False,
                       pad_token_id=tok.eos_token_id)
    torch.cuda.synchronize()

    times = []
    vram_peak = 0.0
    for _ in range(n_runs):
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        with torch.no_grad():
            model.generate(ids, max_new_tokens=gen_tokens, do_sample=False,
                           pad_token_id=tok.eos_token_id)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
        vram_peak = max(vram_peak, torch.cuda.max_memory_allocated() / 1e9)
    times.sort()
    median = times[len(times) // 2]
    return gen_tokens / median, vram_peak


def bench_graph_decode(model, tok, device, prompt: str, gen_tokens: int = 128, n_runs: int = 3):
    """Per-token decode time via CUDA-graph replay (launch overhead ~0)."""
    from hugo.kernel.decode import GraphDecoder

    dec = GraphDecoder(model, max_len=256, device=device)
    ids = tok.encode(prompt, return_tensors="pt")
    dec.generate(ids, 16)  # eager prefill + graph capture
    torch.cuda.synchronize()
    times = []
    for _ in range(n_runs):
        dec._reset()  # replay fills the cache from position 0 again
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(gen_tokens):
            dec.graph.replay()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) / gen_tokens)
    return 1.0 / min(times), times[-1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="out/qwen0.5b-ternary")
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    parser.add_argument("--prompt", default="Once upon a time")
    parser.add_argument("--gen-tokens", type=int, default=128)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--graph", action="store_true", help="also measure CUDA-graph decode")
    parser.add_argument("--output", default="out/perf_kernel.json")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = getattr(torch, args.dtype)
    pack_dir = Path(args.checkpoint) / "ternary_packed"
    if not pack_dir.exists():
        err = f"error: no sidecar at {pack_dir} (run the checkpoint with --pack)"
        print(err, file=sys.stderr)
        return 2

    tok = AutoTokenizer.from_pretrained(args.checkpoint)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint, dtype=dtype, trust_remote_code=True
    ).to(device).eval()

    print(f"=== fp16 torch matmul ({device}) ===")
    tps_fp16, vram_fp16 = bench_generate(model, tok, device, args.prompt,
                                         args.gen_tokens, args.runs)
    print(f"  decode {tps_fp16:.1f} tok/s  peak VRAM {vram_fp16:.2f} GB")

    if args.graph:
        print("=== fp16 + CUDA graph ===")
        tps_fp16_graph, _ = bench_graph_decode(model, tok, device, args.prompt,
                                               args.gen_tokens, args.runs)
        print(f"  decode {tps_fp16_graph:.1f} tok/s")

    replaced = replace_linears_with_kernel(model, pack_dir)
    print(f"=== triton packed-ternary kernel ({device}, {replaced} layers) ===")
    tps_kernel, vram_kernel = bench_generate(model, tok, device, args.prompt,
                                             args.gen_tokens, args.runs)
    print(f"  decode {tps_kernel:.1f} tok/s  peak VRAM {vram_kernel:.2f} GB")

    results: dict = {
        "checkpoint": str(Path(args.checkpoint)),
        "dtype": args.dtype,
        "layers_replaced": replaced,
        "fp16_decode_tok_s": round(tps_fp16, 2),
        "kernel_decode_tok_s": round(tps_kernel, 2),
        "speedup": round(tps_kernel / tps_fp16, 3),
        "fp16_peak_vram_gb": round(vram_fp16, 3),
        "kernel_peak_vram_gb": round(vram_kernel, 3),
    }
    if args.graph:
        print("=== triton packed-ternary + CUDA graph ===")
        tps_kernel_graph, _ = bench_graph_decode(model, tok, device, args.prompt,
                                                 args.gen_tokens, args.runs)
        print(f"  decode {tps_kernel_graph:.1f} tok/s")
        results["fp16_graph_tok_s"] = round(tps_fp16_graph, 2)
        results["kernel_graph_tok_s"] = round(tps_kernel_graph, 2)
        results["graph_speedup"] = round(tps_kernel_graph / tps_fp16_graph, 3)
    out = Path(args.output)
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nspeedup: {results['speedup']:.2f}x  -> saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
