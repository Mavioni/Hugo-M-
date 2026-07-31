#!/usr/bin/env python3
"""CLI + helper: publish a Hugo checkpoint to the Hugging Face Hub.

Authentication comes from the `HF_TOKEN` environment variable (or an
existing `huggingface-cli login` cache) -- never from a command-line flag,
so the token can't end up in shell history, process listings, or CI logs.
The token needs *write* scope to create/push to the target repo.

Example:
    export HF_TOKEN=...            # write-scoped token, set outside version control
    hugo-push --checkpoint ./out/qwen3.6-27b-qat --repo-id <you>/Hugo-Qwen3.6-27B-ternary

Pushing a large checkpoint uploads tens of GB; run it somewhere with the
bandwidth and disk for it, and prefer --private until you've verified the
model card and that the checkpoint loads.
"""
from __future__ import annotations

import argparse
import json
import os
from collections.abc import Callable
from pathlib import Path


def _resolve_token() -> str:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        return token

    # Fall back to a cached login (`huggingface-cli login`) if there is one.
    from huggingface_hub import get_token

    cached = get_token()
    if cached:
        return cached

    raise SystemExit(
        "No Hugging Face credentials found. Set a write-scoped token in the environment:\n"
        "    export HF_TOKEN=<your token>\n"
        "or run `huggingface-cli login`. Do not pass the token as a command-line argument."
    )


def build_model_card(checkpoint_dir: Path, repo_id: str) -> str:
    """Generate a model card from the run metadata Hugo wrote at train time.

    Being explicit about *how* the ternary weights were produced matters:
    a QAT'd model and a naively-rounded PTQ model look identical on disk but
    behave very differently, and anyone downloading this deserves to know
    which one they're getting.
    """
    qat_meta = checkpoint_dir / "hugo_qat_run.json"
    ptq_meta = checkpoint_dir / "ternary_quant_stats.json"

    lines = [
        "---",
        "library_name: transformers",
        "tags:",
        "- ternary",
        "- quantization",
        "- bitnet",
        "- 1.58-bit",
        "- hugo",
        "---",
        "",
        f"# {repo_id.split('/')[-1]}",
        "",
        "Ternary-weight (1.58-bit, BitNet b1.58-style) model produced with",
        "[Hugo](https://github.com/Mavioni/Hugo-M-). Every quantized",
        "`nn.Linear` weight takes only three values per scale group:",
        "`{-1, 0, +1} * scale`, where `scale = mean(|W|)`.",
        "",
    ]

    if qat_meta.exists():
        meta = json.loads(qat_meta.read_text())
        lines += [
            "## How this was made: quantization-aware training (QAT)",
            "",
            "The ternary rounding was applied *inside the forward pass during",
            "training* (via a straight-through estimator), so the weights were",
            "trained to tolerate being rounded to ternary rather than being",
            "rounded after the fact. This is the meaningful difference versus",
            "naive post-training quantization.",
            "",
            f"- Source model: `{meta.get('source_model')}`",
            f"- Scale granularity: `{meta.get('granularity')}`"
            + (f" (group size {meta['group_size']})" if meta.get("group_size") else ""),
            f"- Layers trained ternary: {meta.get('converted_layers')}",
            f"- Kept full precision: `{', '.join(meta.get('skip_patterns') or []) or 'none'}`",
            (
                f"- Optimizer steps: {meta.get('optimizer_steps')}, lr {meta.get('lr')}, "
                f"batch size {meta.get('batch_size')} x grad accum {meta.get('grad_accum')}"
            ),
            f"- Training data: `{meta.get('dataset')}`",
            (
                f"- Loss: {meta.get('first_loss'):.4f} at start -> "
                f"{meta.get('mean_final_loss'):.4f} (mean of final micro-batches)"
            )
            if meta.get("first_loss") is not None
            else "",
            "",
        ]
    elif ptq_meta.exists():
        stats = json.loads(ptq_meta.read_text())
        avg_err = sum(s["relative_l2_error"] for s in stats) / max(len(stats), 1)
        lines += [
            "## How this was made: post-training quantization (PTQ)",
            "",
            "**This checkpoint was quantized after training, not trained to be",
            "ternary.** The weights never had to survive ternary rounding, so",
            "expect a substantial quality drop relative to the source model.",
            f"Mean relative L2 weight error across {len(stats)} quantized layers:",
            f"**{avg_err:.4f}**.",
            "",
        ]
    else:
        lines += [
            "## Provenance",
            "",
            "No Hugo run metadata (`hugo_qat_run.json` / `ternary_quant_stats.json`)",
            "was found in this checkpoint directory, so how these ternary weights",
            "were produced could not be recorded automatically.",
            "",
        ]

    lines += [
        "## Caveats worth reading before you use this",
        "",
        "- Weight values are ternary, but they're stored here as ordinary",
        "  fp16/bf16 numbers, so this checkpoint is **not smaller or faster** as-is.",
        "  Realizing the ~8x storage win and any speedup needs a ternary-aware",
        "  runtime (e.g. [bitnet.cpp](https://github.com/microsoft/BitNet)).",
        "- Embeddings, the LM head, and normalization layers are kept at full",
        "  precision, which is standard practice -- rounding those hurts",
        "  disproportionately.",
        "",
        "## Usage",
        "",
        "```python",
        "from transformers import AutoModelForCausalLM, AutoTokenizer",
        "",
        f'model = AutoModelForCausalLM.from_pretrained("{repo_id}")',
        f'tokenizer = AutoTokenizer.from_pretrained("{repo_id}")',
        "```",
        "",
    ]

    return "\n".join(line for line in lines if line is not None)


def push_checkpoint(checkpoint_dir: str | Path, repo_id: str, private: bool = False,
                     write_model_card: bool = True, *,
                     _token_fn: Callable[[], str] = _resolve_token,
                     _api_factory: Callable = None) -> str:
    """Upload a checkpoint directory to the Hub. Returns the repo URL."""

    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.is_dir():
        raise SystemExit(f"checkpoint dir does not exist: {checkpoint_dir}")

    token = _token_fn()
    if _api_factory is None:
        from huggingface_hub import HfApi

        def _default_api(t):
            return HfApi(token=t)

        _api_factory = _default_api
    api = _api_factory(token)

    if write_model_card:
        card_path = checkpoint_dir / "README.md"
        if card_path.exists():
            print(f"  {card_path.name} already exists, leaving it as-is")
        else:
            card_path.write_text(build_model_card(checkpoint_dir, repo_id))
            print(f"  wrote generated model card to {card_path}")

    api.create_repo(repo_id, private=private, exist_ok=True, repo_type="model")
    api.upload_folder(
        folder_path=str(checkpoint_dir),
        repo_id=repo_id,
        repo_type="model",
        ignore_patterns=["_shard_cache/*", "*.lock"],
    )
    return f"https://huggingface.co/{repo_id}"


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True, help="Local checkpoint directory to upload")
    p.add_argument("--repo-id", required=True, help="Target HF repo id, e.g. <user>/Hugo-Qwen3.6-27B-ternary")
    p.add_argument("--private", action="store_true", help="Create the repo as private (recommended first)")
    p.add_argument("--no-model-card", action="store_true", help="Don't generate a README.md model card")
    p.add_argument("--dry-run", action="store_true",
                   help="Generate/preview the model card and check credentials without uploading")
    return p.parse_args(argv)


def main(
    argv: list[str] | None = None,
    *,
    _token_fn: Callable[[], str] = _resolve_token,
    _api_factory: Callable = None,
    _push_fn: Callable = push_checkpoint,
) -> int:
    args = parse_args(argv)
    checkpoint_dir = Path(args.checkpoint)

    if args.dry_run:
        _token_fn()
        print("Credentials: found.")
        print(f"Would create/push to: https://huggingface.co/{args.repo_id} "
              f"({'private' if args.private else 'public'})")
        if not args.no_model_card:
            print("\n--- generated model card ---")
            print(build_model_card(checkpoint_dir, args.repo_id))
        return 0

    url = _push_fn(checkpoint_dir, args.repo_id, private=args.private,
                    write_model_card=not args.no_model_card,
                    _token_fn=_token_fn, _api_factory=_api_factory)
    print(f"Pushed: {url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
