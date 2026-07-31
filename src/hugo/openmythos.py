"""
Bridge between Hugo ternary quantization and OpenMythos RDT models.

OpenMythos uses custom transformer layers (MLA, GQA, MoEFFN) that are not
standard Hugging Face modules, but all weight matrices inside them are plain
`nn.Linear` instances. Hugo's core quantization functions walk `nn.Module`
trees and find `nn.Linear` children, so PTQ and QAT work on OpenMythos
models without modification to the quantization logic.

This module provides:
  - `load_mythos_checkpoint` — load a trained OpenMythos from a `.pt` file
  - `MythosLMWrapper`      — wraps OpenMythos with an HF-compatible interface
                              for use with Hugo's existing QAT training loop
  - `quantize_mythos`      — apply Hugo PTQ to an OpenMythos model
  - `MythosQATTrainer`     — QAT fine-tuning loop for OpenMythos models

Example (PTQ):
    from hugo.openmythos import load_mythos_checkpoint, quantize_mythos
    model = load_mythos_checkpoint("checkpoints/step_0020000.pt")
    stats = quantize_mythos(model)
    torch.save(model.state_dict(), "mythos-ternary.pt")

Example (QAT):
    from hugo.openmythos import load_mythos_checkpoint, MythosQATTrainer
    model = load_mythos_checkpoint("checkpoints/step_0020000.pt")
    trainer = MythosQATTrainer(model, dataset_subset="sample-10BT")
    trainer.train(steps=1000, lr=1e-5)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn

from hugo.qat import bake_bitlinear_to_linear, convert_to_bitlinear
from hugo.quantize import quantize_linear_modules

# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------


def load_mythos_checkpoint(
    path: str | Path,
    device: str = "cpu",
    dtype: torch.dtype | None = None,
) -> nn.Module:
    """Load a trained OpenMythos model from a checkpoint file.

    Args:
        path:   Path to a `step_XXXXXXX.pt` file produced by
                `training/3b_fine_web_edu.py`.
        device: Torch device string ("cpu", "cuda", etc.).
        dtype:  Optional torch dtype override (default: use saved config).

    Returns:
        An OpenMythos model on the requested device in eval mode.
    """
    from open_mythos import OpenMythos
    from open_mythos.main import MythosConfig

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    cfg = ckpt.get("cfg")
    if cfg is None:
        raise ValueError("Checkpoint missing 'cfg' field — cannot reconstruct model")

    if isinstance(cfg, dict):
        cfg = MythosConfig(**cfg)

    model = OpenMythos(cfg)
    model.load_state_dict(ckpt["model"])
    model = model.to(device=device, dtype=dtype or torch.float32)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# MythosLMWrapper — HuggingFace-compatible interface
# ---------------------------------------------------------------------------


class MythosLMWrapper(nn.Module):
    """Wraps OpenMythos with an HF-compatible forward interface.

    Hugo's QAT training loop calls `model(**batch).loss` where `batch` has
    keys `input_ids`, `attention_mask`, `labels`. This wrapper maps those
    to OpenMythos's forward signature and computes cross-entropy loss.

    Use this wrapper to pass an OpenMythos model through Hugo's existing
    `train_qat.py` QAT training infrastructure.
    """

    def __init__(self, mythos_model: nn.Module, vocab_size: int):
        super().__init__()
        self.model = mythos_model
        self.vocab_size = vocab_size

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        **kwargs,
    ) -> dict:
        logits = self.model(input_ids)

        result = {"logits": logits}
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = nn.functional.cross_entropy(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            result["loss"] = loss
        return type("ModelOutput", (), result)()

    @property
    def config(self):
        return type(
            "Config",
            (),
            {"vocab_size": self.vocab_size, "name_or_path": "openmythos"},
        )()


# ---------------------------------------------------------------------------
# PTQ: one-shot ternary quantization of OpenMythos
# ---------------------------------------------------------------------------


def quantize_mythos(
    model: nn.Module,
    granularity: str = "channel",
    group_size: int | None = None,
    skip_patterns: list[str] | None = None,
) -> dict:
    """Apply Hugo PTQ to an OpenMythos model in-place.

    By default skips the embedding table and LM head (weight-tied in
    OpenMythos, so quantizing one quantizes both — and you typically
    don't want to ternarize the embedding).

    Args:
        model:         OpenMythos model (or wrapper).
        granularity:   "tensor", "channel", or "group".
        group_size:    Required if granularity="group".
        skip_patterns: Module name substrings to skip.

    Returns:
        Dict mapping layer name to (stats, codes, scale).
    """
    if skip_patterns is None:
        skip_patterns = ["embed", "head", "norm", "act", "halting", "injection"]

    stats, quantized = quantize_linear_modules(
        model, granularity=granularity, group_size=group_size, skip_patterns=skip_patterns
    )

    return {
        name: {
            "stats": s,
            "codes": codes,
            "scale": scale,
        }
        for name, (codes, scale) in quantized.items()
        for s in stats
        if s.name == name
    }


# ---------------------------------------------------------------------------
# QAT: quantization-aware fine-tuning for OpenMythos
# ---------------------------------------------------------------------------


@dataclass
class MythosQATConfig:
    granularity: str = "channel"
    group_size: int | None = None
    skip_patterns: list[str] | None = None
    lr: float = 1e-5
    batch_size: int = 1
    grad_accum: int = 8
    max_length: int = 512
    log_every: int = 10


class MythosQATTrainer:
    """QAT fine-tuning loop for OpenMythos models using Hugo's BitLinear.

    Converts all eligible nn.Linear layers to BitLinear (which fake-quantizes
    weights to ternary during the forward pass), runs fine-tuning with the
    standard STE + AdamW, then bakes the ternary weights back to plain
    nn.Linear for inference.

    Example:
        trainer = MythosQATTrainer(model, dataset_subset="sample-10BT")
        trainer.train(steps=500, lr=1e-5)
        trainer.bake()
    """

    def __init__(
        self,
        model: nn.Module,
        dataset_subset: str = "sample-10BT",
        cfg: MythosQATConfig | None = None,
    ):
        self.cfg = cfg or MythosQATConfig()
        self.model = model
        self.dataset_subset = dataset_subset

        if self.cfg.skip_patterns is None:
            self.cfg.skip_patterns = ["embed", "head", "norm", "act", "halting"]

    def _get_model_for_qat(self) -> nn.Module:
        m = self.model
        while isinstance(m, MythosLMWrapper):
            m = m.model
        return m

    def convert(self) -> int:
        m = self._get_model_for_qat()
        replaced = convert_to_bitlinear(
            m,
            granularity=self.cfg.granularity,
            group_size=self.cfg.group_size,
            skip_patterns=self.cfg.skip_patterns,
        )
        return len(replaced)

    def bake(self) -> int:
        m = self._get_model_for_qat()
        baked = bake_bitlinear_to_linear(m)
        return len(baked)

    def train(
        self,
        steps: int,
        lr: float | None = None,
        device: str = "cuda",
        bf16: bool = True,
    ) -> list[float]:
        from open_mythos.tokenizer import MythosTokenizer

        lr = lr or self.cfg.lr
        dtype = torch.bfloat16 if bf16 else torch.float32

        tokenizer = MythosTokenizer()
        vocab_size = tokenizer.vocab_size

        n_converted = self.convert()
        print(f"Converted {n_converted} layers to BitLinear")

        self.model.to(device=device, dtype=dtype)
        self.model.train()

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        loader = self._build_dataloader(tokenizer)
        data_iter = iter(loader)

        losses = []
        accum_loss = 0.0

        for step_idx in range(1, steps + 1):
            try:
                x, y = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                x, y = next(data_iter)

            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with torch.amp.autocast(
                device_type="cuda", dtype=dtype
            ) if bf16 else torch.no_grad():
                logits = self.model(x)
                loss = nn.functional.cross_entropy(
                    logits.view(-1, vocab_size), y.view(-1)
                ) / self.cfg.grad_accum

            loss.backward()
            accum_loss += loss.item() * self.cfg.grad_accum

            if step_idx % self.cfg.grad_accum == 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                losses.append(accum_loss)
                accum_loss = 0.0

                if step_idx % (self.cfg.log_every * self.cfg.grad_accum) == 0:
                    recent = sum(losses[-self.cfg.log_every :]) / min(
                        self.cfg.log_every, len(losses)
                    )
                    print(f"  qat step {step_idx}/{steps}  loss={recent:.4f}")

        baked = self.bake()
        print(f"QAT complete. {n_converted} layers converted, {baked} baked to ternary.")
        return losses

    def _build_dataloader(self, encoding):
        from datasets import load_dataset

        ds = load_dataset(
            "HuggingFaceFW/fineweb-edu",
            name=self.dataset_subset,
            split="train",
            streaming=True,
        )

        buf = []
        for sample in ds:
            buf.extend(encoding.encode(sample["text"]))
            while len(buf) >= self.cfg.max_length + 1:
                chunk = buf[: self.cfg.max_length + 1]
                buf = buf[self.cfg.max_length + 1 :]
                yield (
                    torch.tensor(chunk[:-1], dtype=torch.long),
                    torch.tensor(chunk[1:], dtype=torch.long),
                )
