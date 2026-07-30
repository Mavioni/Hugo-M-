"""Quantization-aware training (QAT) for ternary weights.

`quantize.py` and `streaming.py` do post-training quantization (PTQ): take
an already-trained model and round its weights to ternary after the fact.
That works as a storage/tool exercise, but the measured error is large
(~0.5 relative L2 on a real 27B model, see the stream_ternarize.py run this
was validated against) because the weights were never trained to survive
being rounded to {-1, 0, +1}.

This module instead makes the ternary rounding part of the forward pass
*during* training, so gradients push the underlying weights toward values
that round well -- the same idea BitNet b1.58 training uses. The mechanism
is a straight-through estimator (STE): the forward pass quantizes (which
has zero gradient almost everywhere -- round() is piecewise constant), but
the backward pass pretends the quantization step was the identity function
(clipped to where the clamp didn't saturate), so gradients still reach the
underlying full-precision weight and normal optimizers work unmodified.

This uses the exact same absmean scale formula as quantize.py
(`_absmean_scale`), so a model QAT-trained here rounds the same way
quantize.py's PTQ path would round it -- the whole point is for training
and eventual deployment-time rounding to agree.
"""
from __future__ import annotations

import torch
from torch import nn

from hugo.quantize import _absmean_scale, should_skip


class _TernaryFakeQuantSTE(torch.autograd.Function):
    """Forward: ternary fake-quant (differentiable-looking, but round() has
    zero true gradient). Backward: clipped straight-through estimator --
    pass the gradient through unchanged wherever the forward pass's clamp
    didn't saturate, zero it out where it did (matching clamp's real
    gradient there). This is the standard STE used for binary/ternary
    weight training (e.g. BinaryConnect, BitNet)."""

    @staticmethod
    def forward(ctx, weight, granularity, group_size):
        w = weight.float()
        scale = _absmean_scale(w, granularity, group_size)

        if granularity == "group":
            out_features, in_features = w.shape
            grouped = w.view(out_features, in_features // group_size, group_size)
            normalized = grouped / scale
            codes = normalized.round().clamp(-1, 1)
            dequant = (codes * scale).view(out_features, in_features)
            normalized = normalized.view(out_features, in_features)
        else:
            normalized = w / scale
            codes = normalized.round().clamp(-1, 1)
            dequant = codes * scale

        ctx.save_for_backward(normalized)
        return dequant.to(weight.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        (normalized,) = ctx.saved_tensors
        in_range = (normalized.abs() <= 1).to(grad_output.dtype)
        return grad_output * in_range, None, None


def ternary_fake_quant(weight, granularity: str = "channel", group_size: int | None = None):
    """Differentiable ternary fake-quantization of a 2D weight tensor.

    Returns a tensor the same shape/dtype as `weight`, whose *values* are
    ternary (codes * scale) but whose gradient (via the clipped STE) flows
    back to `weight` as if quantization were the identity within the clamp
    range. Use this inside a forward pass during training; use
    `hugo.quantize.ternarize_weight` for one-shot, no-gradient PTQ.
    """
    if weight.dim() != 2:
        raise ValueError(f"expected a 2D weight tensor, got shape {tuple(weight.shape)}")
    return _TernaryFakeQuantSTE.apply(weight, granularity, group_size)


class BitLinear(nn.Linear):
    """Drop-in replacement for `nn.Linear` whose forward pass fake-quantizes
    its own weight to ternary before the matmul. Gradients still reach the
    real (full-precision) `self.weight` via the STE in `ternary_fake_quant`,
    so it trains with a normal optimizer -- the weight just learns to be
    "round-friendly" because rounding happens on every forward pass.
    """

    def __init__(self, in_features, out_features, bias=True, granularity="channel", group_size=None):
        super().__init__(in_features, out_features, bias=bias)
        self.granularity = granularity
        self.group_size = group_size

    def forward(self, x):
        w_q = ternary_fake_quant(self.weight, self.granularity, self.group_size)
        return nn.functional.linear(x, w_q, self.bias)

    @classmethod
    def from_linear(cls, linear: nn.Linear, granularity: str = "channel", group_size: int | None = None) -> BitLinear:
        """Build a BitLinear that trains the SAME parameter tensors as
        `linear` (not copies) -- so converting a pretrained model in place
        continues training its actual pretrained weights, fake-quantized,
        rather than starting fresh."""
        bl = cls(linear.in_features, linear.out_features, bias=linear.bias is not None,
                  granularity=granularity, group_size=group_size)
        bl.weight = linear.weight
        if linear.bias is not None:
            bl.bias = linear.bias
        return bl


def convert_to_bitlinear(
    model: nn.Module,
    granularity: str = "channel",
    group_size: int | None = None,
    skip_patterns: list[str] | None = None,
) -> list[str]:
    """In-place replace every eligible `nn.Linear` in `model` with a
    `BitLinear` sharing the same weight/bias Parameters, so subsequent
    training makes those weights tolerate ternary rounding. Returns the
    list of replaced module names."""
    skip_patterns = skip_patterns or []
    replaced = []
    for parent_name, parent in list(model.named_modules()):
        for child_name, child in list(parent.named_children()):
            full_name = f"{parent_name}.{child_name}" if parent_name else child_name
            if isinstance(child, nn.Linear) and not isinstance(child, BitLinear) and not should_skip(full_name, skip_patterns):
                setattr(parent, child_name, BitLinear.from_linear(child, granularity, group_size))
                replaced.append(full_name)
    return replaced


def bake_bitlinear_to_linear(model: nn.Module) -> list[str]:
    """In-place replace every `BitLinear` in `model` with a plain
    `nn.Linear` holding the *already-ternarized* weight values (no-grad).
    Use this before saving a checkpoint -- the result is an ordinary,
    drop-in-loadable model (same as ternarize.py's PTQ output), just one
    whose ternary weights were trained to be there instead of rounded
    after the fact."""
    baked = []
    for parent_name, parent in list(model.named_modules()):
        for child_name, child in list(parent.named_children()):
            if isinstance(child, BitLinear):
                full_name = f"{parent_name}.{child_name}" if parent_name else child_name
                with torch.no_grad():
                    w_q = ternary_fake_quant(child.weight, child.granularity, child.group_size).clone()
                plain = nn.Linear(child.in_features, child.out_features, bias=child.bias is not None)
                plain.weight = nn.Parameter(w_q.to(child.weight.dtype))
                if child.bias is not None:
                    plain.bias = child.bias
                setattr(parent, child_name, plain)
                baked.append(full_name)
    return baked
