"""Kernel tests: packed-ternary GEMM correctness + fallback paths.

GPU-dependent cases are skipped when triton or CUDA are unavailable (e.g. CI
runs CPU-only torch); the reference-path tests always run.
"""
from __future__ import annotations

import pytest
import torch
from torch import nn

from hugo.kernel import TernaryLinear, replace_linears_with_kernel
from hugo.kernel.ternary import pack_ternary_2bit_rows as pack_ternary_2bit
from hugo.kernel.ternary import ternary_matmul

try:
    import triton  # noqa: F401

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

CUDA_OK = torch.cuda.is_available() and HAS_TRITON


def make_codes(n: int, k: int, zero_frac: float = 0.3, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    codes = torch.randint(-1, 2, (n, k), generator=g, dtype=torch.int8)
    mask = torch.rand(n, k, generator=g) < zero_frac
    codes[mask] = 0
    return codes


def reference_y(x: torch.Tensor, packed: torch.Tensor, scale: torch.Tensor, n: int, k: int) -> torch.Tensor:
    from hugo.kernel.ternary import _reference

    return _reference(x, packed, scale, n, k)


@pytest.mark.skipif(not CUDA_OK, reason="requires CUDA + triton")
@pytest.mark.parametrize(
    "m,n,k,dtype",
    [
        (64, 64, 64, torch.float16),
        (100, 31, 70, torch.float16),
        (1, 1, 4, torch.float16),
        (128, 192, 96, torch.float16),
        (33, 66, 130, torch.float16),
        (64, 64, 130, torch.bfloat16),
        (7, 9, 34, torch.float16),
        (17, 5, 3, torch.float16),
    ],
)
def test_kernel_matches_reference(m: int, n: int, k: int, dtype: torch.dtype) -> None:
    torch.manual_seed(0)
    x = torch.randn(m, k, device="cuda", dtype=dtype)
    codes = make_codes(n, k)
    packed = pack_ternary_2bit(codes).cuda()
    scale = torch.rand(n, device="cuda") * 2.0 + 0.5

    y_kernel = ternary_matmul(x, packed, scale, n, k)
    y_ref = reference_y(x, packed, scale, n, k)

    assert y_kernel.shape == (m, n)
    assert y_kernel.dtype == dtype
    assert torch.allclose(y_kernel.float(), y_ref.float(), atol=1e-2, rtol=1e-2)


@pytest.mark.skipif(not CUDA_OK, reason="requires CUDA + triton")
def test_kernel_zero_padding_contributes_nothing() -> None:
    k = 14  # not a multiple of 4: packed bytes pad with 0 codes
    torch.manual_seed(0)
    x = torch.randn(8, k, device="cuda", dtype=torch.float16)
    codes = make_codes(8, k)
    packed = pack_ternary_2bit(codes).cuda()
    scale = torch.rand(8, device="cuda") + 0.5
    y = ternary_matmul(x, packed, scale, 8, k)
    ref = reference_y(x, packed, scale, 8, k)
    assert torch.allclose(y.float(), ref.float(), atol=1e-2, rtol=1e-2)


@pytest.mark.skipif(not CUDA_OK, reason="requires CUDA + triton")
def test_ternary_linear_equals_original_linear() -> None:
    torch.manual_seed(1)
    linear = nn.Linear(64, 32, bias=True).half().cuda()
    codes = make_codes(32, 64)
    tl = TernaryLinear.from_linear(linear, codes, torch.rand(32, 1) + 0.5).cuda()
    x = torch.randn(16, 64, device="cuda", dtype=torch.float16)
    y_kernel = tl(x)
    w = codes.float().cuda() * tl.scale.reshape(32, 1)
    y_ref = (x.float() @ w.T + linear.bias.float().reshape(1, 32)).to(torch.float16)
    assert torch.allclose(y_kernel.float(), y_ref.float(), atol=1e-2, rtol=1e-2)


def test_reference_path_cpu() -> None:
    codes = make_codes(16, 40)
    packed = pack_ternary_2bit(codes)
    scale = torch.rand(16) + 0.5
    x = torch.randn(5, 40, dtype=torch.float16)
    y = ternary_matmul(x, packed, scale, 16, 40)
    ref = reference_y(x, packed, scale, 16, 40)
    assert torch.allclose(y, ref, atol=1e-3, rtol=1e-3)


def test_ternary_linear_cpu_fallback() -> None:
    codes = make_codes(12, 34)
    linear = nn.Linear(34, 12, bias=True)
    tl = TernaryLinear.from_linear(linear, codes, torch.rand(12, 1) + 0.5)
    x = torch.randn(4, 34)
    y = tl(x)
    w = codes.float() * tl.scale.reshape(12, 1)
    ref = x.float() @ w.T + linear.bias.reshape(1, 12)
    assert torch.allclose(y, ref, atol=1e-4, rtol=1e-4)


def test_ternary_linear_shape_validation() -> None:
    with pytest.raises(ValueError):
        TernaryLinear(torch.zeros(4, 3, dtype=torch.uint8), torch.zeros(3), in_features=8, out_features=4)
    with pytest.raises(ValueError):
        TernaryLinear(torch.zeros(4, 3, dtype=torch.uint8), torch.zeros(2), in_features=12, out_features=4)


def test_in_features_property() -> None:
    tl = TernaryLinear(torch.zeros(12, dtype=torch.uint8), torch.zeros(4), in_features=12, out_features=4)
    assert tl.in_features == 12
    assert tl.out_features == 4
    assert tl.packed.shape == (4, 3)


@pytest.mark.skipif(not CUDA_OK, reason="requires CUDA + triton")
def test_replace_linears_with_kernel(tmp_path) -> None:
    import json as jsonlib

    from safetensors.torch import save_file

    from hugo.quantize import quantize_linear_modules
    from hugo.ternarize import build_packed_sidecar

    torch.manual_seed(2)
    model = nn.Sequential(nn.Linear(16, 8), nn.Linear(8, 8, bias=False))
    biases = [
        m.bias.detach().float() if m.bias is not None else None for m in model.modules()
        if isinstance(m, nn.Linear)
    ]
    _, quantized = quantize_linear_modules(model, granularity="channel")
    assert len(quantized) == 2

    packed_tensors, manifest = build_packed_sidecar(model, quantized, "channel", None)
    pack_dir = tmp_path / "ternary_packed"
    pack_dir.mkdir()
    (pack_dir / "manifest.json").write_text(jsonlib.dumps(manifest))
    save_file(packed_tensors, str(pack_dir / "packed.safetensors"))

    model = model.half().cuda()
    replaced = replace_linears_with_kernel(model, pack_dir)
    assert replaced == 2

    x = torch.randn(6, 16, device="cuda", dtype=torch.float16)
    y_kernel = model(x)

    # reference: chain of dequantized matmuls with the original biases
    y_ref = x
    for (codes, scale), bias in zip(quantized.values(), biases, strict=True):
        w = codes.float().cuda() * scale.cuda()
        y_ref = y_ref.float() @ w.T
        if bias is not None:
            y_ref = y_ref + bias.cuda().reshape(1, -1)

    assert y_kernel.shape == y_ref.shape
    assert torch.allclose(y_kernel.float(), y_ref, atol=1e-2, rtol=1e-2)


@pytest.mark.skipif(not CUDA_OK, reason="requires CUDA + triton")
def test_graph_decoder_matches_eager() -> None:
    from transformers import LlamaConfig, LlamaForCausalLM

    from hugo.kernel.decode import GraphDecoder

    cfg = LlamaConfig(
        vocab_size=64,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
    )
    model = LlamaForCausalLM(cfg).half().cuda().eval()
    dec = GraphDecoder(model, max_len=48)

    # eager path: exactly the engine's own _step
    dec._reset()
    dec.input_ids.copy_(torch.tensor([7], device="cuda"))
    eager = []
    for _ in range(32):
        dec._step()
        eager.append(dec.input_ids.item())

    # graph path: same code, captured and replayed
    dec._reset()
    dec.input_ids.copy_(torch.tensor([7], device="cuda"))
    dec.capture()
    dec._reset()  # capture consumed one step; start replay from pos 0
    dec.input_ids.copy_(torch.tensor([7], device="cuda"))
    graph = []
    for _ in range(32):
        dec.graph.replay()
        graph.append(dec.input_ids.item())

    assert eager == graph
