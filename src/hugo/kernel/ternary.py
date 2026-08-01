"""Triton kernel: 2-bit-packed ternary GEMM (BitNet b1.58-style).

Consumes the exact sidecar layout written by ``pack_ternary_2bit``:
``packed`` uint8 of shape ``[N, ceil(K / 4)]`` (4 ternary codes per byte,
little-endian nibbles: -1 -> 0b00, 0 -> 0b01, 1 -> 0b10) and a per-channel
scale (``granularity="channel"``) of shape ``[N]`` or ``[N, 1]``.

Computes ``y = x @ (codes * scale).T`` in fp32 accumulation, storing in
``x.dtype``. The scale is applied once per output channel *after* the GEMM,
so the inner loop is a pure add/subtract dot product over codes {-1, 0, +1}.
"""
# ruff: noqa: N803, N806  -- Triton kernel params conventionally uppercase
from __future__ import annotations

import torch

from hugo.pure import unpack_ternary_2bit

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:
    triton = None
    tl = None
    HAS_TRITON = False


def _jit(fn):
    """No-op decorator when triton is not installed (kernel extra)."""
    return fn


_jit = triton.jit if HAS_TRITON else _jit

_BLOCK_M = 64
_BLOCK_N = 64
_BLOCK_K = 32

_GEMV_BLOCK_N = 64
_GEMV_BLOCK_K = 128


@_jit
def _ternary_gemv_kernel(
    x_ptr,
    packed_ptr,
    scale_ptr,
    y_ptr,
    N,
    K,
    stride_pn,
    stride_pk,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """GEMV: y[n] = sum_k x[k] * v[n, k] for a single activation row.

    Autoregressive decode feeds one token at a time (M == 1); a tiled GEMM
    kernel would waste most of its rows on masked data. This kernel keeps
    the whole x-block in registers and is weight-bandwidth-bound, which is
    exactly where 2-bit packing wins.
    """
    pid = tl.program_id(0)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_j = tl.arange(0, BLOCK_K // 4)
    out_dtype = x_ptr.dtype.element_ty

    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for k0 in range(0, tl.cdiv(K, BLOCK_K)):
        j_off = k0 * (BLOCK_K // 4) + offs_j
        p = tl.load(
            packed_ptr + offs_n[:, None] * stride_pn + j_off[None, :] * stride_pk,
            mask=(offs_n[:, None] < N) & (j_off[None, :] < tl.cdiv(K, 4)),
            other=0,
        )  # [BN, BK/4] u8
        for i in tl.static_range(4):
            bits = (p >> (2 * i)) & 3  # uint8 nibble
            code = tl.where(bits.to(tl.int32) < 3, bits.to(tl.int32) - 1, 0).to(out_dtype)
            k_pos = k0 * BLOCK_K + offs_j * 4 + i
            xk_i = tl.load(x_ptr + k_pos, mask=k_pos < K, other=0.0)  # [BK/4]
            acc += tl.sum(code * xk_i[None, :], axis=1)

    scale = tl.load(scale_ptr + offs_n, mask=offs_n < N, other=1.0)
    tl.store(
        y_ptr + offs_n,
        (acc * scale).to(out_dtype),
        mask=offs_n < N,
    )


def _ternary_gemv(x: torch.Tensor, packed: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    n = scale.shape[0]
    y = torch.empty((x.shape[0], n), device=x.device, dtype=x.dtype)
    grid = (triton.cdiv(n, _GEMV_BLOCK_N),)
    _ternary_gemv_kernel[grid](
        x,
        packed,
        scale,
        y,
        n,
        x.shape[1],
        packed.stride(0),
        packed.stride(1),
        BLOCK_N=_GEMV_BLOCK_N,
        BLOCK_K=_GEMV_BLOCK_K,
        num_warps=4,
    )
    return y


@_jit
def _ternary_gemm_kernel(
    x_ptr,
    packed_ptr,
    scale_ptr,
    y_ptr,
    M,
    N,
    K,
    stride_xm,
    stride_xk,
    stride_pn,
    stride_pk,
    stride_ym,
    stride_yn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    offs_i = tl.arange(0, 4)

    out_dtype = x_ptr.dtype.element_ty
    scale_ptrs = scale_ptr + offs_n

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, tl.cdiv(K, BLOCK_K)):
        k_off = k0 * BLOCK_K + offs_k
        j_off = k0 * (BLOCK_K // 4) + tl.arange(0, BLOCK_K // 4)
        x = tl.load(
            x_ptr + offs_m[:, None] * stride_xm + k_off[None, :] * stride_xk,
            mask=(offs_m[:, None] < M) & (k_off[None, :] < K),
            other=0.0,
        )
        p = tl.load(
            packed_ptr + offs_n[None, :] * stride_pn + j_off[:, None] * stride_pk,
            mask=(offs_n[None, :] < N) & (j_off[:, None] < tl.cdiv(K, 4)),
            other=0,
        )

        nibbles = (p[:, None, :] >> (2 * offs_i)[None, :, None]) & 3
        codes = tl.reshape(nibbles, (BLOCK_K, BLOCK_N)).to(tl.int32)
        v = tl.where(codes < 3, codes - 1, 0).to(x.dtype)
        acc += tl.dot(x, v)

    scale = tl.load(scale_ptrs, mask=offs_n < N, other=1.0)
    acc = acc * scale[None, :]
    tl.store(
        y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn,
        acc.to(out_dtype),
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


def ternary_matmul(
    x: torch.Tensor,
    packed: torch.Tensor,
    scale: torch.Tensor,
    out_channels: int,
    in_features: int,
) -> torch.Tensor:
    """Packed-ternary matmul: ``y = x @ (codes * scale).T``.

    Args:
        x: activations, ``[M, K]`` fp16/bf16, CUDA.
        packed: ``[N, ceil(K / 4)]`` uint8 (see ``pack_ternary_2bit``).
        scale: per-channel scales, ``[N]`` or ``[N, 1]`` fp32.
        out_channels: N (output channels / rows of the weight matrix).
        in_features: K (input dim / weight columns).

    Returns:
        ``[M, N]`` tensor in ``x.dtype``.
    """
    if x.dim() != 2:
        raise ValueError(f"expected 2D activations, got shape {tuple(x.shape)}")
    if not x.is_cuda:
        return _reference(x, packed, scale, out_channels, in_features)
    if not HAS_TRITON:
        raise RuntimeError(
            "CUDA packed-ternary matmul requires triton; install it with "
            "`pip install hugo[kernel]` (or triton-windows on Windows)"
        )
    if x.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(f"kernel supports fp16/bf16 activations, got {x.dtype}")

    m, k = x.shape
    if in_features != k:
        raise ValueError(f"activation width {k} != weight in_features {in_features}")
    expected = (out_channels, (in_features + 3) // 4)
    if packed.shape != expected:
        if packed.numel() == out_channels * ((in_features + 3) // 4):
            packed = packed.view(expected)
        else:
            raise ValueError(
                f"packed numel {packed.numel()} != expected "
                f"{out_channels * ((in_features + 3) // 4)} for shape {expected}"
            )
    if not packed.is_contiguous():
        packed = packed.contiguous()
    scale = scale.reshape(-1)
    if scale.shape != (out_channels,):
        raise ValueError(f"scale shape {tuple(scale.shape)} != ({out_channels},)")

    if m == 1:
        return _ternary_gemv(x, packed, scale)

    y = torch.empty((m, out_channels), device=x.device, dtype=x.dtype)
    grid = (triton.cdiv(m, _BLOCK_M), triton.cdiv(out_channels, _BLOCK_N))
    _ternary_gemm_kernel[grid](
        x,
        packed,
        scale,
        y,
        m,
        out_channels,
        k,
        x.stride(0),
        x.stride(1),
        packed.stride(0),
        packed.stride(1),
        y.stride(0),
        y.stride(1),
        BLOCK_M=_BLOCK_M,
        BLOCK_N=_BLOCK_N,
        BLOCK_K=_BLOCK_K,
        num_warps=4,
    )
    return y


def pack_ternary_2bit_rows(codes: torch.Tensor) -> torch.Tensor:
    """Pack int8 codes {-1, 0, 1} to 2 bits each, 4 per byte, ROW-aligned.

    ``pack_ternary_2bit`` pads globally (whole tensor), which misaligns
    per-row byte boundaries when ``K % 4 != 0``. The kernel needs every row
    to start at a fresh byte boundary, so this packs each row independently,
    padding each row's tail with 0 codes. Returns ``[N, ceil(K / 4)]`` uint8.
    """
    n, k = codes.shape
    k_padded = (k + 3) // 4 * 4
    if k_padded != k:
        codes = torch.nn.functional.pad(codes, (0, k_padded - k))
    flat = codes.reshape(-1).to(torch.uint8)
    encoded = torch.zeros_like(flat)
    encoded[flat == 255] = 0  # -1 stored as 255 in uint8 view -> 0b00
    encoded[flat == 0] = 1  # 0 -> 0b01
    encoded[flat == 1] = 2  # 1 -> 0b10
    packed = (
        (encoded[0::4] & 0b11)
        | ((encoded[1::4] & 0b11) << 2)
        | ((encoded[2::4] & 0b11) << 4)
        | ((encoded[3::4] & 0b11) << 6)
    )
    return packed.to(torch.uint8).view(n, k_padded // 4)


def _reference(
    x: torch.Tensor,
    packed: torch.Tensor,
    scale: torch.Tensor,
    out_channels: int,
    in_features: int,
) -> torch.Tensor:
    k_padded = (in_features + 3) // 4 * 4
    codes = unpack_ternary_2bit(packed.cpu().reshape(-1), out_channels * k_padded).view(
        out_channels, k_padded
    )[:, :in_features]
    codes = codes.to(x.device)
    w = (codes.float() * scale.reshape(out_channels, 1))[:, :in_features]
    return (x.float() @ w.T).to(x.dtype)
