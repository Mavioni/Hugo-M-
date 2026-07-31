"""Pure mathematical operations for BitNet b1.58-style ternary quantization.

This module contains ONLY pure functions -- no I/O, no model mutation, no
global state, no nn.Module traversal. Every function here is:

    - Side-effect-free: produces a result, modifies nothing.
    - Deterministic: same inputs → same outputs (within floating-point
      reproducibility limits of the underlying hardware/PyTorch build).
    - Self-contained: no imports from other Hugo modules.

Invariants that hold for all valid inputs are documented as pre/post
conditions. Use property-based testing (Hypothesis) to statistically verify
these hold across randomly generated inputs.

Reference: Ma et al., "The Era of 1-bit LLMs" (https://arxiv.org/abs/2402.17764).
"""
from __future__ import annotations

import dataclasses
import hashlib
from typing import Literal

import torch

Granularity = Literal["tensor", "channel", "group"]


@dataclasses.dataclass(frozen=True)
class LayerQuantStats:
    """Immutable per-layer quantization statistics.

    POSTCONDITION: 0 <= relative_l2_error  (will be 0 only if W was already ternary)
    POSTCONDITION: 0 <= zero_fraction <= 1
    """
    name: str
    shape: tuple[int, ...]
    granularity: str
    relative_l2_error: float
    zero_fraction: float


@dataclasses.dataclass(frozen=True)
class QuantResult:
    """Immutable result of ternarize_weight.

    POSTCONDITION: all(code in {-1, 0, 1} for code in codes.flatten())
    POSTCONDITION: codes.shape == original_shape
    POSTCONDITION: scale.min() > 0
    POSTCONDITION: codes.dtype == torch.int8
    POSTCONDITION: scale.dtype == torch.float32
    """
    codes: torch.Tensor
    scale: torch.Tensor
    original_shape: tuple[int, ...]
    granularity: Granularity
    group_size: int | None


# ── Core quantization math ───────────────────────────────────────────

def _absmean_scale(weight: torch.Tensor, granularity: str, group_size: int | None) -> torch.Tensor:
    """Compute the absmean scale tensor, broadcastable against `weight`.

    PRECONDITION: weight.dim() == 2
    PRECONDITION: granularity in {"tensor", "channel", "group"}
    PRECONDITION: if granularity == "group": group_size > 0 and in_features % group_size == 0

    POSTCONDITION: scale.min() > 0  (clamped to epsilon)
    POSTCONDITION: scale.shape matches the broadcast shape for the granularity

    weight is [out_features, in_features]. Returns a scale tensor of shape:
      - "tensor":  scalar (broadcasts to everything)
      - "channel": [out_features, 1] (one scale per output row)
      - "group":   [out_features, in_features // group_size, 1]
    """
    eps = 1e-6
    if granularity == "tensor":
        return weight.abs().mean().clamp_min(eps)
    if granularity == "channel":
        return weight.abs().mean(dim=1, keepdim=True).clamp_min(eps)
    if granularity == "group":
        if group_size is None or group_size <= 0:
            raise ValueError("group granularity requires a positive group_size")
        out_features, in_features = weight.shape
        if in_features % group_size != 0:
            raise ValueError(
                f"in_features={in_features} is not divisible by group_size={group_size}"
            )
        grouped = weight.view(out_features, in_features // group_size, group_size)
        return grouped.abs().mean(dim=2, keepdim=True).clamp_min(eps)
    raise ValueError(f"unknown granularity: {granularity!r}")


def ternarize_weight(
    weight: torch.Tensor,
    granularity: str = "channel",
    group_size: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a 2D weight tensor to ternary values.

    PRECONDITION:  weight.dim() == 2
    PRECONDITION:  granularity in {"tensor", "channel", "group"}
    PRECONDITION:  if granularity == "group": group_size > 0 and in_features % group_size == 0

    POSTCONDITION: codes.shape == weight.shape
    POSTCONDITION: codes.dtype == torch.int8
    POSTCONDITION: ∀ code ∈ codes.unique(): code ∈ {-1, 0, +1}
    POSTCONDITION: scale.min() > 0
    POSTCONDITION: scale.dtype == torch.float32
    POSTCONDITION: scale is broadcastable against codes

    Returns (ternary_codes, scale):
      - ternary_codes: same shape as `weight`, dtype int8, values in {-1, 0, 1}
      - scale: the absmean scale tensor used, dtype float32
    """
    if weight.dim() != 2:
        raise ValueError(f"expected a 2D weight tensor, got shape {tuple(weight.shape)}")

    w = weight.float()
    scale = _absmean_scale(w, granularity, group_size)

    if granularity == "group":
        out_features, in_features = w.shape
        grouped = w.view(out_features, in_features // group_size, group_size)
        codes = (grouped / scale).round().clamp(-1, 1)
        codes = codes.view(out_features, in_features)
    else:
        codes = (w / scale).round().clamp(-1, 1)

    return codes.to(torch.int8), scale.float()


def dequantize_weight(
    codes: torch.Tensor, scale: torch.Tensor, group_size: int | None = None
) -> torch.Tensor:
    """Reconstruct a float weight tensor from ternary codes + scale.

    PRECONDITION:  codes.dim() == 2
    PRECONDITION:  scale is broadcastable against codes (same granularity as used for ternarize)

    POSTCONDITION: result.shape == codes.shape
    POSTCONDITION: each row contains at most 3 distinct values: {-s, 0, +s} for some s > 0
    """
    codes_f = codes.float()
    if scale.dim() == 3:  # group granularity
        out_features, in_features = codes.shape
        grouped = codes_f.view(out_features, in_features // group_size, group_size)
        return (grouped * scale).view(out_features, in_features)
    return codes_f * scale


def quantization_stats(name: str, original: torch.Tensor, codes: torch.Tensor, scale: torch.Tensor,
                       granularity: str, group_size: int | None = None) -> LayerQuantStats:
    """Compute per-layer quantization error statistics.

    PRECONDITION: original.shape == codes.shape
    PRECONDITION: scale is broadcastable against codes

    POSTCONDITION: 0 <= stats.relative_l2_error
    POSTCONDITION: 0 <= stats.zero_fraction <= 1
    """
    dequant = dequantize_weight(codes, scale, group_size).to(original.dtype)
    diff = (original - dequant).float()
    rel_l2 = torch.linalg.norm(diff) / torch.linalg.norm(original.float()).clamp_min(1e-12)
    zero_fraction = (codes == 0).float().mean().item()
    return LayerQuantStats(
        name=name,
        shape=tuple(original.shape),
        granularity=granularity,
        relative_l2_error=rel_l2.item(),
        zero_fraction=zero_fraction,
    )


# ── 2-bit packing / unpacking ────────────────────────────────────────

def pack_ternary_2bit(codes: torch.Tensor) -> torch.Tensor:
    """Pack ternary codes {-1,0,1} into 2 bits each, 4 values per byte.

    PRECONDITION: all(code in {-1, 0, 1} for code in codes.flatten())

    POSTCONDITION: packed.dtype == torch.uint8
    POSTCONDITION: packed.numel() == ceil(codes.numel() / 4)
    POSTCONDITION: unpack(pack(codes))[:codes.numel()] == codes  (round-trip)

    Encoding: -1 -> 0b00, 0 -> 0b01, 1 -> 0b10 (0b11 unused).
    2 bits/weight, 4 codes per byte, i.e. ~8x compression vs fp16.
    """
    flat = codes.reshape(-1).to(torch.uint8)
    encoded = torch.zeros_like(flat)
    encoded[flat == 255] = 0  # -1 stored as 255 in uint8 view -> map to 0b00
    encoded[flat == 0] = 1    # 0 -> 0b01
    encoded[flat == 1] = 2    # 1 -> 0b10

    pad = (-flat.numel()) % 4
    if pad:
        encoded = torch.cat([encoded, torch.zeros(pad, dtype=torch.uint8)])
    encoded = encoded.view(-1, 4)
    packed = (
        (encoded[:, 0] & 0b11)
        | ((encoded[:, 1] & 0b11) << 2)
        | ((encoded[:, 2] & 0b11) << 4)
        | ((encoded[:, 3] & 0b11) << 6)
    )
    return packed.to(torch.uint8)


def unpack_ternary_2bit(packed: torch.Tensor, num_elements: int) -> torch.Tensor:
    """Inverse of pack_ternary_2bit. Returns int8 codes in {-1, 0, 1}.

    PRECONDITION: packed.dtype == torch.uint8
    PRECONDITION: num_elements >= 0

    POSTCONDITION: result.numel() >= num_elements
    POSTCONDITION: result[:num_elements] values ∈ {-1, 0, 1}
    POSTCONDITION: result.dtype == torch.int8
    """
    lookup = torch.tensor([-1, 0, 1, 0], dtype=torch.int8)
    out = torch.empty(packed.numel() * 4, dtype=torch.int8)
    for i in range(4):
        bits = (packed >> (2 * i)) & 0b11
        out[i::4] = lookup[bits.long()]
    return out[:num_elements]


# ── Utility predicates ──────────────────────────────────────────────

def should_skip(module_name: str, skip_patterns: list[str]) -> bool:
    """Check whether a module name matches any skip pattern.

    POSTCONDITION: returns True iff any skip_pattern is a substring of module_name
    POSTCONDITION: if skip_patterns is empty, returns False
    """
    return any(pattern in module_name for pattern in skip_patterns)


def codes_are_ternary(codes: torch.Tensor) -> bool:
    """Verify that all elements of the tensor are in {-1, 0, 1}.

    POSTCONDITION: returns True iff codes.unique() ⊆ {-1, 0, 1}
    """
    return set(codes.unique().tolist()).issubset({-1, 0, 1})


def active_code_fraction(codes: torch.Tensor) -> float:
    """Fraction of non-zero (active) ternary codes.

    PRECONDITION: codes are in {-1, 0, 1} (not enforced, just documented)

    POSTCONDITION: 0 <= result <= 1
    POSTCONDITION: result = 1 - zero_fraction
    """
    return (codes != 0).float().mean().item()


def ternarize_is_contractive(
    weight: torch.Tensor,
    granularity: str = "channel",
    group_size: int | None = None,
) -> bool:
    """Verify the contraction property: ‖deq(tern(W))‖₂ ≤ ‖W‖₂.

    This is a mathematically guaranteed property (rounding toward zero and
    clamping to [-1,1] never increases element magnitudes), proved here
    empirically for a given concrete tensor. Returns True iff the property
    holds for the given input.
    """
    codes, scale = ternarize_weight(weight, granularity, group_size)
    dequant = dequantize_weight(codes, scale, group_size)
    return bool(torch.linalg.norm(dequant.float()) <= torch.linalg.norm(weight.float()) + 1e-6)


def compute_tensor_sha256(tensor: torch.Tensor) -> str:
    """Content-addressable SHA-256 hash of a tensor's canonical byte representation.

    Serializes the tensor as contiguous float32 bytes (numpy canonical layout)
    and returns the hex digest. Two logically equal tensors in different memory
    layouts will produce the same hash.

    POSTCONDITION: len(result) == 64  (32-byte hex digest)
    POSTCONDITION: hash is deterministic for the same tensor values
    """
    return hashlib.sha256(tensor.detach().float().contiguous().numpy().tobytes()).hexdigest()


def hash_packed_layer(codes: torch.Tensor, scale: torch.Tensor) -> str:
    """SHA-256 hash of a packed (codes, scale) pair.

    POSTCONDITION: len(result) == 64
    """
    codes_hash = compute_tensor_sha256(codes)
    scale_hash = compute_tensor_sha256(scale)
    return hashlib.sha256(f"{codes_hash}:{scale_hash}".encode()).hexdigest()


def merkle_root(leaf_hashes: list[str]) -> str:
    """Build a Merkle tree root from leaf hashes.

    Each leaf is a hex string. Pairs are concatenated and re-hashed until
    a single root hash remains. An empty list returns the SHA-256 of b"".
    If the number of leaves is odd, the last leaf is duplicated.

    POSTCONDITION: len(result) == 64
    """
    if not leaf_hashes:
        return hashlib.sha256(b"").hexdigest()
    level = [bytes.fromhex(h) for h in leaf_hashes]
    while len(level) > 1:
        if len(level) % 2 == 1:
            level.append(level[-1])
        next_level = []
        for i in range(0, len(level), 2):
            combined = level[i] + level[i + 1]
            next_level.append(hashlib.sha256(combined).digest())
        level = next_level
    return level[0].hex()


def hash_manifest_shard(packed_file_path_data: bytes) -> str:
    """SHA-256 hash of a serialized packed shard file's bytes.

    PRECONDITION: path exists and is readable (caller's responsibility)
    POSTCONDITION: len(result) == 64
    """
    return hashlib.sha256(packed_file_path_data).hexdigest()


def verify_manifest_integrity(manifest: dict) -> tuple[bool, list[str]]:
    """Verify the content-integrity Merkle root of a stream_ternarize manifest.

    Walks every "done" shard entry in the manifest, reads each shard's
    `sha256` field (if present), and recomputes the Merkle root. Returns
    (valid, errors) where errors is a list of human-readable violation
    messages. If no shards have `sha256` entries, returns (True, []) since
    there is nothing to verify (pre-Merkle manifest).

    POSTCONDITION: if not manifest["shards"], returns (True, [])
    """
    shards = manifest.get("shards", {})
    leaf_hashes: list[str] = []
    errors: list[str] = []

    shard_names = sorted(shards)
    for name in shard_names:
        entry = shards[name]
        if entry.get("status") != "done":
            errors.append(f"shard {name!r}: status is {entry.get('status')!r}, not 'done'")
            continue
        stored_hash = entry.get("sha256")
        if stored_hash is None:
            errors.append(f"shard {name!r}: missing 'sha256' field")
            continue
        leaf_hashes.append(stored_hash)

    if not leaf_hashes and not errors:
        return True, []

    if errors:
        return False, errors

    computed_root = merkle_root(leaf_hashes)
    stored_root = manifest.get("merkle_root")

    if stored_root is None:
        errors.append("manifest is missing 'merkle_root' field")
        return False, errors

    if computed_root != stored_root:
        errors.append(
            f"Merkle root mismatch: manifest has {stored_root}, computed {computed_root}"
        )
        return False, errors

    return True, []


def build_shard_integrity_hash(packed_tensors: dict[str, torch.Tensor]) -> str:
    """Compute SHA-256 over all packed tensor bytes in a deterministic order.

    Tensors are hashed in sorted key order for reproducibility.

    POSTCONDITION: len(result) == 64
    """
    h = hashlib.sha256()
    for key in sorted(packed_tensors):
        h.update(key.encode())
        h.update(compute_tensor_sha256(packed_tensors[key]).encode())
    return h.hexdigest()
