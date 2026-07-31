"""Property-based tests (Hypothesis) for hugo.pure — the pure quantization math.

These tests statistically prove the documented invariants hold for all
valid randomly generated inputs. Hypothesis automatically shrinks
counterexamples to minimal failing cases.

Run with:
    pytest tests/test_pure.py -v
"""
import hashlib

import pytest
import torch
from hypothesis import assume, given
from hypothesis import strategies as st

from hugo.pure import (
    active_code_fraction,
    build_shard_integrity_hash,
    codes_are_ternary,
    compute_tensor_sha256,
    dequantize_weight,
    hash_packed_layer,
    merkle_root,
    pack_ternary_2bit,
    quantization_stats,
    should_skip,
    ternarize_is_contractive,
    ternarize_weight,
    unpack_ternary_2bit,
    verify_manifest_integrity,
)

# ── Strategies ───────────────────────────────────────────────────────

_ternary_value = st.sampled_from([-1, 0, 1])


@st.composite
def weight_matrix(draw, min_rows=1, max_rows=32, min_cols=1, max_cols=64) -> torch.Tensor:
    rows = draw(st.integers(min_rows, max_rows))
    cols = draw(st.integers(min_cols, max_cols))
    return draw(
        st.builds(
            lambda shape, values: torch.tensor(values, dtype=torch.float32).reshape(shape),
            shape=st.just((rows, cols)),
            values=st.lists(st.floats(-100, 100), min_size=rows * cols, max_size=rows * cols),
        )
    )


@st.composite
def ternary_codes(draw, min_size=1, max_size=256) -> torch.Tensor:
    size = draw(st.integers(min_size, max_size))
    values = draw(st.lists(_ternary_value, min_size=size, max_size=size))
    return torch.tensor(values, dtype=torch.int8)


@st.composite
def weight_with_granularity(draw):
    """Generate a weight matrix and a valid granularity/group_size combo."""
    granularity = draw(st.sampled_from(["tensor", "channel", "group"]))
    rows = draw(st.integers(1, 16))
    if granularity == "group":
        group_size = draw(st.sampled_from([2, 4, 8]))
        cols = group_size * draw(st.integers(1, 8))
    else:
        cols = draw(st.integers(1, 32))
        group_size = None
    n_elements = rows * cols
    values = draw(st.lists(st.floats(-50, 50), min_size=n_elements, max_size=n_elements))
    w = torch.tensor(values, dtype=torch.float32).reshape(rows, cols)
    return w, granularity, group_size


# ── ternarize_weight ─────────────────────────────────────────────────


@given(weight_matrix(), st.sampled_from(["tensor", "channel"]))
def test_ternarize_values_are_ternary(w, granularity):
    codes, scale = ternarize_weight(w, granularity=granularity)
    assert codes_are_ternary(codes), f"codes contained: {codes.unique().tolist()}"
    assert scale.min().item() > 0


@given(weight_matrix(), st.sampled_from(["tensor", "channel"]))
def test_ternarize_shapes_match(w, granularity):
    codes, scale = ternarize_weight(w, granularity=granularity)
    assert codes.shape == w.shape
    assert codes.dtype == torch.int8
    assert scale.dtype == torch.float32


@given(weight_matrix(min_rows=1, max_rows=16, min_cols=1, max_cols=32))
def test_channel_granularity_scale_shape(w):
    codes, scale = ternarize_weight(w, granularity="channel")
    assert scale.shape == (w.shape[0], 1)


@given(weight_matrix(min_rows=1, max_rows=16, min_cols=1, max_cols=32))
def test_tensor_granularity_scale_scalar(w):
    codes, scale = ternarize_weight(w, granularity="tensor")
    assert scale.numel() == 1


@given(weight_matrix())
def test_ternarize_contractive_property(w):
    assert ternarize_is_contractive(w)


@given(weight_matrix(min_cols=4, max_cols=16))
def test_group_granularity_valid(w):
    assume(w.shape[1] % 4 == 0)
    codes, scale = ternarize_weight(w, granularity="group", group_size=4)
    assert codes.shape == w.shape
    assert codes_are_ternary(codes)
    assert scale.dim() == 3
    assert scale.shape == (w.shape[0], w.shape[1] // 4, 1)


@given(weight_matrix())
def test_channel_finer_than_tensor(w):
    """Channel granularity should produce ≤ error vs tensor on the same weight."""
    assume(w.shape[0] >= 4)  # need multiple rows for channel to matter
    codes_t, scale_t = ternarize_weight(w, granularity="tensor")
    codes_c, scale_c = ternarize_weight(w, granularity="channel")
    err_t = quantization_stats("w", w, codes_t, scale_t, "tensor").relative_l2_error
    err_c = quantization_stats("w", w, codes_c, scale_c, "channel").relative_l2_error
    assert err_c <= err_t + 1e-5, f"channel={err_c} vs tensor={err_t}"


@given(weight_matrix())
def test_ternarize_dequantize_per_row_at_most_3_values(w):
    """Dequantized channel-ternary weights have ≤ 3 distinct values per row."""
    codes, scale = ternarize_weight(w, granularity="channel")
    dequant = dequantize_weight(codes, scale)
    for row in dequant:
        assert row.unique().numel() <= 3


@given(weight_matrix())
def test_dequantize_equals_codes_times_scale(w):
    codes, scale = ternarize_weight(w, granularity="channel")
    recon = dequantize_weight(codes, scale)
    expected = codes.float() * scale
    assert torch.allclose(recon, expected)


# ── pack / unpack ─────────────────────────────────────────────────────


@given(ternary_codes())
def test_pack_unpack_roundtrip(codes):
    packed = pack_ternary_2bit(codes)
    restored = unpack_ternary_2bit(packed, codes.numel())
    assert torch.equal(codes, restored)


@given(ternary_codes(min_size=1, max_size=256))
def test_pack_output_size(codes):
    packed = pack_ternary_2bit(codes)
    expected_bytes = (codes.numel() + 3) // 4
    assert packed.numel() == expected_bytes, f"n={codes.numel()}, got {packed.numel()} bytes"


@given(ternary_codes(min_size=1, max_size=200))
def test_packed_values_in_valid_range(codes):
    packed = pack_ternary_2bit(codes)
    assert packed.dtype == torch.uint8
    assert (packed <= 0xFF).all()


@given(ternary_codes(min_size=1, max_size=128))
def test_unpack_all_ternary(codes):
    packed = pack_ternary_2bit(codes)
    restored = unpack_ternary_2bit(packed, codes.numel())
    assert codes_are_ternary(restored)


@given(ternary_codes(min_size=5, max_size=50))
def test_pack_unpack_roundtrip_odd_size(codes):
    assume(codes.numel() % 4 != 0)
    packed = pack_ternary_2bit(codes)
    restored = unpack_ternary_2bit(packed, codes.numel())
    assert torch.equal(codes, restored)


# ── hashing / Merkle ──────────────────────────────────────────────────


@given(weight_matrix())
def test_compute_tensor_sha256_deterministic(w):
    h1 = compute_tensor_sha256(w)
    h2 = compute_tensor_sha256(w)
    assert h1 == h2
    assert len(h1) == 64


@given(weight_matrix())
def test_hash_packed_layer_consistent(w):
    codes, scale = ternarize_weight(w, granularity="channel")
    h1 = hash_packed_layer(codes, scale)
    h2 = hash_packed_layer(codes, scale)
    assert h1 == h2
    assert len(h1) == 64


@given(weight_matrix())
def test_build_shard_integrity_hash_deterministic(w):
    codes, scale = ternarize_weight(w, granularity="channel")
    packed = pack_ternary_2bit(codes)
    tensors = {"test.packed": packed, "test.scale": scale}
    h1 = build_shard_integrity_hash(tensors)
    h2 = build_shard_integrity_hash(tensors)
    assert h1 == h2
    assert len(h1) == 64


@given(st.lists(st.text(alphabet="0123456789abcdef", min_size=64, max_size=64), min_size=1, max_size=8))
def test_merkle_root_deterministic(leaf_hexes):
    r1 = merkle_root(leaf_hexes)
    r2 = merkle_root(leaf_hexes)
    assert r1 == r2
    assert len(r1) == 64


def test_merkle_root_empty():
    r = merkle_root([])
    assert len(r) == 64


def test_merkle_root_single_leaf():
    leaf = hashlib.sha256(b"hello").hexdigest()
    r = merkle_root([leaf])
    assert r == leaf  # single leaf IS the root


def test_merkle_root_two_leaves_known():
    l1 = hashlib.sha256(b"a").hexdigest()
    l2 = hashlib.sha256(b"b").hexdigest()
    r = merkle_root([l1, l2])
    expected = hashlib.sha256(bytes.fromhex(l1) + bytes.fromhex(l2)).hexdigest()
    assert r == expected


def test_merkle_root_odd_leaves_duplicates_last():
    l1 = hashlib.sha256(b"a").hexdigest()
    l2 = hashlib.sha256(b"b").hexdigest()
    l3 = hashlib.sha256(b"c").hexdigest()
    r = merkle_root([l1, l2, l3])
    # With 3 leaves, effectively: merkle([a, b, c, c])
    left = hashlib.sha256(bytes.fromhex(l1) + bytes.fromhex(l2)).hexdigest()
    right = hashlib.sha256(bytes.fromhex(l3) + bytes.fromhex(l3)).hexdigest()
    expected = hashlib.sha256(bytes.fromhex(left) + bytes.fromhex(right)).hexdigest()
    assert r == expected


def test_verify_manifest_integrity_valid():
    manifest = {
        "shards": {
            "shard_0": {
                "status": "done",
                "sha256": "a" * 64,
            },
            "shard_1": {
                "status": "done",
                "sha256": "b" * 64,
            },
        },
        "merkle_root": merkle_root(["a" * 64, "b" * 64]),
    }
    valid, errors = verify_manifest_integrity(manifest)
    assert valid
    assert errors == []


def test_verify_manifest_integrity_tampered():
    manifest = {
        "shards": {
            "shard_0": {
                "status": "done",
                "sha256": "a" * 64,
            },
        },
        "merkle_root": "0" * 64,  # wrong root
    }
    valid, errors = verify_manifest_integrity(manifest)
    assert not valid
    assert len(errors) > 0


def test_verify_manifest_integrity_no_hashes():
    manifest = {
        "shards": {
            "shard_0": {"status": "done"},
        }
    }
    valid, errors = verify_manifest_integrity(manifest)
    assert not valid
    assert any("missing 'sha256'" in e for e in errors)


def test_verify_manifest_integrity_empty():
    manifest = {"shards": {}}
    valid, errors = verify_manifest_integrity(manifest)
    assert valid
    assert errors == []


# ── predicates ────────────────────────────────────────────────────────


@given(st.lists(st.text(alphabet="abcdefghijklmn", min_size=3, max_size=10), min_size=0, max_size=5),
       st.text(alphabet="0123456789abcdef", min_size=4, max_size=12))
def test_should_skip_matches_substring(patterns, name):
    result = should_skip(name, patterns)
    assert result == any(p in name for p in patterns)


def test_should_skip_empty_patterns():
    assert not should_skip("model.layers.0.fc1", [])


def test_should_skip_exact_match():
    assert should_skip("lm_head", ["lm_head"])


def test_should_skip_substring_match():
    assert should_skip("model.lm_head.weight", ["lm_head"])


@given(ternary_codes(min_size=1, max_size=256))
def test_codes_are_ternary_true_for_ternary(codes):
    assert codes_are_ternary(codes)


def test_codes_are_ternary_false_for_non_ternary():
    codes = torch.tensor([-2, 0, 1], dtype=torch.int8)
    assert not codes_are_ternary(codes)
    codes = torch.tensor([-1, 0, 2], dtype=torch.int8)
    assert not codes_are_ternary(codes)


@given(ternary_codes(min_size=1, max_size=256))
def test_active_code_fraction_range(codes):
    f = active_code_fraction(codes)
    assert 0.0 <= f <= 1.0


def test_active_code_fraction_all_active():
    codes = torch.tensor([-1, 1, -1, 1], dtype=torch.int8)
    assert active_code_fraction(codes) == 1.0


def test_active_code_fraction_all_zeros():
    codes = torch.zeros(10, dtype=torch.int8)
    assert active_code_fraction(codes) == 0.0


@given(ternary_codes(min_size=1, max_size=100))
def test_active_code_fraction_zero_fraction_sum(codes):
    active = active_code_fraction(codes)
    zero_fraction = (codes == 0).float().mean().item()
    assert abs(active + zero_fraction - 1.0) < 1e-6


# ── quantization_stats ────────────────────────────────────────────────


@given(weight_matrix())
def test_quantization_stats_error_non_negative(w):
    codes, scale = ternarize_weight(w, granularity="channel")
    stats = quantization_stats("test", w, codes, scale, "channel")
    assert stats.relative_l2_error >= 0
    assert 0 <= stats.zero_fraction <= 1
    assert stats.name == "test"
    assert stats.shape == tuple(w.shape)
    assert stats.granularity == "channel"


# ── Boundary / edge cases ─────────────────────────────────────────────


def test_ternarize_1d_raises():
    w = torch.randn(16)
    with pytest.raises(ValueError, match="expected a 2D weight tensor"):
        ternarize_weight(w)


def test_ternarize_3d_raises():
    w = torch.randn(4, 8, 16)
    with pytest.raises(ValueError, match="expected a 2D weight tensor"):
        ternarize_weight(w)


def test_ternarize_unknown_granularity_raises():
    w = torch.randn(8, 16)
    with pytest.raises(ValueError, match="unknown granularity"):
        ternarize_weight(w, granularity="invalid")


def test_ternarize_group_no_size_raises():
    w = torch.randn(4, 16)
    with pytest.raises(ValueError, match="positive group_size"):
        _ = ternarize_weight(w, granularity="group")


def test_ternarize_group_bad_divisibility_raises():
    w = torch.randn(4, 15)
    with pytest.raises(ValueError, match="not divisible"):
        _ = ternarize_weight(w, granularity="group", group_size=4)


def test_zero_weight_produces_all_zeros():
    w = torch.zeros(8, 16)
    codes, scale = ternarize_weight(w, granularity="channel")
    assert (codes == 0).all()
    assert not torch.isnan(scale).any()
    assert not torch.isinf(scale).any()


def test_uniform_weight_produces_consistent_codes():
    w = torch.full((4, 8), 3.0)
    codes, scale = ternarize_weight(w, granularity="tensor")
    assert (codes == 1).all()
    assert (scale > 2.9).all()


def test_neg_uniform_weight():
    w = torch.full((4, 8), -3.0)
    codes, scale = ternarize_weight(w, granularity="tensor")
    assert (codes == -1).all()


def test_unpack_partial_recovery():
    """Only recover first N elements from a packed chunk with padding."""
    codes = torch.tensor([-1, 0, 1, -1, 0], dtype=torch.int8)
    packed = pack_ternary_2bit(codes)
    restored = unpack_ternary_2bit(packed, codes.numel())
    assert torch.equal(codes, restored)


def test_pack_empty():
    codes = torch.tensor([], dtype=torch.int8)
    packed = pack_ternary_2bit(codes)
    assert packed.numel() == 0
    restored = unpack_ternary_2bit(packed, 0)
    assert restored.numel() == 0


def test_ternarize_is_contractive_true():
    w = torch.randn(16, 32)
    assert ternarize_is_contractive(w)


def test_ternarize_is_contractive_group():
    w = torch.randn(8, 16)
    assert ternarize_is_contractive(w, granularity="group", group_size=4)


# ── Merkle manifest round-trip ────────────────────────────────────────


@given(weight_matrix(min_rows=1, max_rows=8, min_cols=4, max_cols=8))
def test_merkle_manifest_from_weight(w):
    """Build a simulated streaming manifest from a single weight, verify it."""
    codes, scale = ternarize_weight(w, granularity="channel")
    packed = pack_ternary_2bit(codes)
    tensors = {"test.packed": packed, "test.scale": scale.float()}
    shard_hash = build_shard_integrity_hash(tensors)

    manifest = {
        "shards": {
            "shard_0": {
                "status": "done",
                "sha256": shard_hash,
            },
        },
    }
    manifest["merkle_root"] = merkle_root([shard_hash])

    valid, errors = verify_manifest_integrity(manifest)
    assert valid, f"verification failed: {errors}"
    assert errors == []


def test_merkle_manifest_tampered_sha():
    """Simulate a tampered SHA in one shard."""
    manifest = {
        "shards": {
            "shard_0": {
                "status": "done",
                "sha256": "a" * 64,
            },
        },
        "merkle_root": merkle_root(["a" * 64]),
    }
    # Tamper the SHA of shard_0
    manifest["shards"]["shard_0"]["sha256"] = "b" * 64
    valid, errors = verify_manifest_integrity(manifest)
    assert not valid
    assert len(errors) > 0
