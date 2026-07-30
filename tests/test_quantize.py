import torch
from torch import nn

from hugo.quantize import (
    dequantize_weight,
    pack_ternary_2bit,
    quantize_linear_modules,
    ternarize_weight,
    unpack_ternary_2bit,
)


def test_ternarize_values_are_ternary():
    w = torch.randn(8, 16)
    codes, scale = ternarize_weight(w, granularity="tensor")
    assert set(codes.unique().tolist()) <= {-1, 0, 1}
    assert scale.numel() == 1


def test_channel_granularity_shapes():
    w = torch.randn(8, 16)
    codes, scale = ternarize_weight(w, granularity="channel")
    assert codes.shape == w.shape
    assert scale.shape == (8, 1)


def test_group_granularity_shapes_and_roundtrip():
    w = torch.randn(4, 16)
    codes, scale = ternarize_weight(w, granularity="group", group_size=4)
    assert scale.shape == (4, 4, 1)
    recon = dequantize_weight(codes, scale, group_size=4)
    assert recon.shape == w.shape


def test_pack_unpack_roundtrip():
    codes = torch.tensor([-1, 0, 1, -1, 0, 0, 1, 1, -1], dtype=torch.int8)
    packed = pack_ternary_2bit(codes)
    restored = unpack_ternary_2bit(packed, codes.numel())
    assert torch.equal(codes, restored)


def test_dequantize_matches_scale_times_codes():
    w = torch.tensor([[3.0, -3.0, 0.1, -0.1]])
    codes, scale = ternarize_weight(w, granularity="tensor")
    recon = dequantize_weight(codes, scale)
    expected = codes.float() * scale
    assert torch.allclose(recon, expected)


def test_quantize_linear_modules_in_place_and_skip():
    model = nn.Sequential()
    model.add_module("layer1", nn.Linear(16, 8, bias=False))
    model.add_module("lm_head", nn.Linear(8, 4, bias=False))
    original_head = model.lm_head.weight.data.clone()
    original_layer1 = model.layer1.weight.data.clone()

    stats, quantized = quantize_linear_modules(model, granularity="channel", skip_patterns=["lm_head"])

    assert len(stats) == 1
    assert stats[0].name == "layer1"
    assert set(quantized.keys()) == {"layer1"}
    # lm_head was skipped -> untouched
    assert torch.equal(model.lm_head.weight.data, original_head)
    # layer1 weights are now exactly the dequantized ternary codes derived from the original weights
    codes, scale = ternarize_weight(original_layer1, granularity="channel")
    assert torch.allclose(model.layer1.weight.data, dequantize_weight(codes, scale))
    # and each row now has at most 3 distinct values: {-s, 0, s}
    for row in model.layer1.weight.data:
        assert row.unique().numel() <= 3


def test_relative_error_decreases_with_finer_granularity():
    torch.manual_seed(0)
    w = torch.randn(32, 64) * torch.linspace(0.1, 5, 32).unsqueeze(1)  # rows with varied scale

    from hugo.quantize import quantization_stats

    codes_t, scale_t = ternarize_weight(w, granularity="tensor")
    codes_c, scale_c = ternarize_weight(w, granularity="channel")

    err_tensor = quantization_stats("w", w, codes_t, scale_t, "tensor").relative_l2_error
    err_channel = quantization_stats("w", w, codes_c, scale_c, "channel").relative_l2_error

    assert err_channel < err_tensor
