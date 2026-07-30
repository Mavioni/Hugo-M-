import torch
from torch import nn

from hugo.qat import (
    BitLinear,
    bake_bitlinear_to_linear,
    convert_to_bitlinear,
    ternary_fake_quant,
)
from hugo.quantize import dequantize_weight, ternarize_weight


def test_ternary_fake_quant_matches_ptq_values():
    torch.manual_seed(0)
    w = torch.randn(8, 16)
    fake = ternary_fake_quant(w, granularity="channel")

    codes, scale = ternarize_weight(w, granularity="channel")
    expected = dequantize_weight(codes, scale)
    assert torch.allclose(fake, expected)


def test_ste_gradient_passes_through_in_range_and_blocks_saturated():
    # With "tensor" granularity, scale = mean(|w|) over the whole tensor.
    # The huge outlier dominates that mean, so every OTHER entry normalizes
    # to something tiny (well within the clamp -> gradient passes), while
    # the outlier itself normalizes to something >> 1 (clamp saturates ->
    # gradient blocked, matching clamp's real, zero, gradient there).
    w = torch.tensor([[0.1, -0.1], [100.0, -0.1]], requires_grad=True)
    scale = w.detach().abs().mean()
    normalized = w.detach() / scale
    expected_in_range = normalized.abs() <= 1
    assert expected_in_range.tolist() == [[True, True], [False, True]]  # sanity-check the setup itself

    out = ternary_fake_quant(w, granularity="tensor")
    out.sum().backward()

    assert w.grad is not None
    assert torch.equal(w.grad != 0, expected_in_range)


def test_bitlinear_forward_shape_and_ternary_values():
    torch.manual_seed(0)
    layer = BitLinear(16, 8, bias=True)
    x = torch.randn(4, 16)
    out = layer(x)
    assert out.shape == (4, 8)

    w_q = ternary_fake_quant(layer.weight, layer.granularity, layer.group_size)
    for row in w_q:
        assert row.unique().numel() <= 3


def test_bitlinear_from_linear_shares_parameters_not_copies():
    linear = nn.Linear(16, 8)
    original_weight = linear.weight
    bl = BitLinear.from_linear(linear)
    assert bl.weight is original_weight  # same Parameter object, not a clone


def test_convert_to_bitlinear_respects_skip_patterns():
    model = nn.Sequential()
    model.add_module("layer1", nn.Linear(16, 8))
    model.add_module("lm_head", nn.Linear(8, 4))

    replaced = convert_to_bitlinear(model, skip_patterns=["lm_head"])

    assert replaced == ["layer1"]
    assert isinstance(model.layer1, BitLinear)
    assert isinstance(model.lm_head, nn.Linear) and not isinstance(model.lm_head, BitLinear)


def test_training_step_reduces_loss_on_a_toy_regression():
    # This is the real behavioral claim of this module: gradients flowing
    # through the STE should actually let an optimizer improve the model,
    # not just pass shape/type checks.
    torch.manual_seed(0)
    model = nn.Sequential(BitLinear(4, 4, bias=False))
    target = torch.randn(2, 4)
    x = torch.randn(2, 4)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

    def loss_fn():
        return nn.functional.mse_loss(model(x), target)

    loss_before = loss_fn().item()
    for _ in range(20):
        optimizer.zero_grad()
        loss = loss_fn()
        loss.backward()
        optimizer.step()
    loss_after = loss_fn().item()

    assert loss_after < loss_before


def test_bake_bitlinear_to_linear_produces_plain_linear_with_ternary_weights():
    model = nn.Sequential(BitLinear(16, 8, bias=False))
    baked = bake_bitlinear_to_linear(model)

    assert baked == ["0"]
    assert isinstance(model[0], nn.Linear) and not isinstance(model[0], BitLinear)
    for row in model[0].weight.data:
        assert row.unique().numel() <= 3
    # baked weight should require no grad tracking surprises: it's a fresh leaf Parameter
    assert model[0].weight.requires_grad
