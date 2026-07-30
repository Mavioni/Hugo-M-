
import torch
from safetensors.torch import save_file

from hugo.quantize import dequantize_weight, ternarize_weight
from hugo.reconstruct import reconstruct_shard
from hugo.streaming import DEFAULT_SKIP_SUBSTRINGS, process_shard


def test_reconstruct_shard_matches_original_quantization(tmp_path, monkeypatch):
    fake_shard = tmp_path / "src" / "model-00001-of-00001.safetensors"
    fake_shard.parent.mkdir(parents=True)
    torch.manual_seed(1)
    tensors = {
        "model.layers.0.mlp.down_proj.weight": torch.randn(8, 16),
        "model.embed_tokens.weight": torch.randn(4, 16),
    }
    save_file(tensors, str(fake_shard))

    def fake_hf_hub_download(repo_id, filename, revision=None, token=None, local_dir=None):
        dest = (local_dir or tmp_path) / filename
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(fake_shard.read_bytes())
        return str(dest)

    monkeypatch.setattr("hugo.streaming.hf_hub_download", fake_hf_hub_download)

    work_dir, packed_dir, plain_dir = tmp_path / "work", tmp_path / "packed", tmp_path / "plain"
    for d in (work_dir, packed_dir, plain_dir):
        d.mkdir()

    result = process_shard(
        repo_id="fake/repo", revision=None, token=None,
        shard_name="model-00001-of-00001.safetensors",
        tensor_names=list(tensors.keys()),
        work_dir=work_dir, packed_dir=packed_dir, plain_dir=plain_dir,
        shard_index=0, granularity="channel", group_size=None,
        skip_patterns=DEFAULT_SKIP_SUBSTRINGS,
    )

    input_dir = tmp_path  # packed_dir/plain_dir referenced relative to this in shard_entry
    shard_entry = {
        "packed_file": str(packed_dir.relative_to(input_dir) / "packed_shard_00000.safetensors"),
        "plain_file": str(plain_dir.relative_to(input_dir) / "plain_shard_00000.safetensors"),
        "tensors": result.manifest_entries,
    }

    rebuilt = reconstruct_shard(input_dir, shard_entry, dtype=torch.float32)

    # quantized tensor: rebuilt value must equal codes*scale computed from the ORIGINAL weight
    codes, scale = ternarize_weight(tensors["model.layers.0.mlp.down_proj.weight"], granularity="channel")
    expected = dequantize_weight(codes, scale)
    assert torch.allclose(rebuilt["model.layers.0.mlp.down_proj.weight"], expected)

    # plain tensor: rebuilt value must be exactly the original
    assert torch.equal(rebuilt["model.embed_tokens.weight"], tensors["model.embed_tokens.weight"])


def test_reconstruct_shard_handles_group_granularity(tmp_path, monkeypatch):
    # Regression: reconstruct_shard used to call dequantize_weight without the
    # manifest's group_size. Group scales are 3D, so dequantize_weight then hit
    # `in_features // None` and every grouped checkpoint failed to reconstruct.
    fake_shard = tmp_path / "src" / "model-00001-of-00001.safetensors"
    fake_shard.parent.mkdir(parents=True)
    torch.manual_seed(3)
    original = torch.randn(4, 16)
    save_file({"model.layers.0.mlp.down_proj.weight": original}, str(fake_shard))

    def fake_hf_hub_download(repo_id, filename, revision=None, token=None, local_dir=None):
        dest = (local_dir or tmp_path) / filename
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(fake_shard.read_bytes())
        return str(dest)

    monkeypatch.setattr("hugo.streaming.hf_hub_download", fake_hf_hub_download)

    work_dir, packed_dir, plain_dir = tmp_path / "w", tmp_path / "p", tmp_path / "pl"
    for d in (work_dir, packed_dir, plain_dir):
        d.mkdir()

    result = process_shard(
        repo_id="fake/repo", revision=None, token=None,
        shard_name="model-00001-of-00001.safetensors",
        tensor_names=["model.layers.0.mlp.down_proj.weight"],
        work_dir=work_dir, packed_dir=packed_dir, plain_dir=plain_dir,
        shard_index=0, granularity="group", group_size=4,
        skip_patterns=DEFAULT_SKIP_SUBSTRINGS,
    )

    shard_entry = {
        "packed_file": str(packed_dir.relative_to(tmp_path) / "packed_shard_00000.safetensors"),
        "plain_file": None,
        "tensors": result.manifest_entries,
    }

    rebuilt = reconstruct_shard(tmp_path, shard_entry, dtype=torch.float32, group_size=4)

    codes, scale = ternarize_weight(original, granularity="group", group_size=4)
    expected = dequantize_weight(codes, scale, group_size=4)
    assert torch.allclose(rebuilt["model.layers.0.mlp.down_proj.weight"], expected)
