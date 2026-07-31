import json

import torch
from safetensors.torch import save_file as safetensors_save

from hugo.reconstruct import _copy_aux_to_output, main, parse_args


def test_parse_args_minimal():
    args = parse_args(["--input", "./in", "--output", "./out"])
    assert args.input == "./in"
    assert args.output == "./out"
    assert args.dtype == "bfloat16"


def test_main_reconstructs_a_complete_pipeline(tmp_path):
    # Build a minimal streaming output structure and verify main() reconstructs it.
    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    in_dir.mkdir()
    packed_dir = in_dir / "ternary_packed"
    plain_dir = in_dir / "plain_tensors"
    packed_dir.mkdir()
    plain_dir.mkdir()

    # One quantized tensor + one plain tensor
    weight = torch.randn(4, 8)
    from hugo.quantize import pack_ternary_2bit, ternarize_weight

    codes, scale = ternarize_weight(weight, granularity="channel")
    packed = pack_ternary_2bit(codes)

    safetensors_save(
        {"model.layer.weight__packed": packed, "model.layer.weight__scale": scale.float()},
        str(packed_dir / "packed_shard_00000.safetensors"),
    )
    safetensors_save(
        {"model.embed.weight": torch.randn(2, 8)},
        str(plain_dir / "plain_shard_00000.safetensors"),
    )
    # Add an aux file that should be copied
    (in_dir / "config.json").write_text('{"key": "val"}')

    manifest = {
        "repo_id": "test/model",
        "revision": None,
        "granularity": "channel",
        "group_size": None,
        "skip_patterns": [],
        "shards": {
            "model-00001-of-00001.safetensors": {
                "status": "done",
                "packed_file": "ternary_packed/packed_shard_00000.safetensors",
                "plain_file": "plain_tensors/plain_shard_00000.safetensors",
                "tensors": {
                    "model.layer.weight": {
                        "kind": "quantized",
                        "shape": [4, 8],
                        "packed_key": "model.layer.weight__packed",
                        "scale_key": "model.layer.weight__scale",
                        "scale_shape": [4, 1],
                    },
                    "model.embed.weight": {
                        "kind": "plain",
                        "shape": [2, 8],
                        "dtype": "torch.float32",
                    },
                },
            }
        },
        "stats": {},
    }
    (in_dir / "manifest.json").write_text(json.dumps(manifest))

    rc = main(["--input", str(in_dir), "--output", str(out_dir)])
    assert rc == 0

    # The reconstructed shard was written
    assert (out_dir / "model-00001-of-00001.safetensors").exists()
    assert (out_dir / "model.safetensors.index.json").exists()
    assert (out_dir / "config.json").exists()

    # Verify the index is well-formed
    index = json.loads((out_dir / "model.safetensors.index.json").read_text())
    assert "weight_map" in index
    assert "model.layer.weight" in index["weight_map"]
    assert "model.embed.weight" in index["weight_map"]


def test_main_rejects_unfinished_shards(tmp_path, capsys):
    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    in_dir.mkdir()
    manifest = {
        "shards": {
            "shard-1.safetensors": {"status": "in_progress"},
        }
    }
    (in_dir / "manifest.json").write_text(json.dumps(manifest))

    import pytest
    with pytest.raises(SystemExit):
        main(["--input", str(in_dir), "--output", str(out_dir)])


def test_copy_aux_to_output_skips_reserved_dirs(tmp_path):
    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    in_dir.mkdir()
    out_dir.mkdir()

    # Create files in directories that should be skipped
    (in_dir / "ternary_packed").mkdir(parents=True, exist_ok=True)
    (in_dir / "ternary_packed" / "x.safetensors").write_text("packed")
    (in_dir / "plain_tensors").mkdir(parents=True, exist_ok=True)
    (in_dir / "plain_tensors" / "y.safetensors").write_text("plain")
    # A file that should be copied
    (in_dir / "config.json").write_text('{"a":1}')

    copied = []

    def record_copy(src, dst):
        copied.append((str(src), str(dst)))

    _copy_aux_to_output(in_dir, out_dir, copy_fn=record_copy)

    assert len(copied) == 1
    assert "config.json" in copied[0][0]
    assert not any("ternary_packed" in src for src, _ in copied)
    assert not any("plain_tensors" in src for src, _ in copied)
