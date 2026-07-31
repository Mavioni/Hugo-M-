
import json

import pytest
import torch
from huggingface_hub.errors import EntryNotFoundError
from safetensors.torch import save_file

from hugo.stream_ternarize import aggregate_stats, load_manifest
from hugo.streaming import (
    DEFAULT_SKIP_SUBSTRINGS,
    copy_aux_files,
    is_quantizable,
    process_shard,
)


def test_is_quantizable_shape_and_suffix_rules():
    assert is_quantizable("model.layers.0.mlp.down_proj.weight", (16, 8), [])
    assert not is_quantizable("model.layers.0.input_layernorm.weight", (16,), [])  # 1D
    assert not is_quantizable("model.layers.0.mlp.down_proj.bias", (16, 8), [])  # not *.weight
    assert not is_quantizable("model.embed_tokens.weight", (16, 8), DEFAULT_SKIP_SUBSTRINGS)
    assert not is_quantizable("lm_head.weight", (16, 8), DEFAULT_SKIP_SUBSTRINGS)


def test_process_shard_against_a_local_fake_shard(tmp_path, monkeypatch):
    # Build a tiny fake safetensors shard mimicking a real checkpoint slice:
    # one quantizable Linear weight, one skipped embedding, one 1D norm.
    fake_shard = tmp_path / "src" / "model-00001-of-00001.safetensors"
    fake_shard.parent.mkdir(parents=True)
    torch.manual_seed(0)
    tensors = {
        "model.layers.0.mlp.down_proj.weight": torch.randn(8, 16),
        "model.embed_tokens.weight": torch.randn(4, 16),
        "model.layers.0.input_layernorm.weight": torch.ones(16),
    }
    save_file(tensors, str(fake_shard))

    def fake_hf_hub_download(repo_id, filename, revision=None, token=None, local_dir=None):
        # process_shard downloads into work_dir/<filename>; simulate that by
        # copying our fake shard there instead of hitting the network.
        dest = (local_dir or tmp_path) / filename
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(fake_shard.read_bytes())
        return str(dest)

    monkeypatch.setattr("hugo.streaming.hf_hub_download", fake_hf_hub_download)

    work_dir = tmp_path / "work"
    packed_dir = tmp_path / "packed"
    plain_dir = tmp_path / "plain"
    work_dir.mkdir()
    packed_dir.mkdir()
    plain_dir.mkdir()

    result = process_shard(
        repo_id="fake/repo",
        revision=None,
        token=None,
        shard_name="model-00001-of-00001.safetensors",
        tensor_names=list(tensors.keys()),
        work_dir=work_dir,
        packed_dir=packed_dir,
        plain_dir=plain_dir,
        shard_index=0,
        granularity="channel",
        group_size=None,
        skip_patterns=DEFAULT_SKIP_SUBSTRINGS,
    )

    # exactly one tensor was quantizable
    assert len(result.layer_stats) == 1
    assert result.layer_stats[0].name == "model.layers.0.mlp.down_proj.weight"
    assert result.manifest_entries["model.embed_tokens.weight"]["kind"] == "plain"
    assert result.manifest_entries["model.layers.0.input_layernorm.weight"]["kind"] == "plain"

    # the downloaded shard was deleted to reclaim disk
    assert not (work_dir / "model-00001-of-00001.safetensors").exists()

    # both sidecar files were actually written
    assert (packed_dir / "packed_shard_00000.safetensors").exists()
    assert (plain_dir / "plain_shard_00000.safetensors").exists()


def test_resolve_weight_map_falls_back_to_single_file(tmp_path, monkeypatch):
    from hugo.streaming import resolve_weight_map

    single_file = tmp_path / "model.safetensors"
    save_file({"a.weight": torch.randn(2, 2)}, str(single_file))

    def fake_hf_hub_download(repo_id, filename, revision=None, token=None):
        if filename == "model.safetensors.index.json":
            raise EntryNotFoundError("no index for this repo")
        assert filename == "model.safetensors"
        return str(single_file)

    monkeypatch.setattr("hugo.streaming.hf_hub_download", fake_hf_hub_download)

    weight_map = resolve_weight_map("fake/repo", None, None)
    assert weight_map == {"a.weight": "model.safetensors"}
def _done_shard(names_and_shapes, errors):
    return {
        "status": "done",
        "layer_stats": [
            {"name": n, "shape": list(s), "granularity": "channel",
             "relative_l2_error": e, "zero_fraction": 0.3}
            for (n, s), e in zip(names_and_shapes, errors, strict=False)
        ],
    }


def test_aggregate_stats_includes_shards_from_earlier_runs():
    # Regression: totals used to be computed only from shards the current
    # process handled, so a resumed run under-reported everything an earlier
    # invocation had already finished.
    manifest = {"shards": {
        "shard-A": _done_shard([("a", (2, 4))], [0.4]),   # processed by an earlier run
        "shard-B": _done_shard([("b", (2, 4))], [0.6]),   # processed by this run
    }}

    stats = aggregate_stats(manifest)

    assert stats["num_quantized_layers"] == 2  # not 1
    assert stats["total_quantized_elements"] == 16
    assert stats["avg_relative_l2_error"] == pytest.approx(0.5)
    assert stats["worst_layer"] == "b"


def test_aggregate_stats_ignores_unfinished_shards():
    manifest = {"shards": {
        "shard-A": _done_shard([("a", (2, 4))], [0.4]),
        "shard-B": {"status": "in_progress"},
    }}
    assert aggregate_stats(manifest)["num_quantized_layers"] == 1


def test_aggregate_stats_returns_none_when_nothing_done():
    assert aggregate_stats({"shards": {}}) is None


def test_resume_rejects_changed_settings(tmp_path):
    # Resuming with different quantization settings would skip already-done
    # shards and process the rest differently, yielding a checkpoint whose
    # shards disagree with each other and with the manifest.
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({
        "repo_id": "some/model", "revision": None, "granularity": "channel",
        "group_size": None, "skip_patterns": ["lm_head"], "shards": {}, "stats": {},
    }))

    # Same settings -> resumes fine.
    assert load_manifest(path, "some/model", None, "channel", None, ["lm_head"])["repo_id"] == "some/model"

    # Changed granularity -> refused, and the message names the offending field.
    with pytest.raises(SystemExit) as excinfo:
        load_manifest(path, "some/model", None, "group", 128, ["lm_head"])
    assert "granularity" in str(excinfo.value)

    # Changed skip patterns -> also refused.
    with pytest.raises(SystemExit):
        load_manifest(path, "some/model", None, "channel", None, [])

    # Different model -> still refused (the original check).
    with pytest.raises(SystemExit):
        load_manifest(path, "other/model", None, "channel", None, ["lm_head"])


def test_copy_aux_files_keeps_custom_code_and_nested_layout(tmp_path, monkeypatch):
    # Custom architectures -- exactly what the streaming path exists for --
    # ship modeling_*.py that config.json's auto_map points at. Dropping .py
    # files or flattening subfolders makes the output unloadable.
    repo_files = [
        "config.json",
        "modeling_custom.py",
        "nested/configuration_custom.py",
        "model-00001-of-00001.safetensors",   # weight -> excluded
        "model.safetensors.index.json",       # index -> excluded
    ]
    monkeypatch.setattr("hugo.streaming.list_repo_files", lambda *a, **k: repo_files)

    def fake_download(repo_id, filename, revision=None, token=None):
        src = tmp_path / "src" / filename
        src.parent.mkdir(parents=True, exist_ok=True)
        src.write_text(f"content of {filename}")
        return str(src)

    monkeypatch.setattr("hugo.streaming.hf_hub_download", fake_download)

    out = tmp_path / "out"
    out.mkdir()
    copied = copy_aux_files("fake/repo", None, None, out)

    assert "modeling_custom.py" in copied
    assert "nested/configuration_custom.py" in copied
    assert (out / "nested" / "configuration_custom.py").exists()  # layout preserved
    assert "model-00001-of-00001.safetensors" not in copied
    assert "model.safetensors.index.json" not in copied


def test_copy_aux_files_refuses_paths_escaping_output_dir(tmp_path, monkeypatch):
    monkeypatch.setattr("hugo.streaming.list_repo_files", lambda *a, **k: ["../escaped.json"])
    monkeypatch.setattr("hugo.streaming.hf_hub_download",
                        lambda *a, **k: pytest.fail("should not download a traversing path"))

    out = tmp_path / "out"
    out.mkdir()
    assert copy_aux_files("fake/repo", None, None, out) == []
    assert not (tmp_path / "escaped.json").exists()
