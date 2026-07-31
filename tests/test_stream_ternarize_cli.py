import json

from hugo.stream_ternarize import main, parse_args


def test_parse_args_minimal():
    args = parse_args(["--model", "org/repo", "--output", "./out"])
    assert args.model == "org/repo"
    assert args.output == "./out"
    assert args.granularity == "channel"


def test_main_runs_pipeline_with_injected_deps(tmp_path):

    from hugo.quantize import LayerQuantStats
    from hugo.streaming import ShardResult

    copy_calls = []
    resolve_calls = []
    process_calls = []
    cleanup_calls = []

    def fake_copy_aux(model, revision, token, output_dir):
        copy_calls.append(model)
        return ["config.json"]

    def fake_resolve_map(model, revision, token):
        resolve_calls.append(model)
        return {"a.weight": "shard-01.safetensors", "b.weight": "shard-01.safetensors"}

    def fake_process_shard(**kwargs):
        process_calls.append(kwargs["shard_name"])
        return ShardResult(
            shard_name=kwargs["shard_name"],
            manifest_entries={
                "a.weight": {"kind": "quantized", "shape": [4, 4], "packed_key": "a.packed",
                             "scale_key": "a.scale", "scale_shape": [4, 1]},
            },
            layer_stats=[LayerQuantStats(
                name="a.weight", shape=(4, 4), granularity="channel",
                relative_l2_error=0.1, zero_fraction=0.3,
            )],
            packed_file="p/shard.safetensors",
            plain_file=None,
        )

    def fake_cleanup(path, ignore_errors=False):
        cleanup_calls.append(str(path))

    out_dir = tmp_path / "out"
    rc = main(
        ["--model", "org/repo", "--output", str(out_dir)],
        _copy_aux_fn=fake_copy_aux,
        _resolve_map_fn=fake_resolve_map,
        _process_shard_fn=fake_process_shard,
        _cleanup_fn=fake_cleanup,
    )
    assert rc == 0
    assert len(copy_calls) == 1
    assert len(resolve_calls) == 1
    assert len(process_calls) == 1
    assert len(cleanup_calls) == 1  # all shards done -> cleanup

    # manifest was written
    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert manifest["repo_id"] == "org/repo"
    assert "stats" in manifest
    assert manifest["stats"]["num_quantized_layers"] == 1


def test_main_rejects_group_without_group_size(capsys):
    rc = main(
        ["--model", "x", "--output", "y", "--granularity", "group"],
        _copy_aux_fn=lambda *a, **k: [],
        _resolve_map_fn=lambda *a, **k: {},
    )
    assert rc == 2
    assert "group-size" in capsys.readouterr().err


def test_main_skips_already_done_shards(tmp_path):

    resolve_calls = []

    def fake_copy_aux(*a, **k):
        return []

    def fake_resolve_map(model, revision, token):
        resolve_calls.append(model)
        return {"a.weight": "shard-01.safetensors"}

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    # Pre-populate the manifest with a "done" shard
    initial_manifest = {
        "repo_id": "org/repo",
        "revision": None,
        "granularity": "channel",
        "group_size": None,
        "skip_patterns": [],
        "shards": {
            "shard-01.safetensors": {
                "status": "done",
                "packed_file": None, "plain_file": None,
                "tensors": {}, "layer_stats": [],
            }
        },
        "stats": {},
    }
    (out_dir / "manifest.json").write_text(json.dumps(initial_manifest))

    rc = main(
        ["--model", "org/repo", "--output", str(out_dir), "--skip", ""],
        _copy_aux_fn=fake_copy_aux,
        _resolve_map_fn=fake_resolve_map,
        _process_shard_fn=fake_process_shard_not_called,
    )
    assert rc == 0
    # If fake_process_shard_not_called were invoked, the test would fail with AssertionError


def fake_process_shard_not_called(**kwargs):
    raise AssertionError("process_shard should not be called for a done shard")


def test_main_does_not_cleanup_when_shards_remain(tmp_path):
    from hugo.streaming import ShardResult

    resolve_calls = []

    def fake_copy_aux(*a, **k):
        return []

    def fake_resolve_map(model, revision, token):
        resolve_calls.append(model)
        return {"a.weight": "shard-01.safetensors", "b.weight": "shard-02.safetensors"}

    out_dir = tmp_path / "out"
    out_dir.mkdir()

    def fake_process(**k):
        return ShardResult(
            shard_name="x", manifest_entries={}, layer_stats=[], packed_file=None, plain_file=None,
        )

    rc = main(
        ["--model", "org/repo", "--output", str(out_dir), "--max-shards", "1"],
        _copy_aux_fn=fake_copy_aux,
        _resolve_map_fn=fake_resolve_map,
        _process_shard_fn=fake_process,
    )
    assert rc == 0
