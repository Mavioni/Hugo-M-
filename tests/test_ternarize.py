import json

import torch
from torch import nn

from hugo.ternarize import (
    build_packed_sidecar,
    dataclasses_asdict,
    human_bytes,
    main,
    parse_args,
    summarize,
)


class FakeTokenizer:
    def save_pretrained(self, path):
        pass


def _tiny_model():
    return nn.Sequential(
        nn.Linear(16, 8, bias=False),
        nn.Linear(8, 4, bias=False),
    )


# ---------------------------------------------------------------------------
# pure helpers
# ---------------------------------------------------------------------------

def test_parse_args_minimal():
    args = parse_args(["--model", "org/repo", "--output", "./out"])
    assert args.model == "org/repo"
    assert args.output == "./out"
    assert args.granularity == "channel"
    assert not args.pack
    assert not args.dry_run


def test_parse_args_all_flags():
    args = parse_args([
        "--model", "org/repo", "--output", "./out",
        "--granularity", "group", "--group-size", "64",
        "--pack", "--dry-run", "--trust-remote-code",
        "--skip", "head,norm", "--revision", "dev",
    ])
    assert args.granularity == "group"
    assert args.group_size == 64
    assert args.pack
    assert args.dry_run
    assert args.trust_remote_code
    assert args.skip == "head,norm"
    assert args.revision == "dev"


def test_human_bytes():
    assert human_bytes(0) == "0.00B"
    assert human_bytes(500) == "500.00B"
    assert human_bytes(2048) == "2.00KB"
    assert human_bytes(5 * 1024 ** 3) == "5.00GB"


def test_dataclasses_asdict():
    from hugo.quantize import LayerQuantStats
    s = LayerQuantStats(name="x", shape=(4, 4), granularity="channel",
                        relative_l2_error=0.5, zero_fraction=0.3)
    d = dataclasses_asdict(s)
    assert d["name"] == "x"
    assert d["relative_l2_error"] == 0.5


def test_summarize_no_layers(capsys):
    summarize([])
    assert "No nn.Linear" in capsys.readouterr().out


def test_build_packed_sidecar_produces_correct_structure():
    model = _tiny_model()
    codes = torch.tensor([[1, 0, -1, 0]], dtype=torch.int8)
    scale = torch.tensor([[0.5]])
    quantized = {"0": (codes, scale)}
    # patch shapes lookup in build_packed_sidecar via a small helper
    packed_tensors, manifest = build_packed_sidecar(model, quantized, "channel", None)

    assert "0.packed" in packed_tensors
    assert "0.scale" in packed_tensors
    assert len(manifest["layers"]) == 1
    assert manifest["layers"]["0"]["shape"] == [8, 16]


# ---------------------------------------------------------------------------
# main() with injected deps — dry run
# ---------------------------------------------------------------------------

def test_main_dry_run_loads_model_and_quantizes(capsys):
    model = _tiny_model()
    tokenizer = FakeTokenizer()

    def fake_load(model_id, revision, dtype, trust_remote_code):
        return model, tokenizer

    rc = main(
        ["--model", "org/repo", "--output", "./out", "--dry-run"],
        _load_fn=fake_load,
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "Dry run" in out
    assert "Quantized 2 Linear layers" in out


def test_main_group_requires_group_size(capsys):
    rc = main(
        ["--model", "x", "--output", "y", "--granularity", "group"],
        _load_fn=lambda *a, **k: (_tiny_model(), FakeTokenizer()),
    )
    assert rc == 2
    assert "group-size" in capsys.readouterr().err


def test_main_save_checkpoint_is_called(capsys, tmp_path):
    model = _tiny_model()
    tokenizer = FakeTokenizer()
    save_calls = []

    def fake_load(*a, **k):
        return model, tokenizer

    def fake_save(m, tok, out_dir, max_shard):
        save_calls.append((out_dir, max_shard))

    out = str(tmp_path / "out")
    rc = main(
        ["--model", "x", "--output", out],
        _load_fn=fake_load,
        _save_fn=fake_save,
    )
    assert rc == 0
    assert len(save_calls) == 1
    assert save_calls[0][0].name == "out"


def test_main_pack_calls_pack_save_fn(capsys, tmp_path):
    model = _tiny_model()
    tokenizer = FakeTokenizer()
    pack_calls = []

    def fake_load(*a, **k):
        return model, tokenizer

    def fake_save(m, tok, out_dir, max_shard):
        pass

    def fake_pack_save(packed, manifest, pack_dir):
        pack_calls.append((len(packed), pack_dir))

    out = str(tmp_path / "out")
    rc = main(
        ["--model", "x", "--output", out, "--pack"],
        _load_fn=fake_load,
        _save_fn=fake_save,
        _pack_save_fn=fake_pack_save,
    )
    assert rc == 0
    assert len(pack_calls) == 1
    assert pack_calls[0][0] > 0  # at least one (packed, scale) pair


def test_main_skip_pattern_whitelists_modules(capsys):
    model = _tiny_model()
    tokenizer = FakeTokenizer()

    def fake_load(*a, **k):
        return model, tokenizer

    rc = main(
        ["--model", "x", "--output", "/tmp/out", "--skip", "0", "--dry-run"],
        _load_fn=fake_load,
    )
    assert rc == 0
    out = capsys.readouterr().out
    # "0" matches the first linear layer's name -> it is skipped
    assert "Quantized 1 Linear layers" in out


def test_main_writes_stats_json(capsys, tmp_path):
    model = _tiny_model()
    tokenizer = FakeTokenizer()

    def fake_load(*a, **k):
        return model, tokenizer

    def fake_save(m, tok, out_dir, max_shard):
        pass

    out = str(tmp_path / "out")
    rc = main(
        ["--model", "x", "--output", out],
        _load_fn=fake_load,
        _save_fn=fake_save,
    )
    assert rc == 0
    stats_path = tmp_path / "out" / "hugo_stats.json"
    assert stats_path.exists()
    stats_data = json.loads(stats_path.read_text())
    assert len(stats_data) == 2
    assert "relative_l2_error" in stats_data[0]
