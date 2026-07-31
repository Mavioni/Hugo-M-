import json

import pytest

from hugo.push_to_hub import (
    _resolve_token,
    build_model_card,
    main,
    push_checkpoint,
)


def test_resolve_token_prefers_env(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "tok_from_env")
    assert _resolve_token() == "tok_from_env"


def test_resolve_token_exits_cleanly_when_no_credentials(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    # Also neutralize any cached CLI login so this doesn't pass/fail based on
    # the machine it runs on.
    monkeypatch.setattr("huggingface_hub.get_token", lambda: None)

    with pytest.raises(SystemExit) as excinfo:
        _resolve_token()
    assert "HF_TOKEN" in str(excinfo.value)


def test_model_card_reports_qat_provenance(tmp_path):
    (tmp_path / "hugo_qat_run.json").write_text(json.dumps({
        "source_model": "some/model",
        "granularity": "channel",
        "group_size": None,
        "skip_patterns": ["lm_head"],
        "converted_layers": 14,
        "optimizer_steps": 20,
        "lr": 0.001,
        "batch_size": 2,
        "grad_accum": 1,
        "dataset": "Salesforce/wikitext/wikitext-2-raw-v1:train",
        "first_loss": 20.3,
        "mean_final_loss": 19.6,
    }))

    card = build_model_card(tmp_path, "user/Hugo-test")

    assert "quantization-aware training" in card
    assert "some/model" in card
    assert "straight-through estimator" in card
    # A QAT card must not claim to be PTQ.
    assert "quantized after training" not in card


def test_model_card_is_honest_about_ptq_provenance(tmp_path):
    (tmp_path / "ternary_quant_stats.json").write_text(json.dumps([
        {"name": "a", "shape": [4, 4], "granularity": "channel",
         "relative_l2_error": 0.5, "zero_fraction": 0.3},
        {"name": "b", "shape": [4, 4], "granularity": "channel",
         "relative_l2_error": 0.7, "zero_fraction": 0.3},
    ]))

    card = build_model_card(tmp_path, "user/Hugo-test")

    assert "post-training quantization" in card
    assert "quantized after training, not trained to be" in card
    assert "0.6000" in card  # mean of 0.5 and 0.7, surfaced so users see the real error


def test_model_card_says_so_when_provenance_is_unknown(tmp_path):
    card = build_model_card(tmp_path, "user/Hugo-test")
    assert "could not be recorded automatically" in card


class FakeApi:
    def __init__(self, token):
        self.token = token
        self.created_repo = None
        self.uploaded_folder = None
        self.uploaded_repo_id = None

    def create_repo(self, repo_id, private=False, exist_ok=True, repo_type="model"):
        self.created_repo = repo_id

    def upload_folder(self, folder_path, repo_id, repo_type="model", ignore_patterns=None):
        self.uploaded_folder = folder_path
        self.uploaded_repo_id = repo_id


def test_push_checkpoint_calls_api(tmp_path):
    (tmp_path / "hugo_qat_run.json").write_text(json.dumps({
        "source_model": "some/model", "granularity": "channel",
        "group_size": None, "skip_patterns": [],
        "converted_layers": 2, "optimizer_steps": 10,
        "lr": 0.001, "batch_size": 1, "grad_accum": 1,
        "dataset": "x", "first_loss": 10.0, "mean_final_loss": 9.5,
    }))

    def fake_token():
        return "fake-token"

    def fake_api_factory(token):
        api = FakeApi(token)
        assert token == "fake-token"
        return api

    url = push_checkpoint(
        tmp_path, "user/test-repo",
        _token_fn=fake_token,
        _api_factory=fake_api_factory,
    )
    assert url == "https://huggingface.co/user/test-repo"


def test_push_checkpoint_skips_model_card_when_readme_exists(tmp_path):
    (tmp_path / "README.md").write_text("existing card")

    class SpyApi:
        def create_repo(self, *a, **k): pass
        def upload_folder(self, *a, **k): pass

    push_checkpoint(
        tmp_path, "user/r", write_model_card=True,
        _token_fn=lambda: "t", _api_factory=lambda t: SpyApi(),
    )
    assert "existing card" in (tmp_path / "README.md").read_text()


def test_push_checkpoint_writes_model_card_when_missing(tmp_path):
    (tmp_path / "hugo_qat_run.json").write_text(json.dumps({
        "source_model": "s", "granularity": "channel",
        "group_size": None, "skip_patterns": [],
        "converted_layers": 1, "optimizer_steps": 1,
        "lr": 0.001, "batch_size": 1, "grad_accum": 1,
        "dataset": "x", "first_loss": 1.0, "mean_final_loss": 0.4,
    }))

    class SpyApi:
        def create_repo(self, *a, **k): pass
        def upload_folder(self, *a, **k): pass

    push_checkpoint(
        tmp_path, "user/r", write_model_card=True,
        _token_fn=lambda: "t", _api_factory=lambda t: SpyApi(),
    )
    card = (tmp_path / "README.md").read_text()
    assert "quantization-aware training" in card


def test_main_dry_run_with_model_card(tmp_path, capsys):
    (tmp_path / "hugo_qat_run.json").write_text(json.dumps({
        "source_model": "s", "granularity": "channel",
        "group_size": None, "skip_patterns": [],
        "converted_layers": 1, "optimizer_steps": 1,
        "lr": 0.001, "batch_size": 1, "grad_accum": 1,
        "dataset": "x", "first_loss": 1.0, "mean_final_loss": 0.4,
    }))

    rc = main(
        ["--checkpoint", str(tmp_path), "--repo-id", "u/r", "--dry-run"],
        _token_fn=lambda: "t",
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "Credentials: found." in out
    assert "generated model card" in out


def test_main_calls_push_fn(tmp_path):
    push_calls = []

    def fake_push(checkpoint_dir, repo_id, private=False, write_model_card=True,
                   _token_fn=None, _api_factory=None):
        push_calls.append((repo_id, private, write_model_card))
        return "https://fake.url"

    rc = main(
        ["--checkpoint", str(tmp_path), "--repo-id", "u/r", "--private"],
        _token_fn=lambda: "t",
        _push_fn=fake_push,
    )
    assert rc == 0
    assert len(push_calls) == 1
    assert push_calls[0] == ("u/r", True, True)
