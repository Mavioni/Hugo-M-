import json

import pytest

from hugo.push_to_hub import _resolve_token, build_model_card


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
