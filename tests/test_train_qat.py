import torch
from torch import nn
from torch.utils.data import DataLoader

from hugo.train_qat import _resolve_device, main, parse_args


def test_parse_args_minimal():
    args = parse_args(["--model", "org/repo", "--output", "./out"])
    assert args.model == "org/repo"
    assert args.output == "./out"
    assert args.granularity == "channel"
    assert args.lr == 1e-5


def test_parse_args_full():
    args = parse_args(
        "--model org/repo --output ./out --granularity group --group-size 64 "
        "--bf16 --max-steps 50 --limit-samples 200 --lr 1e-3 --push-to-hub u/r --private".split()
    )
    assert args.granularity == "group"
    assert args.group_size == 64
    assert args.bf16
    assert args.max_steps == 50
    assert args.limit_samples == 200
    assert args.lr == 1e-3
    assert args.push_to_hub == "u/r"
    assert args.private


def test_resolve_device_uses_cuda_when_available(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert _resolve_device(None) == "cuda"


def test_resolve_device_falls_back_to_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert _resolve_device(None) == "cpu"


def test_resolve_device_respects_explicit_choice():
    assert _resolve_device("cpu") == "cpu"


class FakeTokenizer:
    def __init__(self):
        self.pad_token = None
        self.eos_token = "[EOS]"

    def __call__(self, texts, return_tensors="pt", padding=False, truncation=False, max_length=0):
        return {"input_ids": torch.ones(len(texts), 4, dtype=torch.long),
                "attention_mask": torch.ones(len(texts), 4, dtype=torch.long)}

    def save_pretrained(self, path):
        pass


class FakeLinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 4, bias=False)
        self.fc2 = nn.Linear(4, 2, bias=False)

    def forward(self, input_ids, attention_mask=None, labels=None):
        x = input_ids.float().mean(dim=-1, keepdim=True)  # [B, 1]
        x = x.expand(-1, 8)  # [B, 8]
        x = self.fc1(x)
        x = self.fc2(x)
        if labels is not None:
            loss = ((x - labels.float().mean(dim=-1, keepdim=True).expand(-1, 2)) ** 2).mean()
            return type("LossHolder", (), {"loss": loss})()
        return x


def _tiny_dataloader(args, tokenizer):
    class TinyDataset:
        def __len__(self):
            return 4

        def __getitem__(self, idx):
            return "hello world"

    return DataLoader(TinyDataset(), batch_size=args.batch_size,
                      collate_fn=lambda b: {
                          "input_ids": torch.ones(len(b), 4, dtype=torch.long),
                          "attention_mask": torch.ones(len(b), 4, dtype=torch.long),
                          "labels": torch.ones(len(b), 4, dtype=torch.long),
                      })


def test_main_end_to_end_on_tiny_model(tmp_path):
    model = FakeLinearModel()
    tokenizer = FakeTokenizer()
    save_calls = []

    def fake_load(model_id, dtype, trust_remote_code):
        return model, tokenizer

    def fake_save(m, tok, out_dir, run_metadata):
        save_calls.append((out_dir, run_metadata))

    out = str(tmp_path / "out")
    rc = main(
        ["--model", "org/repo", "--output", out,
         "--max-steps", "2", "--batch-size", "2", "--lr", "0.1",
         "--max-length", "4", "--log-every", "1"],
        _load_fn=fake_load,
        _device_fn=lambda d: "cpu",
        _dataloader_fn=_tiny_dataloader,
        _save_fn=fake_save,
    )
    assert rc == 0
    assert len(save_calls) == 1
    run_meta = save_calls[0][1]
    assert "first_loss" in run_meta
    assert "mean_final_loss" in run_meta
    assert run_meta["optimizer_steps"] == 2


def test_main_rejects_group_without_group_size(capsys):
    rc = main(
        ["--model", "x", "--output", "y", "--granularity", "group"],
        _load_fn=lambda *a, **k: (FakeLinearModel(), FakeTokenizer()),
        _device_fn=lambda d: "cpu",
        _dataloader_fn=_tiny_dataloader,
        _save_fn=lambda *a, **k: None,
    )
    assert rc == 2
    assert "group-size" in capsys.readouterr().err


def test_main_no_layers_converted_returns_1():
    model = nn.Sequential(nn.LayerNorm(8))
    tokenizer = FakeTokenizer()

    def fake_load(model_id, dtype, trust_remote_code):
        return model, tokenizer

    rc = main(
        ["--model", "org/repo", "--output", "/tmp/o", "--max-steps", "1"],
        _load_fn=fake_load,
        _device_fn=lambda d: "cpu",
        _dataloader_fn=_tiny_dataloader,
        _save_fn=lambda *a, **k: None,
    )
    assert rc == 1


def test_main_calls_push_fn_when_requested(tmp_path):
    model = FakeLinearModel()
    tokenizer = FakeTokenizer()
    push_calls = []

    def fake_push(checkpoint_dir, repo_id, private=False):
        push_calls.append((checkpoint_dir, repo_id, private))
        return "url"

    out = str(tmp_path / "out")
    rc = main(
        ["--model", "x", "--output", out, "--max-steps", "1",
         "--batch-size", "2", "--max-length", "4", "--push-to-hub", "u/r", "--private"],
        _load_fn=lambda *a, **k: (model, tokenizer),
        _device_fn=lambda d: "cpu",
        _dataloader_fn=_tiny_dataloader,
        _save_fn=lambda *a, **k: None,
        _push_fn=fake_push,
    )
    assert rc == 0
    assert len(push_calls) == 1
    assert push_calls[0][1] == "u/r"
    assert push_calls[0][2] is True
