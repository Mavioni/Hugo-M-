# Changelog

All notable changes to Hugo are documented here.

## [Unreleased]

### Fixed
Four correctness bugs surfaced by an automated review on PR #1, each verified
against the code before fixing and each now covered by a regression test:

- **Grouped checkpoints could not be reconstructed at all.**
  `reconstruct_shard` called `dequantize_weight` without the manifest's
  `group_size`; group-granularity scales are 3D, so dequantization hit
  `in_features // None` and raised `TypeError`. The group size is now threaded
  through from `manifest.json`.
- **Resumed runs silently accepted incompatible settings.** `load_manifest`
  validated only `repo_id`, so resuming with a different granularity, group
  size, revision, or skip pattern would keep the already-quantized shards and
  process the rest differently — yielding a checkpoint whose shards disagree
  with each other and a manifest that misdescribes it. All data-affecting
  settings are now checked, and the check raises `SystemExit` rather than
  using `assert` (asserts vanish under `python -O`, which would disable the
  guard exactly when running optimized).
- **Custom-architecture model code was dropped.** `copy_aux_files` excluded
  `.py` files and every nested path, so models needing `trust_remote_code`
  (`modeling_*.py` referenced by `config.json`'s `auto_map`) produced output
  that could not be loaded — precisely the models the streaming path exists to
  support. Python files and nested layouts are now preserved, with a
  path-traversal guard so nothing is written outside the output directory.
  `reconstruct.py` likewise now copies recursively instead of top-level only.
- **Resumed runs under-reported aggregate statistics.** Totals were computed
  only from shards the current process handled, so shards finished by an
  earlier invocation were omitted. Per-shard stats are now persisted in the
  manifest and aggregated across every completed shard. This had affected real
  reported numbers: the 27B run's stats were computed after a resume and
  omitted the first shard, understating it as 606 layers / 45.67GB / 5.71GB
  when the true totals are **614 layers / 47.00GB / 5.87GB** (the 8x
  compression ratio is unchanged). Docs citing the old figures are corrected.

### Added
- `hugo.qat`: quantization-aware training so weights are *trained* to
  tolerate ternary rounding rather than rounded after the fact. Implements a
  clipped straight-through estimator (`round()` has zero true gradient, so
  the backward pass treats quantization as the identity where the clamp
  didn't saturate), a `BitLinear` layer that fake-quantizes on every forward
  pass, and in-place convert/bake helpers. Uses the same `_absmean_scale` as
  the PTQ path so training and deployment-time rounding agree exactly.
- `hugo.train_qat` (`hugo-train-qat`): QAT fine-tuning CLI with gradient
  accumulation, configurable granularity/skip patterns, and run metadata
  written alongside the checkpoint. Validated end-to-end on a small proxy
  model (loss 20.31 -> 19.49 over 20 steps; all layers bake out with <= 3
  distinct values per row and the checkpoint reloads cleanly).
- `hugo.push_to_hub` (`hugo-push`): publish a checkpoint to the Hugging Face
  Hub, with an auto-generated model card that distinguishes QAT from PTQ
  provenance and reports PTQ's measured weight error rather than implying the
  model was trained ternary. Credentials come only from `HF_TOKEN` /
  `HUGGING_FACE_HUB_TOKEN` / a cached CLI login — never a command-line flag,
  so tokens can't leak into shell history or CI logs. `--dry-run` previews
  the card and checks credentials without uploading.
- `docs/qat.md` and `examples/qat_train_and_publish.sh`.

### Changed
- README now leads with QAT as the recommended path and contrasts it against
  PTQ in a table, replacing the previous PTQ-only framing.
- `datasets` added as a `[train]` extra (QAT-only dependency; PTQ users
  don't need it).

## [0.1.0] - Initial re-tune tooling

### Added
- `hugo.quantize`: BitNet b1.58-style absmean ternary quantization math
  (`tensor`/`channel`/`group` granularity), 2-bit pack/unpack, per-layer
  error stats.
- `hugo.ternarize`: loads a model via `transformers`, quantizes it in
  memory, saves a drop-in HF checkpoint plus an optional 2-bit-packed
  sidecar. Suited to models that fit comfortably in RAM/disk.
- `hugo.stream_ternarize` / `hugo.streaming`: disk-bounded quantization
  that processes one safetensors shard at a time (download, quantize,
  pack, delete, repeat), so checkpoints far larger than local disk can
  still be re-tuned. Resumable via a per-shard `manifest.json`. Never
  instantiates the model class, so it works on architectures the
  installed `transformers` doesn't support yet.
- `hugo.reconstruct`: rebuilds a normal, loadable checkpoint from a
  `stream_ternarize` output, on a machine with enough disk for the full
  model.
- `hugo.load_packed`: reconstruct an individual layer's weight from a
  packed sidecar without a full checkpoint rebuild.
- Repository packaging (`pyproject.toml`, `src/` layout, console scripts),
  CI (`.github/workflows/tests.yml`), and community-health files, modeled
  on the structure of several widely-used open-source repositories
  (`psf/requests`, `pallets/flask`, `tiangolo/fastapi`, `Textualize/rich`,
  `huggingface/transformers`, `ggerganov/llama.cpp`, `microsoft/BitNet`).

### Known limitations
- This is post-training quantization (PTQ), not quantization-aware
  training. Expect a real accuracy hit versus a model trained ternary from
  scratch; error grows as scale granularity gets coarser
  (`channel` < `group` < `tensor`).
- No inference speedup without a ternary-aware kernel -- the drop-in
  checkpoint from `hugo.ternarize` stores ternary values as regular
  fp16/bf16 numbers.
