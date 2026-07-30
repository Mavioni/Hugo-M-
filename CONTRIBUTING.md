# Contributing to Hugo

Thanks for considering a contribution.

## Setup

```bash
git clone <this repo>
cd Hugo-M-
pip install torch --index-url https://download.pytorch.org/whl/cpu  # CPU wheel is much smaller/faster for dev
pip install -e ".[dev]"
```

## Workflow

1. Make your change under `src/hugo/`.
2. Add or update tests under `tests/` -- every new code path should have a
   test that would fail without the fix (see existing tests for the style:
   `tests/test_streaming.py` mocks the network with a local fake safetensors
   shard, no Hub access needed to run the suite).
3. Run the checks that CI runs:
   ```bash
   ruff check src tests
   pytest -v
   ```
4. Open a PR describing the change and why it's needed.

## Design notes worth knowing before you touch the streaming path

- `hugo/quantize.py` holds the pure math (no I/O): ternarize/dequantize,
  2-bit pack/unpack, per-layer stats. Keep it free of `transformers`/network
  dependencies so it stays trivially testable.
- `hugo/streaming.py` + `hugo/stream_ternarize.py` process one safetensors
  shard at a time and delete it after quantizing, specifically so this
  works on models much larger than available disk. Any change here should
  preserve that invariant -- don't introduce a code path that accumulates
  more than one shard's tensors in memory at once.
- Re-deriving a scale from already-dequantized ternary weights gives the
  wrong scale (rounding zeroes out part of the absmean). Always compute
  `(codes, scale)` once from the original weights and thread that pair
  through, rather than recomputing from a quantized tensor -- this bit us
  once already (see the `quantized` dict returned by
  `quantize_linear_modules`).

## Reporting bugs / security issues

Functional bugs: open a GitHub issue with repro steps.
Security issues: see [SECURITY.md](SECURITY.md).
