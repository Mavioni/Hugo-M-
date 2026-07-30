# Security Policy

## Supported versions

Hugo is pre-1.0 and moves fast; only the latest commit on `main` is
supported. Please update before reporting an issue.

## Reporting a vulnerability

Please open a private security advisory on this repository (GitHub ->
Security -> Advisories -> Report a vulnerability) rather than a public
issue, so any fix can ship before the details are public.

## Known-risk areas specific to this project

- **`--trust-remote-code`** (`hugo-ternarize`): this executes arbitrary
  Python shipped in the source model's Hub repo. Only pass it for repos you
  trust, the same caution `transformers` itself documents.
- **Arbitrary Hub repo IDs**: both CLIs download and locally execute the
  quantization pipeline against whatever `--model` repo you point them at.
  They do not sandbox the download -- treat `--model` like you would any
  other "run code/data from the internet" input.
- **Disk cleanup on `hugo-stream-ternarize`**: the streaming path deletes
  each downloaded shard from `--work-dir` after processing it to stay
  disk-bounded. If you point `--work-dir` at a path with other data in it,
  only files this tool itself downloaded are removed, but please use a
  dedicated scratch directory regardless.
