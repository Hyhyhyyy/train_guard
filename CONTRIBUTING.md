# Contributing

Thank you for improving Train Guard. By participating, you agree to follow
[CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

## Before opening a change

Use an issue for substantial behavior or interface changes. Do not include real datasets,
credentials, machine-specific paths, private reports, or training outputs in issues, tests,
commits, or release candidates. Security vulnerabilities belong in the private process in
[SECURITY.md](SECURITY.md).

## Development setup

Python 3.10–3.14 are supported.

```bash
python -m venv .venv
# Activate .venv for your shell, then:
python -m pip install -e ".[dev,all]"
```

Before submitting:

```bash
ruff check .
mypy src/train_guard scripts
pytest --cov=train_guard
python scripts/privacy_scan.py --mode source --root . \
  --allowlist configs/privacy_scan_allowlist.json
python scripts/bundle_singlefile.py
git diff --exit-code -- release/train_guard.py
python scripts/check_release_manifest.py
```

The package must retain zero required runtime dependencies. New integrations should be
optional extras and must fail with a clear installation hint when unavailable. Keep the
tool read-only: it must not stop training, mutate training inputs, install packages, or
upload telemetry.

## Pull requests

Keep changes focused, add tests for observable behavior, update both READMEs when user-facing
instructions change, and add a changelog entry for release-visible changes. Generated
`release/train_guard.py` may only change through `scripts/bundle_singlefile.py`.
When that generated candidate intentionally changes, regenerate
`configs/release_manifest.json` with `python scripts/check_release_manifest.py --generate`,
review both diffs, then run validation again. Keep optional interfaces, including the TUI,
behind explicit extras with clear failure messages. See [docs/RELEASE.md](docs/RELEASE.md)
for the complete gates and release-channel boundary.

All contributions are licensed under Apache License 2.0.
