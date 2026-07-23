# Train Guard migration notes

Version 0.3 introduces a modular package under `src/train_guard/` and a controlled single-file candidate under `release/`.

## Neutral public interface

- Use `group_id` for cross-split grouping.
- Use `keywords` for configurable evaluation terms.
- Use `data check`, `run watch`, `run check`, and `eval` for current command names.
- Generic command aliases remain temporarily accepted and emit a warning. Domain-specific field aliases have been removed from the public source tree.

## Release workflow

1. Edit `src/train_guard/**`.
2. Run `python scripts/bundle_singlefile.py`.
3. Run `python scripts/check_release_manifest.py`.
4. Run `python scripts/privacy_scan.py --allowlist configs/privacy_scan_allowlist.json`.
5. Run `python scripts/privacy_scan.py --mode source --root .`.
6. Publish only the validated contents of `release/` and the scanned public source entries.

The ignored `dist/` and `build/` directories are local packaging scratch space and must not be used as release sources.
