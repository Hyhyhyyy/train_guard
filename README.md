# Train Guard

Read-only, low-dependency quality checks for LLM/VLM fine-tuning workflows.

Train Guard checks environments, dataset structure, media references, and training-run artifacts. It does not validate task correctness, stop training processes, install packages, or upload telemetry.

## Release boundary

The controlled candidate tree is `release/`. Generate and validate it with:

```bash
python scripts/bundle_singlefile.py
python scripts/check_release_manifest.py
python scripts/privacy_scan.py --allowlist configs/privacy_scan_allowlist.json
```

`dist/` and `build/` are ignored local packaging directories and are never release inputs. The manifest gate rejects sensitive directories, reports, caches, logs, databases, bytecode, and symbolic links.

## Quick start

```bash
PYTHONPATH=src python -m train_guard --help
python release/train_guard.py doctor
python release/train_guard.py run watch --once
python release/train_guard.py data check --annotation ./data/train.jsonl --data-root ./data/assets
```

## Commands

| Command | Purpose |
|---|---|
| `doctor` | Environment, GPU, and model-shard checks |
| `data check` | Annotation/media integrity and `group_id` cross-split checks |
| `data inventory` | Streaming field and media histograms |
| `data compare` | Compare two annotation files |
| `run watch` | Sidecar GPU, log, and checkpoint watch |
| `run check` | Adapter and trainer completion checks |
| `run compare` | Compare run directories using metadata |
| `eval` | Prediction/reference metrics with generic `keywords` |
| `manifest` | Run manifest and experiment fingerprint |
| `bundle-info` | Version, optional dependencies, and SHA256 |

Configurable fields include `messages`, `images`, `media`, `input`, `output`, `group_id`, `split`, `prediction`, and `reference`. Deprecated interface aliases remain temporarily available and emit warnings; new integrations should use only the neutral names shown here.

Reports redact absolute paths, usernames, hostnames, and credential-like values. Public HTML and JSON must not embed raw sample text or media bytes. Optional packages are never installed automatically.

See [docs/MIGRATION.md](docs/MIGRATION.md). Licensed under Apache License 2.0; see [LICENSE](LICENSE).
