# Train Guard

Read-only, low-dependency quality checks for LLM/VLM fine-tuning workflows.

Train Guard checks environments, dataset structure, media references, and training-run artifacts. It does not validate task correctness, stop training processes, install packages, or upload telemetry.

**Repository:** https://github.com/Hyhyhyyy/train_guard

## Recommended workflow (one config)

```bash
# 1. Install in editable mode (optional extras: yaml, image, psutil, all)
pip install -e .

# 2. Generate a starter config (generic | transformers | llamafactory)
train-guard init --template transformers --output train-guard.json

# 3. Edit only the placeholder paths in train-guard.json, then:
train-guard doctor --config train-guard.json
train-guard data check --config train-guard.json

# 4. During / after training
train-guard run watch --config train-guard.json
train-guard run check --config train-guard.json
train-guard manifest --config train-guard.json
```

No install? Use the development launcher or the single-file release:

```bash
python train_guard.py init --output train-guard.json
python release/train_guard.py doctor --config train-guard.json
```

Configuration precedence is `CLI option > configuration file > built-in default`. Every configuration has `schema_version: 1`. Relative paths resolve from the config file directory.

## Commands

| Command | Purpose |
|---|---|
| `init` | Generate generic, Transformers, or LLaMAFactory configuration |
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

Stable dataset fields: `group_id`, `keywords`, `messages`, `media`.

Results: `PASS`→0, `WARN`→1, `FAIL`→2. Usage=3, configuration=4, runtime=5, refused overwrite=6.

## Release boundary

The controlled candidate tree is `release/`. Generate and validate it with:

```bash
python scripts/bundle_singlefile.py
python scripts/check_release_manifest.py
python scripts/privacy_scan.py --allowlist configs/privacy_scan_allowlist.json
```

`dist/` and `build/` are ignored local packaging directories and are never release inputs.

Reports redact absolute paths, usernames, hostnames, and credential-like values. Public HTML and JSON must not embed raw sample text or media bytes. Optional packages are never installed automatically.

See [docs/MIGRATION.md](docs/MIGRATION.md). Licensed under Apache License 2.0; see [LICENSE](LICENSE).
