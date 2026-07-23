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

## Five-minute quick start

```bash
# 1. Works with the development launcher or the single-file release.
python train_guard.py init --output train-guard.json

# 2. Edit only the placeholder paths in train-guard.json, then run:
python train_guard.py doctor --config train-guard.json
python train_guard.py data check --config train-guard.json
```

The generated file is JSON and needs no third-party package. Select a framework
starter with `--template generic`, `--template transformers`, or
`--template llamafactory`. Existing files are protected; use `--force` only
when replacement is intentional. YAML input/output is optional and works only
when PyYAML is already installed.

Configuration precedence is `CLI option > configuration file > built-in
default`. Every configuration has `schema_version: 1`; unknown fields, wrong
types, and unsupported versions fail before Train Guard reads data or starts a
watch. Relative paths in a configuration are resolved from that file's
directory, so the same file works from another current directory.

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

The stable, domain-neutral dataset fields are `group_id`, `keywords`,
`messages`, and `media`. Framework-specific artifact and log conventions stay
inside adapters. Deprecated command aliases remain temporarily available and
emit warnings.

Command results use one vocabulary: `PASS` exits 0, `WARN` exits 1, and `FAIL`
exits 2. Usage errors exit 3, invalid configuration exits 4, runtime errors
exit 5, and a refused overwrite exits 6.

Reports redact absolute paths, usernames, hostnames, and credential-like values. Public HTML and JSON must not embed raw sample text or media bytes. Optional packages are never installed automatically.

See [docs/MIGRATION.md](docs/MIGRATION.md). Licensed under Apache License 2.0; see [LICENSE](LICENSE).
