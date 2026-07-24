# Changelog

## 0.6.0rc1

- Add cross-platform quality, packaging, dependency-audit, and release gates.
- Establish dynamic package version metadata, development tooling, and community governance.
- Restrict source privacy scans to Git-tracked files and require exact binary exemptions.
- Separate watcher and training lifecycle semantics so snapshots cannot report false completion.
- Add streaming JSONL checks, bounded media scanning, path containment, and atomic reports.
- Add persistent deterministic reliability rules, deduplicated alerts, Webhook/Prometheus/OTel
  exports, a localhost Web dashboard, and an optional terminal TUI.
- Add a stable Python API, Hugging Face callback, checkpoint-gated supervised restart, and a
  reproducible CPU fault-injection benchmark.
- Synchronize and streamline English/Chinese onboarding and add CLI, configuration,
  reliability, and release references.
- Declare Beta status and Python 3.10–3.14 support with full cross-platform pytest coverage.
- Fix the single-file release allowlist with SHA256 digests and define the sdist boundary.
- Publish release candidates through full gates with wheel/sdist isolation smoke, dependency
  audit, checksums, CycloneDX SBOM, and GitHub prerelease assets.
- Add an authenticated, audited, capability-gated control queue limited to supervised
  processes, plus persistent samples, checkpoints, recovery history, and `run status`.
- Split parser and run watch/check/manifest modules, remove unused abstractions, and share one
  reliability runtime across the public API and Hugging Face callback.
- Retain deprecated command aliases for this candidate cycle only; removal is scheduled for
  0.6.0 final.

## 0.5.0

- Add run lifecycle JSONL (`train_guard_lifecycle.jsonl`) with `start` / `heartbeat` / `checkpoint` / `finish` / `abort`.
- `run watch` records lifecycle beside `watch.jsonl`; `run check`, `manifest`, and `run compare` consume the summary.
- Strengthen LLaMAFactory adapter for nested `saves/.../checkpoint-*` layouts.
- Add GitHub Actions CI (unittest + privacy + release bundle gates).
- Complete init templates and honor `--framework` for `run check` / `run compare`.

## 0.4.0

- Configuration-driven CLI (`init`, schema validation, template starters).
- Privacy-safe P0 baseline and single-file `release/` workflow.
