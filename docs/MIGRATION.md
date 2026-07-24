# Train Guard migration notes

## 0.6.0rc1

- Python 3.10–3.14 are the declared and CI-tested versions.
- Installable and single-file CLIs share the same command model; the wheel remains preferred.
- Reliability state, loopback Web display, and checkpoint-gated supervision are documented in
  [RELIABILITY.md](RELIABILITY.md).
- The source distribution now has an explicit boundary, while the single-file release uses a
  fixed path allowlist and SHA256 manifest.
- Configuration remains `schema_version: 1`; precedence and path resolution are unchanged.

## Command migration

Use `data check`, `run watch`, `run check`, and `eval`. Deprecated `precheck`, `monitor`,
`postcheck`, and `evaluate` aliases emit a warning and are retained for one release-candidate
cycle only: 0.6.0rc1. They are scheduled for removal in 0.6.0 final. `compare` remains a
convenience alias for `run compare` and is not part of that removal.

Stable public dataset fields remain `group_id`, `keywords`, `messages`, and `media`.
Domain-specific field aliases are not public.

## Earlier changes

Version 0.5 added privacy-safe lifecycle JSONL produced by `run watch` and consumed by
`run check`, `manifest`, and `run compare`. Version 0.3 introduced the modular package under
`src/train_guard/` and the generated single-file candidate under `release/`.

See [RELEASE.md](RELEASE.md) for the current release procedure.
