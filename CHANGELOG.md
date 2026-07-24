# Changelog

## 0.5.0

- Add run lifecycle JSONL (`train_guard_lifecycle.jsonl`) with `start` / `heartbeat` / `checkpoint` / `finish` / `abort`.
- `run watch` records lifecycle beside `watch.jsonl`; `run check`, `manifest`, and `run compare` consume the summary.
- Strengthen LLaMAFactory adapter for nested `saves/.../checkpoint-*` layouts.
- Add GitHub Actions CI (unittest + privacy + release bundle gates).
- Complete init templates and honor `--framework` for `run check` / `run compare`.

## 0.4.0

- Configuration-driven CLI (`init`, schema validation, template starters).
- Privacy-safe P0 baseline and single-file `release/` workflow.
