# CLI reference — 0.6.0rc1

Run `train-guard COMMAND --help` for the authoritative option list. Commands accept
configuration through `--config` where shown by help.

## Commands

- `init`: write a validated `generic`, `transformers`, or `llamafactory` starter config.
- `doctor`: inspect Python, GPU visibility, resources, and model shards.
- `data check`: validate annotations, messages, media containment, and cross-split groups.
- `data inventory`: stream field and media summaries.
- `data compare`: compare two annotation files.
- `run launch`: execute preflight, monitored supervision, recovery, post-check, and manifest generation as one workflow.
- `run watch`: continuously observe logs, GPU state, checkpoints, and reliability rules.
- `run snapshot`: collect one watcher sample.
- `run check`: determine whether output evidence supports completion.
- `run compare`: compare run directories.
- `run status`: print one persistent monitoring and recovery snapshot.
- `run supervise`: launch an argv with opt-in, checkpoint-gated, bounded restart.
- `eval`: compare predictions and references.
- `manifest`: write a run manifest and experiment fingerprint.
- `bundle-info`: print version, optional dependency, and deployment information.
- `show`: serve the loopback-only Web dashboard from a reliability state database.
- `tui`: run the optional SSH-friendly terminal dashboard.

`compare` is a convenience alias for `run compare`. The deprecated aliases `precheck`,
`monitor`, `postcheck`, and `evaluate` remain only through the 0.6.0 release-candidate
cycle and emit a warning. They are scheduled for removal in 0.6.0 final; migrate now.

## Status and errors

`PASS`, `WARN`, and `FAIL` map to exit codes `0`, `1`, and `2`. Parser/usage errors use
`3`, invalid configuration uses `4`, runtime failures use `5`, and overwrite refusal
uses `6`. Automation should branch on the integer code, not localized output text.

`data check --issues-jsonl PATH` writes every detected missing-media, empty-answer,
zero-byte, unreadable-media, path-escape, and scan-budget issue as one JSON object per line.
This opt-in ledger is local and may contain media paths; treat it as sensitive data.

Ctrl-C or a watcher snapshot means observation stopped; neither proves training completed.
Use `run check` against the output evidence.

## Single-file CLI

The GitHub release attaches `train_guard.py`. It has the same command surface:

```bash
python train_guard.py --version
python train_guard.py doctor --config train-guard.json
```

Verify it against `SHA256SUMS` before use. The wheel is preferred for normal installation.
