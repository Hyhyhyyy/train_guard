# Train Guard 0.6.0rc1

[![CI](https://github.com/Hyhyhyyy/train_guard/actions/workflows/ci.yml/badge.svg)](https://github.com/Hyhyhyyy/train_guard/actions/workflows/ci.yml)
[![Python 3.10–3.14](https://img.shields.io/badge/python-3.10--3.14-blue)](https://www.python.org/)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)

[简体中文](README_zh-CN.md)

Train Guard is a local-first toolkit for LLM/VLM training checks, reliability events,
alerts, and explicitly controlled recovery. The core has no required dependencies.
Observation is the default: it does not install packages, upload telemetry, stop training,
or modify training data.

## Install

Choose one method:

```bash
# Isolated CLI (recommended)
pipx install train-guard==0.6.0rc1
uv tool install train-guard==0.6.0rc1

# Current Python environment
python -m pip install train-guard==0.6.0rc1

# Source checkout
git clone https://github.com/Hyhyhyyy/train_guard.git
cd train_guard
python -m pip install -e .
```

Optional extras are `yaml`, `image`, `psutil`, `tui`, and `all`:

```bash
python -m pip install "train-guard[all]==0.6.0rc1"
python -m pip install "train-guard[tui]==0.6.0rc1"
```

Upgrade or remove with the tool used to install:

```bash
pipx upgrade train-guard              # or: uv tool upgrade train-guard
python -m pip install --upgrade train-guard
pipx uninstall train-guard            # or: uv tool uninstall train-guard
python -m pip uninstall train-guard
```

## Three-minute workflow

```bash
# 1. Create a generic, Transformers, or LLaMAFactory config.
train-guard init --template transformers --output train-guard.json

# 2. Replace placeholder paths, then validate environment and data.
train-guard doctor --config train-guard.json
train-guard data check --config train-guard.json

# 3. Observe training and verify outputs.
train-guard run watch --config train-guard.json
train-guard run check --config train-guard.json
train-guard manifest --config train-guard.json
```

For a source checkout use `python train_guard.py ...`; for the release asset use
`python train_guard.py ...` after downloading the attached single file.

## Interfaces and recovery

- **CLI:** the supported interface; see [docs/CLI.md](docs/CLI.md).
- **Web:** `train-guard show --state-db PATH` serves metrics, GPU state, alerts,
  checkpoints, recovery history, and managed-process status on loopback only.
- **TUI:** `train-guard tui --state-db PATH` provides the same persistent status over SSH.
- **Status:** `train-guard run status --state-db PATH` emits one scriptable snapshot.
- **Controlled recovery:** `run supervise` can restart only an explicit argv, only when
  `--restart` is supplied, only after checkpoint validation, and within a finite budget.
  It never executes a shell string. See [docs/RELIABILITY.md](docs/RELIABILITY.md).

```bash
train-guard run supervise --restart --max-restarts 1 \
  --checkpoint-dir ./checkpoint-100 \
  --required-checkpoint-file trainer_state.json \
  -- python train.py --resume-from-checkpoint ./checkpoint-100
```

The training argv itself must include its framework-specific resume option.

Control is disabled by default. Start a supervised run with `--enable-control`, then start
the Web dashboard with the same state database and `--enable-control`. The dashboard prints
an in-memory token once and accepts only allowlisted actions for that supervised process:

```bash
train-guard run supervise --enable-control --state-db ./guard.sqlite \
  -- python train.py
train-guard show --enable-control --state-db ./guard.sqlite
```

The dashboard rejects non-loopback clients, non-local origins, expired commands, duplicate
command IDs, unmanaged processes, and unsupported capabilities. Do not proxy it publicly.

## Safety boundary and exit codes

Reports redact absolute paths, usernames, hostnames, and credential-like values. Public
HTML/JSON must not include raw samples or media bytes. Optional packages are never installed
automatically. Webhooks and exported telemetry are opt-in; users control their destinations.
Train Guard is a diagnostic aid, not a security sandbox, compliance system, or guarantee
that a training run is correct.

Stable process results are: `0` PASS, `1` WARN, `2` FAIL, `3` usage error,
`4` configuration error, `5` runtime error, and `6` refused overwrite.

## Topics

- [CLI reference](docs/CLI.md)
- [Configuration and precedence](docs/CONFIGURATION.md)
- [Reliability, Web, and recovery](docs/RELIABILITY.md)
- [Release process and artifacts](docs/RELEASE.md)
- [Migration and one-candidate alias window](docs/MIGRATION.md)
- [Architecture](ARCHITECTURE.md)
- [Contributing](CONTRIBUTING.md), [security](SECURITY.md), and [support](SUPPORT.md)
- [Changelog](CHANGELOG.md)

Licensed under the Apache License 2.0; see [LICENSE](LICENSE).
