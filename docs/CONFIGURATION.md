# Configuration — 0.6.0rc1

Train Guard configuration uses `schema_version: 1` and JSON by default. YAML requires the
`yaml` extra. Generate a complete starting point instead of writing one from memory:

```bash
train-guard init --template generic --output train-guard.json
train-guard init --template transformers --output train-guard.json
train-guard init --template llamafactory --output train-guard.yaml
```

Existing files are not replaced unless `--force` is explicit.

## Resolution rules

The precedence is `explicit CLI option > configuration file > built-in default`.
Only options explicitly supplied on the command line override file values. Relative paths
from a configuration file resolve from that file's directory; CLI paths resolve from the
current working directory. Unknown fields, wrong types, and unsupported schema versions fail
before command work begins.

Top-level sections are `doctor`, `data`, `run`, `eval`, and `manifest`. See
`configs/example.json` or `configs/example.yaml`, then use command help for the corresponding
option names.

## Optional features

- `yaml`: read and write YAML configurations.
- `image`: verify image payloads and dimensions.
- `psutil`: richer local process and resource inspection.
- `all`: all three runtime extras.

Optional features are imported lazily and are never auto-installed. The `tui` extra adds
Textual for the optional terminal dashboard; `all` includes it.

## Sensitive values

Keep credentials out of configuration when possible. Webhook URLs may contain sensitive
query strings; do not commit them, paste them into issues, or include them in release
artifacts. An opt-in `data.check.issues_jsonl` ledger can contain media paths and must remain
local unless explicitly reviewed. Reports should use synthetic paths and data before sharing.
