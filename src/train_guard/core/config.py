"""Config merge: CLI > file > defaults."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from .optional import try_import_yaml


def deep_merge(base: Dict[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    """Recursively merge dictionaries."""
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, Mapping):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config_file(path: Path) -> Dict[str, Any]:
    """Load YAML or JSON config. YAML requires PyYAML."""
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        yaml = try_import_yaml()
        if yaml is None:
            raise RuntimeError(
                "YAML config requires PyYAML. Install with `pip install pyyaml` "
                "or use a JSON config file."
            )
        data = yaml.safe_load(text)
    elif suffix == ".json":
        data = json.loads(text)
    else:
        raise ValueError(f"Unsupported config format: {suffix} (use .yaml/.yml/.json)")
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be an object: {path}")
    return data


def merge_cli_config(
    defaults: Dict[str, Any],
    config_path: Optional[Path],
    cli_overrides: Mapping[str, Any],
) -> Dict[str, Any]:
    """Merge defaults < config file < CLI (None CLI values ignored)."""
    merged = dict(defaults)
    if config_path is not None:
        merged = deep_merge(merged, load_config_file(config_path))
    cleaned = {k: v for k, v in cli_overrides.items() if v is not None}
    return deep_merge(merged, cleaned)
