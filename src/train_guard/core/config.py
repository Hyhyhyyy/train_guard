"""Versioned configuration loading, validation, templates, and precedence."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Dict, Mapping, Optional

from .io_util import atomic_write_text
from .optional import try_import_yaml


SCHEMA_VERSION = 1
TEMPLATE_NAMES = ("generic", "transformers", "llamafactory")


class ConfigError(ValueError):
    """A configuration problem with an actionable, privacy-safe message."""

    def __init__(self, problem: str, location: str, suggestion: str) -> None:
        self.problem = problem
        self.location = location
        self.suggestion = suggestion
        super().__init__(
            f"Configuration error\nProblem: {problem}\nLocation: {location}\nFix: {suggestion}"
        )


def _template(framework: str) -> Dict[str, Any]:
    output_dir = "./outputs"
    log_file = "./outputs/train.log"
    if framework == "llamafactory":
        output_dir = "./saves/example/lora"
        log_file = "./train.log"
    return {
        "schema_version": SCHEMA_VERSION,
        "doctor": {
            "model_path": "./models/example-model",
            "expected_gpus": None,
        },
        "data": {
            "check": {
                "annotation": "./data/train.jsonl",
                "data_root": "./data/media",
                "sample_limit": 1000,
                "full_scan": False,
                "compute_hash": False,
                "verify_images": True,
                "max_image_pixels": 50_000_000,
                "max_media_files": 10_000,
                "max_scan_bytes": 5 * 1024**3,
                "allow_external_media": False,
                "group_id": "group_id",
                "messages": "messages",
                "media": "media",
                "split": "split",
                "report_dir": "./reports/data_check",
            },
            "inventory": {
                "annotation": "./data/train.jsonl",
                "sample_limit": 1000,
                "group_id": "group_id",
                "report_dir": "./reports/data_inventory",
            },
            "compare": {
                "left": "./data/train.jsonl",
                "right": "./data/val.jsonl",
                "sample_limit": 1000,
                "group_id": "group_id",
                "report_dir": "./reports/data_compare",
            },
        },
        "run": {
            "watch": {
                "framework": framework,
                "once": False,
                "interval": 30,
                "output_dir": output_dir,
                "log_file": log_file,
                "expected_gpus": None,
                "stale_log_minutes": 15.0,
                "disk_free_gb_threshold": 10.0,
                "run_id": None,
                "state_db": None,
                "webhook_url": None,
                "prometheus_file": None,
                "otel_file": None,
                "reliability": True,
                "notification_every": 10,
                "step_stall_seconds": 300.0,
                "gpu_overheat_celsius": 90.0,
                "checkpoint_stale_seconds": 1800.0,
            },
            "check": {
                "framework": framework,
                "output_dir": output_dir,
                "expected_steps": None,
                "training_type": "auto",
                "json_output": "./reports/run_check.json",
                "html_output": "./reports/run_check.html",
            },
            "compare": {
                "framework": framework,
                "left": output_dir,
                "right": f"{output_dir}-baseline",
                "json_output": "./reports/run_compare.json",
            },
        },
        "eval": {
            "predictions": "./outputs/predictions.jsonl",
            "references": "./data/references.jsonl",
            "group_id": "group_id",
            "keywords": [],
            "sample_limit": 1000,
            "report_dir": "./reports/eval",
        },
        "manifest": {
            "framework": framework,
            "output_dir": output_dir,
            "manifest_out": "./reports/run_manifest.json",
            "expected_steps": None,
            "seed": None,
        },
    }


def config_template(name: str = "generic") -> Dict[str, Any]:
    """Return a fresh, domain-neutral configuration template."""
    if name not in TEMPLATE_NAMES:
        raise ConfigError(
            f"unsupported template {name!r}",
            "command line: --template",
            f"choose one of: {', '.join(TEMPLATE_NAMES)}",
        )
    framework = "huggingface" if name == "transformers" else name
    return deepcopy(_template(framework))


def write_config_template(path: Path, template: str, force: bool = False) -> None:
    """Write JSON using stdlib; YAML is optional and requires PyYAML."""
    if path.exists() and not force:
        raise FileExistsError(str(path))
    suffix = path.suffix.lower()
    data = config_template(template)
    if suffix in {"", ".json"}:
        text = json.dumps(data, ensure_ascii=False, indent=2) + "\n"
    elif suffix in {".yaml", ".yml"}:
        yaml = try_import_yaml()
        if yaml is None:
            raise ConfigError(
                "YAML output requires the optional PyYAML package",
                str(path),
                "use a .json output path or install Train Guard with the yaml extra",
            )
        text = yaml.safe_dump(data, sort_keys=False, allow_unicode=True)
    else:
        raise ConfigError(
            f"unsupported configuration extension {suffix!r}",
            str(path),
            "use .json, or .yaml/.yml when PyYAML is available",
        )
    atomic_write_text(path, text, overwrite=force)


def load_config_file(path: Path) -> Dict[str, Any]:
    """Load a JSON/YAML object with actionable parse errors."""
    if not path.is_file():
        raise ConfigError(
            "configuration file was not found",
            str(path),
            "check --config or run `train-guard init --output train-guard.json`",
        )
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(
            f"configuration file could not be read ({exc})",
            str(path),
            "check that the file is readable",
        ) from exc
    suffix = path.suffix.lower()
    try:
        if suffix in {".yaml", ".yml"}:
            yaml = try_import_yaml()
            if yaml is None:
                raise ConfigError(
                    "YAML configuration requires the optional PyYAML package",
                    str(path),
                    "convert the file to JSON or install Train Guard with the yaml extra",
                )
            data = yaml.safe_load(text)
        elif suffix == ".json":
            data = json.loads(text)
        else:
            raise ConfigError(
                f"unsupported configuration extension {suffix!r}",
                str(path),
                "use .json, or .yaml/.yml when PyYAML is available",
            )
    except ConfigError:
        raise
    except Exception as exc:
        mark = ""
        if isinstance(exc, json.JSONDecodeError):
            mark = f" at line {exc.lineno}, column {exc.colno}"
        raise ConfigError(
            f"configuration syntax is invalid{mark}",
            str(path),
            "correct the syntax and try again",
        ) from exc
    if not isinstance(data, dict):
        raise ConfigError(
            "configuration root must be an object",
            str(path),
            "wrap settings in a JSON object",
        )
    return data


def _t(*types: type) -> tuple[type, ...]:
    return types


SECTION_FIELDS: Dict[tuple[str, ...], Dict[str, tuple[type, ...]]] = {
    ("doctor",): {
        "model_path": _t(str, type(None)),
        "expected_gpus": _t(int, type(None)),
        "json_output": _t(str, type(None)),
    },
    ("data", "check"): {
        "annotation": _t(str),
        "data_root": _t(str, type(None)),
        "sample_limit": _t(int, type(None)),
        "full_scan": _t(bool),
        "compute_hash": _t(bool),
        "verify_images": _t(bool),
        "max_image_pixels": _t(int, type(None)),
        "max_media_files": _t(int, type(None)),
        "max_scan_bytes": _t(int, type(None)),
        "allow_external_media": _t(bool),
        "group_id": _t(str, type(None)),
        "messages": _t(str, type(None)),
        "media": _t(str, type(None)),
        "split": _t(str, type(None)),
        "report_dir": _t(str),
        "cache_db": _t(str, type(None)),
    },
    ("data", "inventory"): {
        "annotation": _t(str),
        "sample_limit": _t(int, type(None)),
        "group_id": _t(str, type(None)),
        "report_dir": _t(str),
    },
    ("data", "compare"): {
        "left": _t(str),
        "right": _t(str),
        "sample_limit": _t(int, type(None)),
        "group_id": _t(str, type(None)),
        "report_dir": _t(str),
    },
    ("run", "watch"): {
        "once": _t(bool),
        "interval": _t(int),
        "pid": _t(int, type(None)),
        "log_file": _t(str, type(None)),
        "framework": _t(str),
        "output_dir": _t(str),
        "expected_gpus": _t(int, type(None)),
        "stale_log_minutes": _t(int, float),
        "disk_free_gb_threshold": _t(int, float),
        "run_id": _t(str, type(None)),
        "state_db": _t(str, type(None)),
        "webhook_url": _t(str, type(None)),
        "prometheus_file": _t(str, type(None)),
        "otel_file": _t(str, type(None)),
        "reliability": _t(bool),
        "notification_every": _t(int),
        "step_stall_seconds": _t(int, float),
        "gpu_overheat_celsius": _t(int, float),
        "checkpoint_stale_seconds": _t(int, float),
    },
    ("run", "check"): {
        "output_dir": _t(str),
        "framework": _t(str),
        "expected_steps": _t(int, type(None)),
        "training_type": _t(str),
        "json_output": _t(str, type(None)),
        "html_output": _t(str, type(None)),
    },
    ("run", "compare"): {
        "left": _t(str),
        "right": _t(str),
        "framework": _t(str),
        "json_output": _t(str, type(None)),
    },
    ("eval",): {
        "predictions": _t(str),
        "references": _t(str, type(None)),
        "prediction_field": _t(str, type(None)),
        "reference_field": _t(str, type(None)),
        "group_id": _t(str, type(None)),
        "label_field": _t(str, type(None)),
        "predicted_label_field": _t(str, type(None)),
        "keywords": _t(list),
        "report_dir": _t(str),
        "sample_limit": _t(int, type(None)),
    },
    ("manifest",): {
        "output_dir": _t(str),
        "framework": _t(str),
        "manifest_out": _t(str, type(None)),
        "expected_steps": _t(int, type(None)),
        "seed": _t(str, int, type(None)),
    },
}

DEFAULTS: Dict[tuple[str, ...], Dict[str, Any]] = {
    ("doctor",): {"model_path": None, "expected_gpus": None, "json_output": None},
    ("data", "check"): {
        "data_root": None,
        "sample_limit": 1000,
        "full_scan": False,
        "compute_hash": False,
        "verify_images": True,
        "max_image_pixels": 50_000_000,
        "max_media_files": 10_000,
        "max_scan_bytes": 5 * 1024**3,
        "allow_external_media": False,
        "group_id": "group_id",
        "messages": "messages",
        "media": "media",
        "split": "split",
        "report_dir": "reports/data_check",
        "cache_db": None,
    },
    ("data", "inventory"): {
        "sample_limit": None,
        "group_id": "group_id",
        "report_dir": "reports/data_inventory",
    },
    ("data", "compare"): {
        "sample_limit": None,
        "group_id": "group_id",
        "report_dir": "reports/data_compare",
    },
    ("run", "watch"): {
        "once": False,
        "interval": 30,
        "pid": None,
        "log_file": None,
        "framework": "generic",
        "output_dir": "reports/watch",
        "expected_gpus": None,
        "stale_log_minutes": 15.0,
        "disk_free_gb_threshold": 10.0,
        "run_id": None,
        "state_db": None,
        "webhook_url": None,
        "prometheus_file": None,
        "otel_file": None,
        "reliability": True,
        "notification_every": 10,
        "step_stall_seconds": 300.0,
        "gpu_overheat_celsius": 90.0,
        "checkpoint_stale_seconds": 1800.0,
    },
    ("run", "check"): {
        "framework": "generic",
        "expected_steps": None,
        "training_type": "auto",
        "json_output": None,
        "html_output": None,
    },
    ("run", "compare"): {"framework": "generic", "json_output": None},
    ("eval",): {
        "references": None,
        "prediction_field": None,
        "reference_field": None,
        "group_id": "group_id",
        "label_field": None,
        "predicted_label_field": None,
        "keywords": [],
        "sample_limit": 1000,
        "report_dir": "reports/eval",
    },
    ("manifest",): {
        "framework": "generic",
        "manifest_out": None,
        "expected_steps": None,
        "seed": None,
    },
}

REQUIRED: Dict[tuple[str, ...], tuple[str, ...]] = {
    ("data", "check"): ("annotation",),
    ("data", "inventory"): ("annotation",),
    ("data", "compare"): ("left", "right"),
    ("run", "check"): ("output_dir",),
    ("run", "compare"): ("left", "right"),
    ("eval",): ("predictions",),
    ("manifest",): ("output_dir",),
}

PATH_FIELDS = {
    "model_path",
    "json_output",
    "annotation",
    "data_root",
    "report_dir",
    "cache_db",
    "left",
    "right",
    "log_file",
    "output_dir",
    "html_output",
    "predictions",
    "references",
    "manifest_out",
    "state_db",
    "prometheus_file",
    "otel_file",
}


def _location(path: Path, parts: tuple[str, ...]) -> str:
    return f"{path}:{'.'.join(parts)}"


def _valid_type(value: Any, expected: tuple[type, ...]) -> bool:
    if isinstance(value, bool) and bool not in expected:
        return False
    return isinstance(value, expected)


def validate_config(data: Mapping[str, Any], path: Path) -> None:
    """Validate the complete document before any command side effect."""
    allowed_root = {"schema_version", "doctor", "data", "run", "eval", "manifest"}
    unknown_root = sorted(set(data) - allowed_root)
    if unknown_root:
        key = unknown_root[0]
        raise ConfigError(
            f"unknown field {key!r}",
            _location(path, (key,)),
            f"remove it; allowed top-level fields are: {', '.join(sorted(allowed_root))}",
        )
    if "schema_version" not in data:
        raise ConfigError(
            "required field 'schema_version' is missing",
            _location(path, ("schema_version",)),
            f'add "schema_version": {SCHEMA_VERSION}',
        )
    version = data["schema_version"]
    if not isinstance(version, int) or isinstance(version, bool):
        raise ConfigError(
            "'schema_version' must be an integer",
            _location(path, ("schema_version",)),
            f"set it to {SCHEMA_VERSION}",
        )
    if version != SCHEMA_VERSION:
        raise ConfigError(
            f"schema version {version} is not supported",
            _location(path, ("schema_version",)),
            f"use schema_version {SCHEMA_VERSION}",
        )

    containers = {
        ("data",): {"check", "inventory", "compare"},
        ("run",): {"watch", "check", "compare"},
    }
    for parts, allowed in containers.items():
        value = data.get(parts[0])
        if value is None:
            continue
        if not isinstance(value, dict):
            raise ConfigError(
                f"{parts[0]!r} must be an object",
                _location(path, parts),
                "replace it with an object containing command settings",
            )
        unknown = sorted(set(value) - allowed)
        if unknown:
            key = unknown[0]
            raise ConfigError(
                f"unknown field {key!r}",
                _location(path, parts + (key,)),
                f"remove it; allowed fields are: {', '.join(sorted(allowed))}",
            )

    for section_parts, fields in SECTION_FIELDS.items():
        section: Any = data
        present = True
        for part in section_parts:
            if not isinstance(section, dict) or part not in section:
                present = False
                break
            section = section[part]
        if not present:
            continue
        if not isinstance(section, dict):
            raise ConfigError(
                f"{'.'.join(section_parts)!r} must be an object",
                _location(path, section_parts),
                "replace it with an object containing command settings",
            )
        unknown = sorted(set(section) - set(fields))
        if unknown:
            key = unknown[0]
            raise ConfigError(
                f"unknown field {key!r}",
                _location(path, section_parts + (key,)),
                f"remove it; allowed fields are: {', '.join(sorted(fields))}",
            )
        for key, value in section.items():
            if not _valid_type(value, fields[key]):
                names = ", ".join(t.__name__ for t in fields[key])
                raise ConfigError(
                    f"field {key!r} has type {type(value).__name__}; expected {names}",
                    _location(path, section_parts + (key,)),
                    f"set {key!r} to a value of the expected type",
                )
            if key == "keywords" and not all(isinstance(item, str) for item in value):
                raise ConfigError(
                    "'keywords' entries must all be strings",
                    _location(path, section_parts + (key,)),
                    "remove or quote non-string entries",
                )
            if (
                key
                in {
                    "max_image_pixels",
                    "max_media_files",
                    "max_scan_bytes",
                    "notification_every",
                    "step_stall_seconds",
                    "gpu_overheat_celsius",
                    "checkpoint_stale_seconds",
                }
                and value is not None
                and value <= 0
            ):
                raise ConfigError(
                    f"field {key!r} must be positive or null",
                    _location(path, section_parts + (key,)),
                    f"set {key!r} to a positive integer, or null to disable the budget",
                )
            if key == "training_type" and str(value).lower() not in {
                "auto",
                "peft",
                "lora",
                "qlora",
                "full",
            }:
                raise ConfigError(
                    f"unsupported training_type {value!r}",
                    _location(path, section_parts + (key,)),
                    "choose auto, peft, lora, qlora, or full",
                )


def _section(data: Mapping[str, Any], parts: tuple[str, ...]) -> Dict[str, Any]:
    value: Any = data
    for part in parts:
        if not isinstance(value, dict):
            return {}
        value = value.get(part, {})
    return dict(value) if isinstance(value, dict) else {}


def _resolve_config_path(value: str, base: Path) -> str:
    path = Path(value)
    if (
        path.is_absolute()
        or PureWindowsPath(value).is_absolute()
        or PurePosixPath(value).is_absolute()
    ):
        return value
    return str((base / path).resolve())


def resolve_command_config(
    parts: tuple[str, ...],
    config_path: Optional[Path],
    cli_values: Mapping[str, Any],
) -> Dict[str, Any]:
    """Resolve defaults < validated file < explicit CLI settings."""
    data: Dict[str, Any] = {}
    file_values: Dict[str, Any] = {}
    if config_path is not None:
        config_path = config_path.resolve()
        data = load_config_file(config_path)
        validate_config(data, config_path)
        file_values = _section(data, parts)

    result = deepcopy(DEFAULTS.get(parts, {}))
    result.update(file_values)
    allowed = SECTION_FIELDS[parts]
    explicit = {key: value for key, value in cli_values.items() if key in allowed}
    result.update(explicit)

    if config_path is not None:
        for key in PATH_FIELDS & set(file_values):
            if key not in explicit and isinstance(result.get(key), str):
                result[key] = _resolve_config_path(result[key], config_path.parent)

    for key in REQUIRED.get(parts, ()):
        if not result.get(key):
            location = (
                _location(config_path, parts + (key,))
                if config_path is not None
                else f"command line: --{key.replace('_', '-')}"
            )
            raise ConfigError(
                f"required field {key!r} is missing",
                location,
                f"set {'.'.join(parts + (key,))} in the config or pass --{key.replace('_', '-')}",
            )
    return result


def resolve_inline_config(
    parts: tuple[str, ...],
    values: Mapping[str, Any],
) -> Dict[str, Any]:
    """Resolve an API mapping with the same defaults and schema as the CLI."""
    if parts not in SECTION_FIELDS:
        raise ConfigError(
            "unsupported command section",
            f"inline:{'.'.join(parts)}",
            "use a documented public API command section",
        )
    allowed = SECTION_FIELDS[parts]
    unknown = sorted(set(values) - set(allowed))
    if unknown:
        raise ConfigError(
            "unknown configuration field",
            f"inline:{'.'.join((*parts, unknown[0]))}",
            "remove the field or use the documented public field name",
        )
    normalized = {
        key: str(value) if key in PATH_FIELDS and isinstance(value, Path) else value
        for key, value in values.items()
    }
    result = deepcopy(DEFAULTS.get(parts, {}))
    result.update(normalized)

    document: Dict[str, Any] = {"schema_version": SCHEMA_VERSION}
    cursor = document
    for part in parts[:-1]:
        child: Dict[str, Any] = {}
        cursor[part] = child
        cursor = child
    cursor[parts[-1]] = result
    validate_config(document, Path("inline-config.json"))
    return result
