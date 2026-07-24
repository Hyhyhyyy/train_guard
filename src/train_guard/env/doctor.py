"""doctor and bundle-info environment commands."""

from __future__ import annotations

import json
import os
import platform
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from .. import __min_python__, __version__
from ..core.events import CheckItem, overall_status
from ..core.exitcodes import EXIT_FAIL, EXIT_OK, EXIT_RUNTIME, EXIT_WARN
from ..core.io_util import get_disk_usage, sha256_file, utc_now_iso, write_json
from ..core.optional import package_version, try_import_torch
from ..core.privacy import redact_value
from ..run.commands import query_nvidia_smi


def run_doctor(
    model_path: Optional[Path] = None,
    expected_gpus: Optional[int] = None,
) -> Dict[str, Any]:
    """Read-only environment and model integrity check."""
    items: List[CheckItem] = []
    py_ver = sys.version_info
    items.append(
        CheckItem(
            "python_version",
            "PASS" if py_ver >= __min_python__ else "FAIL",
            f"Python {platform.python_version()} (requires >= {__min_python__[0]}.{__min_python__[1]})",
            {"executable": sys.executable},
        )
    )
    items.append(CheckItem("python_executable", "INFO", f"Interpreter: {sys.executable}", {}))
    items.append(
        CheckItem(
            "os",
            "PASS" if platform.system() in {"Linux", "Windows", "Darwin"} else "WARN",
            f"{platform.system()} {platform.release()} ({platform.machine()})",
            {},
        )
    )
    conda_env = os.environ.get("CONDA_DEFAULT_ENV") or os.environ.get("CONDA_PREFIX")
    items.append(
        CheckItem(
            "conda_env",
            "INFO" if conda_env else "WARN",
            f"Conda env: {conda_env}" if conda_env else "No Conda env detected",
            {},
        )
    )

    torch = try_import_torch()
    if torch is None:
        items.append(
            CheckItem("pytorch", "WARN", "PyTorch not installed (optional for doctor)", {})
        )
        items.append(
            CheckItem("torch_cuda_available", "WARN", "Cannot check CUDA without PyTorch", {})
        )
    else:
        cuda_compiled = getattr(getattr(torch, "version", None), "cuda", None)
        cuda_ok = bool(torch.cuda.is_available())
        items.append(
            CheckItem(
                "pytorch",
                "PASS",
                f"PyTorch {getattr(torch, '__version__', '?')}, compiled CUDA {cuda_compiled}",
                {},
            )
        )
        items.append(
            CheckItem(
                "torch_cuda_available",
                "PASS" if cuda_ok else "FAIL",
                f"torch.cuda.is_available = {cuda_ok}",
                {},
            )
        )

    smi = query_nvidia_smi()
    if smi["ok"]:
        items.append(
            CheckItem(
                "nvidia_smi", "PASS", f"nvidia-smi ok, driver {smi.get('driver_version')}", {}
            )
        )
        names = [f"{g['index']}:{g['name']}({g['memory_total_mb']:.0f}MB)" for g in smi["gpus"]]
        status = "PASS"
        msg = f"Detected {smi['count']} GPU(s): {', '.join(names)}"
        if expected_gpus is not None and smi["count"] != expected_gpus:
            status = "WARN"
            msg += f"; expected {expected_gpus}"
        items.append(CheckItem("gpus", status, msg, {"count": smi["count"]}))
    else:
        items.append(
            CheckItem("nvidia_smi", "WARN", f"nvidia-smi unavailable: {smi.get('error')}", {})
        )
        items.append(CheckItem("gpus", "WARN", "Cannot enumerate GPUs", {}))

    for name in (
        "transformers",
        "peft",
        "datasets",
        "llamafactory",
        "torchvision",
        "torchaudio",
        "Pillow",
        "PyYAML",
        "psutil",
    ):
        ver = package_version(name)
        items.append(
            CheckItem(
                f"pkg_{name}",
                "INFO" if ver else "WARN",
                f"{name}={ver}" if ver else f"{name} not installed",
                {},
            )
        )

    if model_path is not None:
        if not model_path.exists():
            items.append(
                CheckItem("model_path", "FAIL", f"Model dir missing: {model_path.name}", {})
            )
        else:
            items.append(
                CheckItem("model_path", "PASS", f"Model dir exists: {model_path.name}", {})
            )
            config_json = model_path / "config.json"
            if config_json.exists():
                try:
                    cfg = json.loads(config_json.read_text(encoding="utf-8"))
                    items.append(
                        CheckItem(
                            "model_config",
                            "PASS",
                            f"config.json ok, model_type={cfg.get('model_type')}",
                            {},
                        )
                    )
                except (OSError, json.JSONDecodeError) as exc:
                    items.append(
                        CheckItem("model_config", "FAIL", f"config.json parse error: {exc}", {})
                    )
            else:
                items.append(CheckItem("model_config", "WARN", "config.json not found", {}))
            shards = sorted(model_path.glob("*.safetensors"))
            items.append(
                CheckItem(
                    "safetensors",
                    "PASS" if shards else "WARN",
                    f"{len(shards)} safetensors shard(s)" if shards else "No *.safetensors shards",
                    {},
                )
            )
            index_file = model_path / "model.safetensors.index.json"
            if index_file.exists():
                try:
                    index_data = json.loads(index_file.read_text(encoding="utf-8"))
                    referenced = sorted(set((index_data.get("weight_map") or {}).values()))
                    missing = [n for n in referenced if not (model_path / n).exists()]
                    if missing:
                        items.append(
                            CheckItem(
                                "safetensors_index", "FAIL", f"Missing {len(missing)} shard(s)", {}
                            )
                        )
                    else:
                        items.append(
                            CheckItem(
                                "safetensors_index",
                                "PASS",
                                f"Index references {len(referenced)} complete shard(s)",
                                {},
                            )
                        )
                except (OSError, json.JSONDecodeError) as exc:
                    items.append(
                        CheckItem("safetensors_index", "FAIL", f"Index parse error: {exc}", {})
                    )

    for mount in (Path("/"), Path.cwd()):
        usage = get_disk_usage(mount)
        if usage.get("ok"):
            free_gb = usage["free_gb"]
            status = "PASS" if free_gb >= 20 else ("WARN" if free_gb >= 5 else "FAIL")
            items.append(
                CheckItem(f"disk_{mount.name or 'root'}", status, f"{mount} free {free_gb} GB", {})
            )

    overall = overall_status(items)
    return redact_value(
        {
            "tool": "train_guard",
            "command": "doctor",
            "version": __version__,
            "timestamp": utc_now_iso(),
            "overall_status": overall,
            "checks": [asdict(i) for i in items],
        }
    )


def status_to_exit(overall: str) -> int:
    """Map overall status to exit code."""
    if overall == "FAIL":
        return EXIT_FAIL
    if overall == "WARN":
        return EXIT_WARN
    return EXIT_OK


def run_bundle_info(self_path: Path) -> Dict[str, Any]:
    """Bundle / deploy metadata."""
    digest = None
    try:
        if self_path.is_file():
            digest = sha256_file(self_path)
    except OSError:
        digest = None
    return {
        "name": "Train Guard",
        "version": __version__,
        "min_python": f"{__min_python__[0]}.{__min_python__[1]}",
        "file": self_path.name,
        "sha256": digest,
        "commands": [
            "init",
            "doctor",
            "data check",
            "data inventory",
            "data compare",
            "run watch",
            "run snapshot",
            "run check",
            "run compare",
            "run status",
            "run supervise",
            "show",
            "tui",
            "eval",
            "manifest",
            "bundle-info",
        ],
        "optional_dependencies": {
            "PyYAML": package_version("PyYAML"),
            "Pillow": package_version("Pillow"),
            "psutil": package_version("psutil"),
        },
        "deploy": (
            "Copy release/train_guard.py to another Linux/Windows host and run:\n"
            "  python train_guard.py init --output train-guard.json\n"
            "  python train_guard.py doctor --config train-guard.json\n"
            "Do not auto-install or upgrade torch/CUDA/transformers/peft/LLaMAFactory."
        ),
        "note": (
            "Read-only by default; process restart requires explicit --restart, "
            "a bounded budget, and checkpoint validation."
        ),
    }
