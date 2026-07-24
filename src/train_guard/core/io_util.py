"""Shared IO helpers."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple


def utc_now_iso() -> str:
    """UTC ISO8601 timestamp."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def run_command(args: Sequence[str], timeout: float = 30.0) -> Tuple[int, str, str]:
    """Run subprocess without shell=True."""
    try:
        proc = subprocess.run(
            list(args),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            shell=False,
        )
        return proc.returncode, proc.stdout or "", proc.stderr or ""
    except FileNotFoundError:
        return 127, "", f"Command not found: {args[0]}"
    except subprocess.TimeoutExpired:
        return 124, "", f"Command timed out ({timeout}s): {' '.join(args)}"
    except OSError as exc:
        return 1, "", f"Command failed: {exc}"


def write_json(path: Path, data: Any, *, overwrite: bool = False) -> None:
    """Write UTF-8 JSON (ensure_ascii=False)."""
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def append_jsonl(path: Path, record: Mapping[str, Any]) -> None:
    """Append one JSONL record."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(dict(record), ensure_ascii=False) + "\n")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """File SHA256 hex digest."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def get_disk_usage(path: Path) -> Dict[str, Any]:
    """Disk usage for a mount/path."""
    try:
        usage = shutil.disk_usage(str(path))
        return {
            "path": str(path),
            "total_bytes": usage.total,
            "used_bytes": usage.used,
            "free_bytes": usage.free,
            "free_gb": round(usage.free / (1024**3), 2),
            "total_gb": round(usage.total / (1024**3), 2),
            "used_percent": round(usage.used / usage.total * 100, 2) if usage.total else 0.0,
            "ok": True,
        }
    except OSError as exc:
        return {"path": str(path), "ok": False, "error": str(exc)}


def get_cpu_load() -> Dict[str, Any]:
    """CPU load averages when available (Unix); Windows returns ok=False."""
    try:
        load1, load5, load15 = os.getloadavg()
        return {"load1": load1, "load5": load5, "load15": load15, "ok": True}
    except (OSError, AttributeError) as exc:
        return {"ok": False, "error": str(exc) or "getloadavg unavailable"}


def get_memory_info() -> Dict[str, Any]:
    """Memory info via psutil or /proc/meminfo."""
    from .optional import try_import_psutil

    psutil = try_import_psutil()
    if psutil is not None:
        vm = psutil.virtual_memory()
        return {
            "total_bytes": int(vm.total),
            "available_bytes": int(vm.available),
            "used_bytes": int(vm.used),
            "used_percent": float(vm.percent),
            "ok": True,
            "source": "psutil",
        }
    if platform.system() != "Linux":
        return {"ok": False, "error": "meminfo unavailable without psutil on this OS"}
    info: Dict[str, int] = {}
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as fh:
            for line in fh:
                parts = line.split(":")
                if len(parts) != 2:
                    continue
                key = parts[0].strip()
                value = parts[1].strip().split()[0]
                try:
                    info[key] = int(value) * 1024
                except ValueError:
                    continue
        total = info.get("MemTotal", 0)
        available = info.get("MemAvailable", info.get("MemFree", 0))
        used = max(total - available, 0)
        return {
            "total_bytes": total,
            "available_bytes": available,
            "used_bytes": used,
            "used_percent": round(used / total * 100, 2) if total else 0.0,
            "ok": True,
            "source": "proc",
        }
    except OSError as exc:
        return {"ok": False, "error": str(exc)}


def pid_alive(pid: int) -> bool:
    """Return True if process exists."""
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
