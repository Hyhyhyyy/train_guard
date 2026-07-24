"""Shared IO helpers."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import platform
import shutil
import subprocess
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, Mapping, Optional, Sequence, Tuple


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


def atomic_write_text(
    path: Path, text: str, *, overwrite: bool = False, encoding: str = "utf-8"
) -> None:
    """Atomically replace a text file using a temporary file in the same directory."""
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_name = ""
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding=encoding,
            dir=str(path.parent),
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as fh:
            tmp_name = fh.name
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        if path.exists() and not overwrite:
            raise FileExistsError(f"Refusing to overwrite: {path}")
        os.replace(tmp_name, path)
        tmp_name = ""
    finally:
        if tmp_name:
            try:
                os.unlink(tmp_name)
            except FileNotFoundError:
                pass


def write_json(path: Path, data: Any, *, overwrite: bool = False) -> None:
    """Atomically write UTF-8 JSON (ensure_ascii=False)."""
    atomic_write_text(
        path,
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        overwrite=overwrite,
    )


@contextmanager
def _exclusive_file_lock(path: Path) -> Iterator[None]:
    """Use a sidecar byte lock for cross-process JSONL writers."""
    lock_path = path.with_name(f".{path.name}.lock")
    with lock_path.open("a+b") as handle:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()
        handle.seek(0)
        module = importlib.import_module("msvcrt" if os.name == "nt" else "fcntl")
        if os.name == "nt":
            module.locking(handle.fileno(), module.LK_LOCK, 1)
        else:
            module.flock(handle.fileno(), module.LOCK_EX)
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                module.locking(handle.fileno(), module.LK_UNLCK, 1)
            else:
                module.flock(handle.fileno(), module.LOCK_UN)


def _rotate_jsonl(path: Path, backup_count: int) -> None:
    if backup_count <= 0:
        path.unlink(missing_ok=True)
        return
    oldest = path.with_name(f"{path.name}.{backup_count}")
    oldest.unlink(missing_ok=True)
    for index in range(backup_count - 1, 0, -1):
        source = path.with_name(f"{path.name}.{index}")
        if source.exists():
            os.replace(source, path.with_name(f"{path.name}.{index + 1}"))
    if path.exists():
        os.replace(path, path.with_name(f"{path.name}.1"))


def append_jsonl(
    path: Path,
    record: Mapping[str, Any],
    *,
    max_bytes: Optional[int] = 10 * 1024**2,
    backup_count: int = 3,
) -> None:
    """Append one complete JSONL record with locking and optional rotation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(dict(record), ensure_ascii=False) + "\n").encode("utf-8")
    with _exclusive_file_lock(path):
        if (
            max_bytes is not None
            and max_bytes > 0
            and path.exists()
            and path.stat().st_size + len(payload) > max_bytes
        ):
            _rotate_jsonl(path, backup_count)
        fd = os.open(str(path), os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
        try:
            written = os.write(fd, payload)
            if written != len(payload):
                raise OSError("short JSONL append")
            os.fsync(fd)
        finally:
            os.close(fd)


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
        getloadavg = getattr(os, "getloadavg")
        load1, load5, load15 = getloadavg()
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


def _windows_pid_alive(pid: int) -> bool:
    import ctypes

    process_query_limited_information = 0x1000
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    handle = kernel32.OpenProcess(process_query_limited_information, False, pid)
    if handle:
        kernel32.CloseHandle(handle)
        return True
    return ctypes.get_last_error() == 5


def pid_alive(pid: int) -> bool:
    """Return True if a process exists without signaling it."""
    if pid <= 0:
        return False
    if os.name == "nt":
        return _windows_pid_alive(pid)
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
