"""Privacy helpers: redact paths, hostnames, usernames, tokens."""

from __future__ import annotations

import getpass
import hashlib
import os
import re
import socket
from pathlib import Path
from typing import Any


_TOKEN_RE = re.compile(
    r"(?i)(api[_-]?key|token|secret|password|authorization)\s*[:=]\s*['\"]?([^\s'\"]+)"
)
_ABS_PATH_RE = re.compile(r"(?P<p>(?:[A-Za-z]:\\|/)[^\s\"']+)")


def path_token(path_like: str | Path) -> str:
    """Stable non-reversible token for a path basename."""
    name = Path(str(path_like)).name or "path"
    digest = hashlib.sha256(name.encode("utf-8")).hexdigest()[:12]
    ext = Path(name).suffix.lower()
    return f"path_{digest}{ext}"


def redact_text(text: str) -> str:
    """Redact absolute paths, username, hostname, and token-like values."""
    if not text:
        return text
    out = text
    try:
        user = getpass.getuser()
        if user:
            out = out.replace(user, "<user>")
    except Exception:  # noqa: BLE001
        pass
    try:
        host = socket.gethostname()
        if host:
            out = out.replace(host, "<host>")
    except Exception:  # noqa: BLE001
        pass
    home = os.path.expanduser("~")
    if home and home != "~":
        out = out.replace(home, "<home>")

    def _sub_path(m: re.Match[str]) -> str:
        return path_token(m.group("p"))

    out = _ABS_PATH_RE.sub(_sub_path, out)
    out = _TOKEN_RE.sub(r"\1=<redacted>", out)
    return out


def redact_value(value: Any) -> Any:
    """Recursively redact strings in JSON-like structures."""
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, dict):
        return {k: redact_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [redact_value(v) for v in value]
    return value


def group_id_hash(group_id: str) -> str:
    """Public hash for group_id (never emit raw id in public reports)."""
    digest = hashlib.sha256(str(group_id).encode("utf-8")).hexdigest()[:12]
    return f"group_{digest}"
