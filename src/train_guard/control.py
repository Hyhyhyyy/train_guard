"""Explicitly authorized control requests for supervised training processes."""

from __future__ import annotations

import hashlib
import hmac
import secrets
import time
from dataclasses import dataclass, field
from typing import Any, Mapping
from urllib.parse import urlparse

CONTROL_ACTIONS = frozenset(
    {
        "pause",
        "resume",
        "graceful_stop",
        "terminate",
        "validated_restart",
    }
)


@dataclass(frozen=True)
class ControlRequest:
    command_id: str
    run_id: str
    action: str
    created_at: float
    expires_at: float
    parameters: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        run_id: str,
        action: str,
        *,
        ttl_seconds: float = 30.0,
        parameters: Mapping[str, Any] | None = None,
        now: float | None = None,
        command_id: str | None = None,
    ) -> "ControlRequest":
        normalized_run_id = run_id.strip()
        if not normalized_run_id:
            raise ValueError("run_id is required")
        if action not in CONTROL_ACTIONS:
            raise ValueError(f"unsupported control action: {action}")
        if ttl_seconds <= 0 or ttl_seconds > 300:
            raise ValueError("control command ttl must be between 0 and 300 seconds")
        created_at = time.time() if now is None else now
        return cls(
            command_id=command_id or secrets.token_hex(16),
            run_id=normalized_run_id,
            action=action,
            created_at=created_at,
            expires_at=created_at + ttl_seconds,
            parameters=dict(parameters or {}),
        )

    def expired(self, now: float | None = None) -> bool:
        return (time.time() if now is None else now) >= self.expires_at

    def to_dict(self) -> dict[str, Any]:
        return {
            "command_id": self.command_id,
            "run_id": self.run_id,
            "action": self.action,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "parameters": dict(self.parameters),
        }


class ControlToken:
    """An in-memory token whose digest is safe to retain for comparisons."""

    def __init__(self, token: str | None = None) -> None:
        self._plain = token or secrets.token_urlsafe(32)
        self._digest = hashlib.sha256(self._plain.encode("utf-8")).digest()

    @property
    def plain(self) -> str:
        return self._plain

    def verify(self, candidate: str) -> bool:
        digest = hashlib.sha256(candidate.encode("utf-8")).digest()
        return hmac.compare_digest(self._digest, digest)


def bearer_token(header: str | None) -> str:
    if not header:
        return ""
    scheme, separator, value = header.partition(" ")
    if not separator or scheme.lower() != "bearer":
        return ""
    return value.strip()


def origin_is_local(origin: str | None, host: str, port: int) -> bool:
    if not origin:
        return False
    parsed = urlparse(origin)
    allowed_hosts = {host.lower(), "127.0.0.1", "localhost", "::1"}
    try:
        parsed_port = parsed.port
    except ValueError:
        return False
    return (
        parsed.scheme == "http"
        and parsed.hostname is not None
        and parsed.hostname.lower() in allowed_hosts
        and parsed_port == port
        and not parsed.username
        and not parsed.password
        and parsed.path in {"", "/"}
        and not parsed.params
        and not parsed.query
        and not parsed.fragment
    )


__all__ = [
    "CONTROL_ACTIONS",
    "ControlRequest",
    "ControlToken",
    "bearer_token",
    "origin_is_local",
]
