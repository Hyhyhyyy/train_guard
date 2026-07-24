"""Localhost-only observability dashboard and authenticated control API."""

from __future__ import annotations

import json
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable, Optional, Type
from urllib.parse import parse_qs, urlparse

from train_guard.control import (
    ControlRequest,
    ControlToken,
    bearer_token,
    origin_is_local,
)
from train_guard.state import StateStore
from train_guard.status import build_status_snapshot

from .assets import CSS, HTML, JS

_MAX_REQUEST_BYTES = 32 * 1024
_LOOPBACK_CLIENTS = frozenset({"127.0.0.1", "::1"})


@dataclass(frozen=True)
class DashboardOptions:
    host: str = "127.0.0.1"
    port: int = 8765
    enable_control: bool = False
    control_token: Optional[ControlToken] = None

    def __post_init__(self) -> None:
        if self.host not in {"127.0.0.1", "::1", "localhost"}:
            raise ValueError("dashboard must bind to localhost")
        if not 0 <= self.port <= 65535:
            raise ValueError("dashboard port must be between 0 and 65535")
        if self.enable_control and self.control_token is None:
            raise ValueError("control mode requires an in-memory control token")


def make_handler(
    store: StateStore,
    options: Optional[DashboardOptions] = None,
) -> Type[BaseHTTPRequestHandler]:
    configured = options or DashboardOptions()

    class DashboardHandler(BaseHTTPRequestHandler):
        server_version = "TrainGuardDashboard/2"

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self._send(HTTPStatus.OK, "text/html; charset=utf-8", HTML.encode())
                return
            if parsed.path == "/assets/app.css":
                self._send(HTTPStatus.OK, "text/css; charset=utf-8", CSS.encode())
                return
            if parsed.path == "/assets/app.js":
                self._send(
                    HTTPStatus.OK,
                    "text/javascript; charset=utf-8",
                    JS.encode(),
                )
                return
            if parsed.path == "/api/status":
                run_id = parse_qs(parsed.query).get("run_id", [None])[0]
                snapshot = build_status_snapshot(
                    store,
                    run_id,
                    control_enabled=configured.enable_control,
                )
                self._send_json(HTTPStatus.OK, snapshot.to_dict())
                return
            if parsed.path == "/api/alerts":
                run_id = parse_qs(parsed.query).get("run_id", [None])[0]
                self._send_json(
                    HTTPStatus.OK,
                    list(store.active_alerts(run_id)),
                )
                return
            if parsed.path == "/healthz":
                self._send_json(HTTPStatus.OK, {"ok": True})
                return
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})

        def do_POST(self) -> None:
            if urlparse(self.path).path != "/api/commands":
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
                return
            if not configured.enable_control or configured.control_token is None:
                self._send_json(
                    HTTPStatus.FORBIDDEN,
                    {"error": "dashboard control mode is disabled"},
                )
                return
            if self.client_address[0] not in _LOOPBACK_CLIENTS:
                self._send_json(
                    HTTPStatus.FORBIDDEN,
                    {"error": "control requests must originate from loopback"},
                )
                return
            token = bearer_token(self.headers.get("Authorization"))
            if not configured.control_token.verify(token):
                self._send_json(HTTPStatus.UNAUTHORIZED, {"error": "invalid control token"})
                return
            if not origin_is_local(
                self.headers.get("Origin"),
                configured.host,
                configured.port,
            ):
                self._send_json(HTTPStatus.FORBIDDEN, {"error": "invalid request origin"})
                return
            try:
                payload = self._read_json()
                ttl_value = payload.get("ttl_seconds", 30.0)
                parameters_value = payload.get("parameters")
                parameters = dict(parameters_value) if isinstance(parameters_value, dict) else {}
                request = ControlRequest.create(
                    str(payload.get("run_id") or ""),
                    str(payload.get("action") or ""),
                    ttl_seconds=float(str(ttl_value)),
                    parameters=parameters,
                    command_id=str(payload["command_id"]) if payload.get("command_id") else None,
                )
                queued = getattr(store, "enqueue_control", None)
                if queued is None:
                    raise RuntimeError("control queue is unavailable")
                accepted = bool(queued(request))
            except (KeyError, TypeError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
                self._send_json(
                    HTTPStatus.BAD_REQUEST,
                    {"error": type(exc).__name__},
                )
                return
            self._send_json(
                HTTPStatus.ACCEPTED if accepted else HTTPStatus.OK,
                {
                    "command_id": request.command_id,
                    "status": "queued" if accepted else "already_queued",
                },
            )

        def _read_json(self) -> dict[str, object]:
            raw_length = self.headers.get("Content-Length", "0")
            length = int(raw_length)
            if length <= 0 or length > _MAX_REQUEST_BYTES:
                raise ValueError("invalid request size")
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("JSON body must be an object")
            return payload

        def _send_json(self, status: HTTPStatus, value: object) -> None:
            self._send(
                status,
                "application/json; charset=utf-8",
                json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8"),
            )

        def _send(self, status: HTTPStatus, content_type: str, body: bytes) -> None:
            self.send_response(int(status))
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; style-src 'self'; script-src 'self'; "
                "connect-src 'self'; frame-ancestors 'none'",
            )
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            del format, args

    return DashboardHandler


def create_server(
    store: StateStore,
    host: str = "127.0.0.1",
    port: int = 8765,
    server_factory: Callable[..., ThreadingHTTPServer] = ThreadingHTTPServer,
    *,
    enable_control: bool = False,
    authorization: Optional[ControlToken] = None,
) -> ThreadingHTTPServer:
    """Create but do not start a loopback-bound dashboard server."""
    options = DashboardOptions(host, port, enable_control, authorization)
    return server_factory((host, port), make_handler(store, options))


def serve(
    store: StateStore,
    host: str = "127.0.0.1",
    port: int = 8765,
    *,
    enable_control: bool = False,
    authorization: Optional[ControlToken] = None,
) -> None:
    server = create_server(
        store,
        host,
        port,
        enable_control=enable_control,
        authorization=authorization,
    )
    try:
        server.serve_forever()
    finally:
        server.server_close()


__all__ = [
    "CSS",
    "HTML",
    "JS",
    "DashboardOptions",
    "create_server",
    "make_handler",
    "serve",
]
