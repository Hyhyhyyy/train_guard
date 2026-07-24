"""Dependency-free event sinks and export formats."""

from __future__ import annotations

import re
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import IO, Any, Iterable, Mapping, Optional, Protocol

from train_guard.core.io_util import append_jsonl, atomic_write_text
from train_guard.domain import Event


class Sink(Protocol):
    def emit(self, event: Event) -> None: ...


class ConsoleSink:
    def __init__(self, stream: Optional[IO[str]] = None) -> None:
        self.stream = stream or sys.stderr

    def emit(self, event: Event) -> None:
        self.stream.write(f"[{event.severity.value.upper()}] {event.kind.value}: {event.message}\n")
        self.stream.flush()


class JsonlSink:
    def __init__(self, path: Path, max_bytes: Optional[int] = 10 * 1024**2) -> None:
        self.path = path
        self.max_bytes = max_bytes

    def emit(self, event: Event) -> None:
        append_jsonl(
            self.path,
            event.to_dict(),
            max_bytes=self.max_bytes,
        )


class WebhookSink:
    """POST events as JSON using urllib; callers control the destination."""

    def __init__(
        self,
        url: str,
        timeout: float = 5.0,
        headers: Optional[Mapping[str, str]] = None,
        retries: int = 2,
        backoff_seconds: float = 0.25,
    ) -> None:
        if not url.startswith(("http://", "https://")):
            raise ValueError("webhook URL must use http or https")
        self.url = url
        self.timeout = timeout
        self.headers = {"Content-Type": "application/json", **dict(headers or {})}
        if retries < 0 or backoff_seconds < 0:
            raise ValueError("webhook retry settings must be non-negative")
        self.retries = retries
        self.backoff_seconds = backoff_seconds

    def emit(self, event: Event) -> None:
        request = urllib.request.Request(
            self.url,
            data=event.to_json().encode("utf-8"),
            headers=self.headers,
            method="POST",
        )
        for attempt in range(self.retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    if not 200 <= response.status < 300:
                        raise RuntimeError(f"webhook returned HTTP {response.status}")
                return
            except (OSError, RuntimeError, TimeoutError, urllib.error.URLError):
                if attempt >= self.retries:
                    raise
                time.sleep(self.backoff_seconds * (2**attempt))


_PROM_SAFE = re.compile(r"[^a-zA-Z0-9_:]")


def _render_prometheus_counts(counts: Mapping[tuple[str, str], int], metric_prefix: str) -> str:
    name = _PROM_SAFE.sub("_", metric_prefix) + "_events_total"
    lines = [f"# HELP {name} Reliability events.", f"# TYPE {name} counter"]
    for (kind, severity), count in sorted(counts.items()):
        lines.append(f'{name}{{kind="{kind}",severity="{severity}"}} {count}')
    return "\n".join(lines) + "\n"


def prometheus_text(events: Iterable[Event], metric_prefix: str = "train_guard") -> str:
    """Render event counters in Prometheus text exposition format."""
    counts: dict[tuple[str, str], int] = {}
    for event in events:
        key = (event.kind.value, event.severity.value)
        counts[key] = counts.get(key, 0) + 1
    return _render_prometheus_counts(counts, metric_prefix)


class PrometheusFileSink:
    """Maintain a local Prometheus textfile for node-exporter collection."""

    def __init__(self, path: Path, metric_prefix: str = "train_guard") -> None:
        self.path = path
        self.metric_prefix = metric_prefix
        self._counts: dict[tuple[str, str], int] = {}
        self._lock = threading.Lock()

    def emit(self, event: Event) -> None:
        key = (event.kind.value, event.severity.value)
        with self._lock:
            self._counts[key] = self._counts.get(key, 0) + 1
            atomic_write_text(
                self.path,
                _render_prometheus_counts(self._counts, self.metric_prefix),
                overwrite=True,
            )


def otel_json(event: Event) -> Mapping[str, Any]:
    """Map an event to a stable OpenTelemetry-compatible log record."""
    return {
        "timeUnixNano": _rfc3339_to_unix_nano(event.timestamp),
        "severityText": event.severity.value.upper(),
        "body": {"stringValue": event.message},
        "attributes": [
            {"key": "train_guard.event_id", "value": {"stringValue": event.event_id}},
            {"key": "train_guard.run_id", "value": {"stringValue": event.run_id}},
            {"key": "train_guard.kind", "value": {"stringValue": event.kind.value}},
            {"key": "train_guard.source", "value": {"stringValue": event.source}},
        ],
    }


def _rfc3339_to_unix_nano(timestamp: str) -> str:
    from datetime import datetime

    parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    return str(int(parsed.timestamp() * 1_000_000_000))


class OtelJsonSink:
    def __init__(self, path: Path, max_bytes: Optional[int] = 10 * 1024**2) -> None:
        self.path = path
        self.max_bytes = max_bytes

    def emit(self, event: Event) -> None:
        append_jsonl(
            self.path,
            otel_json(event),
            max_bytes=self.max_bytes,
        )


__all__ = [
    "ConsoleSink",
    "JsonlSink",
    "OtelJsonSink",
    "PrometheusFileSink",
    "Sink",
    "WebhookSink",
    "otel_json",
    "prometheus_text",
]
