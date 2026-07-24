"""Versioned reliability-domain models shared by all Train Guard components."""

from __future__ import annotations

import json
import math
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Mapping, Optional, Tuple

SCHEMA_VERSION = "1.0"


def json_safe(value: Any) -> Any:
    """Return deterministic, standards-compliant JSON data.

    Non-finite floats are represented as strings because RFC 8259 has no NaN
    or Infinity values. Unsupported objects are rejected instead of being
    silently stringified.
    """
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        if math.isinf(value):
            return "Infinity" if value > 0 else "-Infinity"
        return value
    if isinstance(value, Enum):
        return json_safe(value.value)
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    raise TypeError(f"value of type {type(value).__name__} is not JSON serializable")


def utc_now() -> str:
    """Return an RFC3339 UTC timestamp."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class Severity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class EventKind(str, Enum):
    NAN_INF = "nan_inf"
    LOSS_SPIKE = "loss_spike"
    GRAD_SPIKE = "grad_spike"
    STEP_STALLED = "step_stalled"
    THROUGHPUT_DROP = "throughput_drop"
    GPU_IDLE = "gpu_idle"
    GPU_OVERHEAT = "gpu_overheat"
    DISK_LOW = "disk_low"
    PROCESS_DEAD = "process_dead"
    CUDA_OOM = "cuda_oom"
    NCCL_ERROR = "nccl_error"
    GPU_XID = "gpu_xid"
    CHECKPOINT_STALE = "checkpoint_stale"
    CHECKPOINT_CORRUPT = "checkpoint_corrupt"


@dataclass(frozen=True)
class Event:
    """Immutable, versioned observation or rule result."""

    run_id: str
    kind: EventKind
    severity: Severity
    message: str
    source: str = "unknown"
    timestamp: str = field(default_factory=utc_now)
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    schema_version: str = SCHEMA_VERSION
    step: Optional[int] = None
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["kind"] = self.kind.value
        data["severity"] = self.severity.value
        data["attributes"] = dict(self.attributes)
        return dict(json_safe(data))

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, allow_nan=False)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Event":
        return cls(
            run_id=str(data["run_id"]),
            kind=EventKind(str(data["kind"])),
            severity=Severity(str(data["severity"])),
            message=str(data["message"]),
            source=str(data.get("source", "unknown")),
            timestamp=str(data.get("timestamp", utc_now())),
            event_id=str(data.get("event_id", uuid.uuid4())),
            schema_version=str(data.get("schema_version", SCHEMA_VERSION)),
            step=int(data["step"]) if data.get("step") is not None else None,
            attributes=dict(data.get("attributes", {})),
        )


@dataclass(frozen=True)
class Alert:
    """Deduplicated lifecycle state for an event fingerprint."""

    alert_id: str
    run_id: str
    fingerprint: str
    event: Event
    opened_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)
    occurrence_count: int = 1
    resolved_at: Optional[str] = None
    schema_version: str = SCHEMA_VERSION

    @property
    def active(self) -> bool:
        return self.resolved_at is None


@dataclass(frozen=True)
class Diagnostic:
    """Evidence-backed explanation attached to an event."""

    event_id: str
    summary: str
    evidence: Tuple[str, ...] = ()
    probable_causes: Tuple[str, ...] = ()
    recommendations: Tuple[str, ...] = ()
    schema_version: str = SCHEMA_VERSION


__all__ = [
    "SCHEMA_VERSION",
    "Alert",
    "Diagnostic",
    "Event",
    "EventKind",
    "Severity",
    "json_safe",
    "utc_now",
]
