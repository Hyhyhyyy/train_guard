"""Separate training lifecycle and watcher observation events."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from ..core.io_util import append_jsonl, utc_now_iso
from ..core.privacy import redact_value


LIFECYCLE_SCHEMA_VERSION = 2
LIFECYCLE_FILENAME = "train_guard_lifecycle.jsonl"
LEGACY_TRAINING_EVENT_KINDS = frozenset({"start", "heartbeat", "checkpoint", "finish", "abort"})
TRAINING_EVENT_KINDS = frozenset(
    {"train_start", "train_heartbeat", "train_checkpoint", "train_finish", "train_abort"}
)
WATCH_EVENT_KINDS = frozenset(
    {"watch_start", "watch_heartbeat", "watch_checkpoint", "watch_stop", "watch_error"}
)
EVENT_KINDS = LEGACY_TRAINING_EVENT_KINDS | TRAINING_EVENT_KINDS | WATCH_EVENT_KINDS

# Derived run phase from the latest terminal-ish event, else last activity.
PHASE_NONE = "none"
PHASE_RUNNING = "running"
PHASE_CHECKPOINTED = "checkpointed"
PHASE_FINISHED = "finished"
PHASE_ABORTED = "aborted"


def lifecycle_path(output_dir: Path) -> Path:
    """Canonical lifecycle JSONL path inside a run output directory."""
    return Path(output_dir) / LIFECYCLE_FILENAME


def make_lifecycle_event(
    kind: str,
    *,
    framework: str = "generic",
    global_step: Optional[int] = None,
    checkpoints: Optional[Sequence[str]] = None,
    alert_codes: Optional[Sequence[str]] = None,
    message: str = "",
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a privacy-safe lifecycle event dict."""
    if kind not in EVENT_KINDS:
        raise ValueError(f"unsupported lifecycle event kind: {kind}")
    event: Dict[str, Any] = {
        "schema_version": LIFECYCLE_SCHEMA_VERSION,
        "kind": kind,
        "timestamp": utc_now_iso(),
        "framework": framework,
        "global_step": global_step,
        "checkpoints": list(checkpoints or []),
        "alert_codes": list(alert_codes or []),
        "message": message or "",
    }
    if extra:
        for key, value in extra.items():
            if key in event:
                continue
            event[key] = value
    return redact_value(event)


def append_lifecycle_event(path: Path, event: Dict[str, Any]) -> None:
    """Append one redacted lifecycle event."""
    append_jsonl(path, redact_value(event))


def iter_lifecycle_events(path: Path) -> Iterable[Dict[str, Any]]:
    """Yield valid lifecycle event objects from JSONL (skip bad lines)."""
    if not path.is_file():
        return
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            text = line.strip()
            if not text:
                continue
            try:
                obj = json.loads(text)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict) and obj.get("kind") in EVENT_KINDS:
                yield obj


def load_lifecycle_events(path: Path) -> List[Dict[str, Any]]:
    """Load all valid lifecycle events."""
    return list(iter_lifecycle_events(path))


def summarize_lifecycle(path: Path) -> Dict[str, Any]:
    """Derive a compact, privacy-safe lifecycle summary for check/manifest."""
    events = load_lifecycle_events(path)
    if not events:
        return {
            "present": path.is_file(),
            "event_count": 0,
            "phase": PHASE_NONE,
            "started_at": None,
            "finished_at": None,
            "last_heartbeat_at": None,
            "global_step": None,
            "checkpoints": [],
            "alert_codes": [],
            "has_abort": False,
            "has_finish": False,
            "has_training_events": False,
            "watcher": {
                "present": False,
                "running": False,
                "started_at": None,
                "stopped_at": None,
                "last_heartbeat_at": None,
                "has_error": False,
            },
        }

    started_at = None
    finished_at = None
    last_heartbeat_at = None
    global_step: Optional[int] = None
    checkpoints: List[str] = []
    alert_codes: List[str] = []
    has_abort = False
    has_finish = False
    has_training_events = False
    phase = PHASE_NONE
    watch_started_at = None
    watch_stopped_at = None
    watch_last_heartbeat_at = None
    watch_running = False
    watch_has_error = False

    for event in events:
        kind = event.get("kind")
        ts = event.get("timestamp")
        if kind in {"start", "train_start"} and started_at is None:
            has_training_events = True
            started_at = ts
            phase = PHASE_RUNNING
        elif kind in {"heartbeat", "train_heartbeat"}:
            has_training_events = True
            last_heartbeat_at = ts
            if phase not in {PHASE_FINISHED, PHASE_ABORTED}:
                phase = PHASE_RUNNING
        elif kind in {"checkpoint", "train_checkpoint"}:
            has_training_events = True
            if phase not in {PHASE_FINISHED, PHASE_ABORTED}:
                phase = PHASE_CHECKPOINTED
        elif kind in {"finish", "train_finish"}:
            has_training_events = True
            has_finish = True
            finished_at = ts
            phase = PHASE_FINISHED
        elif kind in {"abort", "train_abort"}:
            has_training_events = True
            has_abort = True
            finished_at = ts
            phase = PHASE_ABORTED
        elif kind == "watch_start":
            watch_started_at = watch_started_at or ts
            watch_running = True
        elif kind == "watch_heartbeat":
            watch_last_heartbeat_at = ts
        elif kind == "watch_stop":
            watch_stopped_at = ts
            watch_running = False
        elif kind == "watch_error":
            watch_has_error = True
            watch_stopped_at = ts
            watch_running = False

        step = event.get("global_step")
        if isinstance(step, int):
            global_step = step
        elif step is not None:
            try:
                global_step = int(step)
            except (TypeError, ValueError):
                pass

        names = event.get("checkpoints") or []
        if isinstance(names, list):
            for name in names:
                if isinstance(name, str) and name and name not in checkpoints:
                    checkpoints.append(name)

        codes = event.get("alert_codes") or []
        if isinstance(codes, list):
            for code in codes:
                if isinstance(code, str) and code and code not in alert_codes:
                    alert_codes.append(code)

    return {
        "present": True,
        "event_count": len(events),
        "phase": phase,
        "started_at": started_at,
        "finished_at": finished_at,
        "last_heartbeat_at": last_heartbeat_at,
        "global_step": global_step,
        "checkpoints": checkpoints,
        "alert_codes": alert_codes,
        "has_abort": has_abort,
        "has_finish": has_finish,
        "has_training_events": has_training_events,
        "watcher": {
            "present": watch_started_at is not None,
            "running": watch_running,
            "started_at": watch_started_at,
            "stopped_at": watch_stopped_at,
            "last_heartbeat_at": watch_last_heartbeat_at,
            "has_error": watch_has_error,
        },
    }


def checkpoint_delta(previous: Sequence[str], current: Sequence[str]) -> List[str]:
    """Return newly appeared checkpoint directory names."""
    prev = set(previous)
    return [name for name in current if name not in prev]
