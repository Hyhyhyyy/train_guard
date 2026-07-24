"""Shared read model for CLI, Web dashboard, and terminal UI."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, Mapping, Optional

from .state import StateStore


@dataclass(frozen=True)
class StatusSnapshot:
    run_id: Optional[str]
    runs: tuple[str, ...] = ()
    phase: str = "unknown"
    latest_sample: Mapping[str, Any] = field(default_factory=dict)
    series: Mapping[str, tuple[float, ...]] = field(default_factory=dict)
    active_alerts: tuple[Mapping[str, Any], ...] = ()
    checkpoints: tuple[Mapping[str, Any], ...] = ()
    recoveries: tuple[Mapping[str, Any], ...] = ()
    managed_process: Mapping[str, Any] = field(default_factory=dict)
    control_enabled: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _call(store: StateStore, method: str, *args: object, default: Any) -> Any:
    candidate = getattr(store, method, None)
    if candidate is None:
        return default
    return candidate(*args)


def _as_tuple(values: Iterable[Mapping[str, Any]]) -> tuple[Mapping[str, Any], ...]:
    return tuple(dict(value) for value in values)


def build_status_snapshot(
    store: StateStore,
    run_id: Optional[str] = None,
    *,
    control_enabled: bool = False,
) -> StatusSnapshot:
    """Build the stable dashboard/TUI view from persistent state."""
    runs = tuple(str(value) for value in _call(store, "list_runs", default=()))
    selected = run_id or (runs[0] if runs else None)
    latest = dict(_call(store, "latest_sample", selected, default={}) if selected else {})
    phase = str(
        _call(store, "get_run_state", selected, "lifecycle.phase", "unknown", default="unknown")
        if selected
        else "unknown"
    )
    series_raw = _call(store, "metric_series", selected, default={}) if selected else {}
    series = {
        str(key): tuple(float(value) for value in values)
        for key, values in dict(series_raw).items()
    }
    alerts = _as_tuple(store.active_alerts(selected)) if selected else ()
    checkpoints = _as_tuple(
        _call(store, "checkpoint_history", selected, default=()) if selected else ()
    )
    recoveries = _as_tuple(
        _call(store, "recovery_history", selected, default=()) if selected else ()
    )
    process = dict(_call(store, "managed_process", selected, default={}) if selected else {})
    return StatusSnapshot(
        run_id=selected,
        runs=runs,
        phase=phase,
        latest_sample=latest,
        series=series,
        active_alerts=alerts,
        checkpoints=checkpoints,
        recoveries=recoveries,
        managed_process=process,
        control_enabled=control_enabled and bool(process),
    )


__all__ = ["StatusSnapshot", "build_status_snapshot"]
