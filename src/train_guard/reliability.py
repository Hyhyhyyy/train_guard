"""Integrated local reliability pipeline.

This module connects deterministic rules, persistent state, audit logging, and
notification sinks without requiring a service or third-party dependency.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .domain import Diagnostic, Event, json_safe
from .rules import RuleContext, RuleEngine
from .sinks import Sink
from .state import AuditLog, StateStore


@dataclass(frozen=True)
class AlertTransition:
    """Explicit lifecycle transition for one alert identity."""

    fingerprint: str
    state: str


@dataclass(frozen=True)
class ReliabilityResult:
    """Events and alert transitions produced for one observation."""

    events: Tuple[Event, ...]
    diagnostics: Tuple[Diagnostic, ...]
    notified: Tuple[str, ...]
    resolved: Tuple[str, ...]
    transitions: Tuple[AlertTransition, ...]


class ReliabilityEngine:
    """Stateful rule pipeline with restart-safe state and low-noise alerts."""

    _RULE_STATE_KEY = "reliability.rule_state"

    def __init__(
        self,
        store: StateStore,
        *,
        rules: Optional[RuleEngine] = None,
        sinks: Sequence[Sink] = (),
        audit_log: Optional[AuditLog] = None,
        notification_every: int = 10,
    ) -> None:
        if notification_every < 1:
            raise ValueError("notification_every must be at least 1")
        self.store = store
        self.rules = rules or RuleEngine()
        self.sinks = tuple(sinks)
        self.audit_log = audit_log
        self.notification_every = notification_every
        self._hydrated: set[str] = set()

    def evaluate(
        self,
        run_id: str,
        values: Mapping[str, object],
        *,
        now: float,
        source: str = "watch",
    ) -> ReliabilityResult:
        self._hydrate(run_id)
        events = tuple(
            self.rules.evaluate(
                RuleContext(run_id=run_id, values=dict(values), now=now, source=source)
            )
        )
        self.store.set_run_state(run_id, self._RULE_STATE_KEY, self.rules.snapshot_state(run_id))

        fingerprints: Dict[str, Event] = {event_fingerprint(event): event for event in events}
        notified: List[str] = []
        diagnostics: List[Diagnostic] = []
        transitions: List[AlertTransition] = []
        for fingerprint, event in fingerprints.items():
            occurrence, reopened = self.store.record_alert_transition(fingerprint, event)
            if occurrence == 1:
                transition = "reopened" if reopened else "opened"
                transitions.append(AlertTransition(fingerprint, transition))
                self._audit(
                    {
                        "schema_version": event.schema_version,
                        "type": f"alert_{transition}",
                        "run_id": event.run_id,
                        "fingerprint": fingerprint,
                    }
                )
            diagnostic = diagnose(event)
            diagnostics.append(diagnostic)
            if reopened or occurrence == 1 or occurrence % self.notification_every == 0:
                delivered = not self.sinks
                for sink in self.sinks:
                    try:
                        sink.emit(event)
                        delivered = True
                    except Exception as exc:
                        self._audit(
                            {
                                "schema_version": event.schema_version,
                                "type": "sink_error",
                                "fingerprint": fingerprint,
                                "sink": type(sink).__name__,
                                "error": type(exc).__name__,
                            }
                        )
                if delivered:
                    notified.append(fingerprint)
            self._audit(
                {
                    "schema_version": event.schema_version,
                    "type": "event",
                    "fingerprint": fingerprint,
                    "occurrence_count": occurrence,
                    "event": event.to_dict(),
                    "diagnostic": {
                        "summary": diagnostic.summary,
                        "evidence": list(diagnostic.evidence),
                        "probable_causes": list(diagnostic.probable_causes),
                        "recommendations": list(diagnostic.recommendations),
                    },
                }
            )

        resolved: List[str] = []
        for alert in self.store.active_alerts(run_id):
            fingerprint = str(alert["fingerprint"])
            if fingerprint in fingerprints:
                continue
            if self.store.resolve_alert(run_id, fingerprint):
                resolved.append(fingerprint)
                transitions.append(AlertTransition(fingerprint, "resolved"))
                self._audit(
                    {
                        "schema_version": "1.0",
                        "type": "alert_resolved",
                        "run_id": run_id,
                        "fingerprint": fingerprint,
                    }
                )
        return ReliabilityResult(
            events=events,
            diagnostics=tuple(diagnostics),
            notified=tuple(notified),
            resolved=tuple(resolved),
            transitions=tuple(transitions),
        )

    def _hydrate(self, run_id: str) -> None:
        if run_id in self._hydrated:
            return
        value = self.store.get_run_state(run_id, self._RULE_STATE_KEY, {})
        if isinstance(value, Mapping):
            self.rules.restore_state(run_id, value)
        self._hydrated.add(run_id)

    def _audit(self, record: Mapping[str, object]) -> None:
        if self.audit_log is not None:
            self.audit_log.append(record)


def event_fingerprint(event: Event) -> str:
    """Build a stable, privacy-safe alert identity."""
    entity_keys = (
        "entity",
        "metric",
        "device",
        "gpu_index",
        "index",
        "rank",
        "worker",
        "checkpoint",
        "mount",
        "path",
    )
    dimensions = {key: event.attributes[key] for key in entity_keys if key in event.attributes}
    identity = {
        "run_id": event.run_id,
        "source": event.source,
        "kind": event.kind.value,
        "message": event.message,
        "entity": dimensions,
    }
    raw = json.dumps(
        json_safe(identity),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:24]


def diagnose(event: Event) -> Diagnostic:
    """Return deterministic evidence and advice for a reliability event."""
    evidence = tuple(f"{key}={value}" for key, value in sorted(event.attributes.items()))
    guidance = {
        "nan_inf": (
            ("invalid data, unstable precision, or exploding gradients",),
            ("inspect the first bad step; validate inputs; restore a healthy checkpoint",),
        ),
        "loss_spike": (
            ("learning-rate instability or an anomalous batch",),
            ("compare recent batches and gradient norms before resuming",),
        ),
        "grad_spike": (
            ("gradient explosion or numerical overflow",),
            ("validate clipping and mixed-precision scaler state",),
        ),
        "step_stalled": (
            ("data loader starvation, checkpoint I/O, or distributed deadlock",),
            ("inspect process stacks, disk I/O, and rank logs",),
        ),
        "throughput_drop": (
            ("input pipeline contention, throttling, or checkpoint I/O",),
            ("compare CPU, disk, GPU utilization, and recent checkpoint activity",),
        ),
        "gpu_idle": (
            ("input starvation, wrong device placement, or a stalled process",),
            ("confirm step progress before treating low utilization as a failure",),
        ),
        "gpu_overheat": (
            ("cooling limitation or sustained power pressure",),
            ("reduce load safely and inspect cooling; do not hard-kill during a write",),
        ),
        "disk_low": (
            ("checkpoint or log growth exhausted available storage",),
            ("free space or change retention before the next checkpoint",),
        ),
        "process_dead": (
            ("training exited, crashed, or was externally terminated",),
            ("inspect exit status and validate a checkpoint before restart",),
        ),
        "cuda_oom": (
            ("model, batch, sequence length, or allocator fragmentation exceeded memory",),
            ("reduce peak memory demand and validate a checkpoint before resuming",),
        ),
        "nccl_error": (
            ("a rank exited, transport failed, or distributed peers diverged",),
            ("compare every rank log and validate network and process health",),
        ),
        "gpu_xid": (
            ("GPU hardware, driver, PCIe, or power instability was reported",),
            ("inspect system logs and GPU health before restarting training",),
        ),
        "checkpoint_stale": (
            ("checkpointing is disabled, blocked, or slower than expected",),
            ("confirm step progress and checkpoint cadence",),
        ),
        "checkpoint_corrupt": (
            ("partial write, missing shard, or storage corruption",),
            ("fall back to the newest validated checkpoint",),
        ),
    }
    causes, recommendations = guidance.get(
        event.kind.value,
        (("unknown deterministic rule trigger",), ("inspect the attached evidence",)),
    )
    return Diagnostic(
        event_id=event.event_id,
        summary=event.message,
        evidence=evidence,
        probable_causes=causes,
        recommendations=recommendations,
    )


__all__ = [
    "AlertTransition",
    "ReliabilityEngine",
    "ReliabilityResult",
    "diagnose",
    "event_fingerprint",
]
