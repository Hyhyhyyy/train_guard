"""Shared reliability runtime used by Python sessions and framework callbacks."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Mapping, Optional, Sequence

from .reliability import ReliabilityEngine, ReliabilityResult
from .rules import RuleConfig, RuleEngine
from .run.lifecycle import (
    append_lifecycle_event,
    lifecycle_path,
    make_lifecycle_event,
)
from .sinks import JsonlSink, Sink
from .state import AuditLog, StateStore


class ReliabilityRuntime:
    """Own one run's state, rule engine, sinks, and lifecycle."""

    def __init__(
        self,
        *,
        run_id: str,
        state_dir: Path,
        rule_config: Optional[RuleConfig] = None,
        sinks: Sequence[Sink] = (),
        notification_every: int = 10,
        source: str = "python-api",
    ) -> None:
        if not run_id.strip():
            raise ValueError("run_id is required")
        self.run_id = run_id
        self.source = source
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self._lifecycle_path = lifecycle_path(self.state_dir)
        self._store = StateStore(self.state_dir / "train_guard_state.sqlite")
        configured_sinks = (
            JsonlSink(self.state_dir / "reliability_events.jsonl"),
            *tuple(sinks),
        )
        self._engine = ReliabilityEngine(
            self._store,
            rules=RuleEngine(rule_config),
            sinks=configured_sinks,
            audit_log=AuditLog(self.state_dir / "reliability_audit.jsonl"),
            notification_every=notification_every,
        )
        self._started = False
        self._closed = False

    @property
    def closed(self) -> bool:
        return self._closed

    def observe(
        self,
        values: Mapping[str, object],
        *,
        timestamp: Optional[float] = None,
        source: Optional[str] = None,
    ) -> ReliabilityResult:
        self._require_open()
        self.ensure_started()
        observed_at = time.time() if timestamp is None else timestamp
        step = values.get("step")
        self.record_lifecycle(
            "train_heartbeat",
            step=step if isinstance(step, int) else None,
        )
        record_sample = getattr(self._store, "record_sample", None)
        if record_sample is not None:
            record_sample(self.run_id, observed_at, dict(values))
        return self._engine.evaluate(
            self.run_id,
            values,
            now=observed_at,
            source=source or self.source,
        )

    def ensure_started(self, *, step: Optional[int] = None) -> None:
        self._require_open()
        if not self._started:
            self.record_lifecycle("train_start", step=step)
            self._started = True

    def checkpoint(self, name: str, *, step: Optional[int] = None) -> None:
        self._require_open()
        self.ensure_started(step=step)
        self.record_lifecycle(
            "train_checkpoint",
            step=step,
            checkpoints=[Path(name).name],
        )

    def record_lifecycle(
        self,
        kind: str,
        *,
        step: Optional[int] = None,
        checkpoints: Sequence[str] = (),
    ) -> None:
        append_lifecycle_event(
            self._lifecycle_path,
            make_lifecycle_event(
                kind,
                framework=self.source,
                global_step=step,
                checkpoints=checkpoints,
                extra={"run_id": self.run_id},
            ),
        )
        phase = {
            "train_start": "running",
            "train_heartbeat": "running",
            "train_checkpoint": "checkpointed",
            "train_finish": "finished",
            "train_abort": "aborted",
        }.get(kind)
        if phase is not None:
            self._store.set_run_state(self.run_id, "lifecycle.phase", phase)
        if kind == "train_checkpoint":
            record_checkpoint = getattr(self._store, "record_checkpoint", None)
            if record_checkpoint is not None:
                for checkpoint in checkpoints:
                    record_checkpoint(
                        self.run_id,
                        checkpoint,
                        "observed",
                        {"step": step},
                    )

    def close(self, *, success: bool = True) -> None:
        if self._closed:
            return
        if self._started:
            self.record_lifecycle("train_finish" if success else "train_abort")
        self._store.close()
        self._closed = True

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("reliability runtime is closed")


__all__ = ["ReliabilityRuntime"]
