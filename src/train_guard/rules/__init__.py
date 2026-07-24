"""Deterministic reliability rules with explicit state and thresholds."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Protocol, Sequence, Tuple

from train_guard.domain import Event, EventKind, Severity


@dataclass(frozen=True)
class RuleConfig:
    loss_spike_ratio: float = 3.0
    grad_spike_ratio: float = 5.0
    throughput_drop_ratio: float = 0.5
    stall_seconds: float = 300.0
    gpu_idle_percent: float = 5.0
    gpu_overheat_celsius: float = 90.0
    disk_free_bytes: int = 5 * 1024**3
    checkpoint_stale_seconds: float = 1800.0


@dataclass(frozen=True)
class RuleContext:
    run_id: str
    values: Mapping[str, Any]
    now: float
    source: str = "rules"


@dataclass
class RuleState:
    previous_step: Optional[int] = None
    last_step_change_at: Optional[float] = None
    previous_loss: Optional[float] = None
    previous_grad_norm: Optional[float] = None
    previous_throughput: Optional[float] = None


class Rule(Protocol):
    name: str

    def evaluate(
        self, context: RuleContext, state: RuleState, config: RuleConfig
    ) -> Iterable[Event]: ...


def _number(values: Mapping[str, Any], key: str) -> Optional[float]:
    value = values.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _event(
    context: RuleContext, kind: EventKind, severity: Severity, message: str, **attributes: Any
) -> Event:
    step = context.values.get("step")
    return Event(
        run_id=context.run_id,
        kind=kind,
        severity=severity,
        message=message,
        source=context.source,
        step=int(step) if isinstance(step, int) else None,
        attributes=attributes,
    )


class FiniteRule:
    name = "finite"

    def evaluate(
        self, context: RuleContext, state: RuleState, config: RuleConfig
    ) -> Iterable[Event]:
        del state, config
        for key in ("loss", "grad_norm"):
            value = _number(context.values, key)
            if value is not None and not math.isfinite(value):
                yield _event(
                    context,
                    EventKind.NAN_INF,
                    Severity.CRITICAL,
                    f"{key} is not finite",
                    metric=key,
                    value=str(value),
                )


class SpikeRule:
    name = "spike"

    def evaluate(
        self, context: RuleContext, state: RuleState, config: RuleConfig
    ) -> Iterable[Event]:
        loss = _number(context.values, "loss")
        if loss is not None and state.previous_loss not in (None, 0.0) and math.isfinite(loss):
            ratio = abs(loss) / max(abs(state.previous_loss), 1e-12)
            if ratio >= config.loss_spike_ratio:
                yield _event(
                    context,
                    EventKind.LOSS_SPIKE,
                    Severity.ERROR,
                    "loss increased abruptly",
                    ratio=ratio,
                )
        grad = _number(context.values, "grad_norm")
        if grad is not None and state.previous_grad_norm not in (None, 0.0) and math.isfinite(grad):
            ratio = abs(grad) / max(abs(state.previous_grad_norm), 1e-12)
            if ratio >= config.grad_spike_ratio:
                yield _event(
                    context,
                    EventKind.GRAD_SPIKE,
                    Severity.ERROR,
                    "gradient norm increased abruptly",
                    ratio=ratio,
                )


class ProgressRule:
    name = "progress"

    def evaluate(
        self, context: RuleContext, state: RuleState, config: RuleConfig
    ) -> Iterable[Event]:
        step = context.values.get("step")
        if (
            isinstance(step, int)
            and step == state.previous_step
            and state.last_step_change_at is not None
        ):
            age = context.now - state.last_step_change_at
            if age >= config.stall_seconds:
                yield _event(
                    context,
                    EventKind.STEP_STALLED,
                    Severity.ERROR,
                    "training step has not advanced",
                    stalled_seconds=age,
                )
        throughput = _number(context.values, "throughput")
        if throughput is not None and state.previous_throughput not in (None, 0.0):
            ratio = throughput / max(state.previous_throughput, 1e-12)
            if ratio <= config.throughput_drop_ratio:
                yield _event(
                    context,
                    EventKind.THROUGHPUT_DROP,
                    Severity.WARNING,
                    "throughput dropped",
                    ratio=ratio,
                )


class ResourceRule:
    name = "resource"

    def evaluate(
        self, context: RuleContext, state: RuleState, config: RuleConfig
    ) -> Iterable[Event]:
        del state
        gpu_util = _number(context.values, "gpu_util_percent")
        if gpu_util is not None and gpu_util <= config.gpu_idle_percent:
            yield _event(
                context,
                EventKind.GPU_IDLE,
                Severity.WARNING,
                "GPU appears idle",
                gpu_util_percent=gpu_util,
            )
        temperature = _number(context.values, "gpu_temperature_c")
        if temperature is not None and temperature >= config.gpu_overheat_celsius:
            yield _event(
                context,
                EventKind.GPU_OVERHEAT,
                Severity.ERROR,
                "GPU temperature is too high",
                temperature_c=temperature,
            )
        free_bytes = _number(context.values, "disk_free_bytes")
        if free_bytes is not None and free_bytes <= config.disk_free_bytes:
            yield _event(
                context,
                EventKind.DISK_LOW,
                Severity.ERROR,
                "disk free space is low",
                free_bytes=int(free_bytes),
            )


class RuntimeSignatureRule:
    name = "runtime_signature"

    def evaluate(
        self, context: RuleContext, state: RuleState, config: RuleConfig
    ) -> Iterable[Event]:
        del state, config
        signatures = (
            ("cuda_oom", EventKind.CUDA_OOM, "CUDA out-of-memory signature detected"),
            ("nccl_error", EventKind.NCCL_ERROR, "NCCL communication error detected"),
            ("gpu_xid", EventKind.GPU_XID, "NVIDIA Xid hardware or driver error detected"),
        )
        for key, kind, message in signatures:
            evidence = context.values.get(key)
            if evidence:
                yield _event(
                    context,
                    kind,
                    Severity.CRITICAL,
                    message,
                    signature=str(evidence)[:500],
                )


class LivenessRule:
    name = "liveness"

    def evaluate(
        self, context: RuleContext, state: RuleState, config: RuleConfig
    ) -> Iterable[Event]:
        del state
        if context.values.get("process_alive") is False:
            yield _event(
                context, EventKind.PROCESS_DEAD, Severity.CRITICAL, "training process is not alive"
            )
        checkpoint_age = _number(context.values, "checkpoint_age_seconds")
        if checkpoint_age is not None and checkpoint_age >= config.checkpoint_stale_seconds:
            yield _event(
                context,
                EventKind.CHECKPOINT_STALE,
                Severity.WARNING,
                "checkpoint is stale",
                age_seconds=checkpoint_age,
            )
        if context.values.get("checkpoint_valid") is False:
            yield _event(
                context,
                EventKind.CHECKPOINT_CORRUPT,
                Severity.CRITICAL,
                "checkpoint validation failed",
            )


DEFAULT_RULES: Tuple[Rule, ...] = (
    FiniteRule(),
    SpikeRule(),
    ProgressRule(),
    ResourceRule(),
    RuntimeSignatureRule(),
    LivenessRule(),
)


class RuleEngine:
    """Stateful engine whose output depends only on ordered inputs and config."""

    def __init__(
        self, config: Optional[RuleConfig] = None, rules: Sequence[Rule] = DEFAULT_RULES
    ) -> None:
        self.config = config or RuleConfig()
        self.rules = tuple(rules)
        self._states: Dict[str, RuleState] = {}

    def evaluate(self, context: RuleContext) -> List[Event]:
        state = self._states.setdefault(context.run_id, RuleState())
        events: List[Event] = []
        for rule in self.rules:
            events.extend(rule.evaluate(context, state, self.config))
        self._advance(context, state)
        return events

    def restore_state(self, run_id: str, values: Mapping[str, Any]) -> None:
        """Restore a run's bounded rule state from persistent storage."""
        self._states[run_id] = RuleState(
            previous_step=_optional_int(values.get("previous_step")),
            last_step_change_at=_optional_float(values.get("last_step_change_at")),
            previous_loss=_optional_float(values.get("previous_loss")),
            previous_grad_norm=_optional_float(values.get("previous_grad_norm")),
            previous_throughput=_optional_float(values.get("previous_throughput")),
        )

    def snapshot_state(self, run_id: str) -> Dict[str, Any]:
        """Return the JSON-serializable state required across watcher restarts."""
        state = self._states.get(run_id, RuleState())
        return {
            "previous_step": state.previous_step,
            "last_step_change_at": state.last_step_change_at,
            "previous_loss": state.previous_loss,
            "previous_grad_norm": state.previous_grad_norm,
            "previous_throughput": state.previous_throughput,
        }

    @staticmethod
    def _advance(context: RuleContext, state: RuleState) -> None:
        step = context.values.get("step")
        if isinstance(step, int):
            if step != state.previous_step:
                state.last_step_change_at = context.now
            state.previous_step = step
        for key, attr in (
            ("loss", "previous_loss"),
            ("grad_norm", "previous_grad_norm"),
            ("throughput", "previous_throughput"),
        ):
            value = _number(context.values, key)
            if value is not None and math.isfinite(value):
                setattr(state, attr, value)


def _optional_int(value: Any) -> Optional[int]:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_float(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


__all__ = ["DEFAULT_RULES", "Rule", "RuleConfig", "RuleContext", "RuleEngine", "RuleState"]
