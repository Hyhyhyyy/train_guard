"""Stable public Python API for embedding Train Guard."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from .core.config import resolve_inline_config
from .data.commands import run_data_check
from .eval.metrics import run_eval
from .reliability import ReliabilityResult
from .rules import RuleConfig
from .run.commands import collect_watch_sample, run_run_check
from .runtime import ReliabilityRuntime
from .sinks import Sink


def check_dataset(config: Mapping[str, Any]) -> Mapping[str, Any]:
    """Run a read-only dataset integrity check."""
    public = dict(config)
    aliases = {
        "group_id_field": "group_id",
        "split_field": "split",
        "media_field": "media",
        "messages_field": "messages",
    }
    for old, new in aliases.items():
        if old in public and new not in public:
            public[new] = public.pop(old)
    resolved = resolve_inline_config(("data", "check"), public)
    resolved["group_id_field"] = resolved.pop("group_id")
    resolved["split_field"] = resolved.pop("split")
    resolved["media_field"] = resolved.pop("media")
    resolved["messages_field"] = resolved.pop("messages")
    return run_data_check(resolved)


def check_run(
    output_dir: Path,
    *,
    expected_steps: Optional[int] = None,
    framework: str = "huggingface",
    training_type: str = "auto",
) -> Mapping[str, Any]:
    """Validate training completion and artifacts."""
    return run_run_check(
        Path(output_dir),
        expected_steps=expected_steps,
        framework=framework,
        training_type=training_type,
    )


def evaluate_predictions(config: Mapping[str, Any]) -> Mapping[str, Any]:
    """Evaluate prediction/reference records and write configured reports."""
    public = dict(config)
    if "group_id_field" in public and "group_id" not in public:
        public["group_id"] = public.pop("group_id_field")
    resolved = resolve_inline_config(("eval",), public)
    resolved["group_id_field"] = resolved.pop("group_id")
    return run_eval(resolved)


def watch_snapshot(
    config: Mapping[str, Any], state: Optional[dict[str, Any]] = None
) -> Mapping[str, Any]:
    """Collect one sidecar observation without entering a watch loop."""
    public = dict(config)
    if "expected_gpu_count" in public and "expected_gpus" not in public:
        public["expected_gpus"] = public.pop("expected_gpu_count")
    resolved = resolve_inline_config(("run", "watch"), public)
    resolved["expected_gpu_count"] = resolved.pop("expected_gpus")
    return collect_watch_sample(resolved, state if state is not None else {})


class ReliabilitySession:
    """Context-managed reliability engine for custom training loops."""

    def __init__(
        self,
        *,
        run_id: str,
        state_dir: Path,
        rule_config: Optional[RuleConfig] = None,
        sinks: Sequence[Sink] = (),
        notification_every: int = 10,
    ) -> None:
        self.run_id = run_id
        self._runtime = ReliabilityRuntime(
            run_id=run_id,
            state_dir=state_dir,
            rule_config=rule_config,
            sinks=sinks,
            notification_every=notification_every,
            source="custom",
        )

    def observe(
        self,
        values: Mapping[str, object],
        *,
        timestamp: Optional[float] = None,
        source: str = "python-api",
    ) -> ReliabilityResult:
        return self._runtime.observe(
            values,
            timestamp=timestamp,
            source=source,
        )

    def checkpoint(self, name: str, *, step: Optional[int] = None) -> None:
        """Record a privacy-safe checkpoint lifecycle event."""
        self._runtime.checkpoint(name, step=step)

    def close(self, *, success: bool = True) -> None:
        self._runtime.close(success=success)

    def __enter__(self) -> "ReliabilitySession":
        self._runtime.ensure_started()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close(success=exc_type is None)


__all__ = [
    "ReliabilitySession",
    "check_dataset",
    "check_run",
    "evaluate_predictions",
    "watch_snapshot",
]
