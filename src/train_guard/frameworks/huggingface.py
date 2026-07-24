"""Dependency-free Hugging Face Trainer callback integration."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from train_guard.reliability import ReliabilityResult
from train_guard.rules import RuleConfig
from train_guard.runtime import ReliabilityRuntime
from train_guard.sinks import Sink


class TrainGuardCallback:
    """Trainer-compatible callback that evaluates logs without importing transformers.

    Pass an instance in ``Trainer(callbacks=[...])``. The callback is observe-only;
    it never changes ``TrainerControl`` or training arguments.
    """

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
        self.state_dir = Path(state_dir)
        self._runtime = ReliabilityRuntime(
            run_id=run_id,
            state_dir=self.state_dir,
            rule_config=rule_config,
            sinks=sinks,
            notification_every=notification_every,
            source="huggingface",
        )
        self.last_result: Optional[ReliabilityResult] = None

    def on_train_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
        del args, kwargs
        step = getattr(state, "global_step", None)
        self._runtime.ensure_started(step=step if isinstance(step, int) else None)
        return control

    def on_log(
        self,
        args: Any,
        state: Any,
        control: Any,
        logs: Optional[Mapping[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        del args, kwargs
        values = dict(logs or {})
        step = getattr(state, "global_step", None)
        if isinstance(step, int):
            values["step"] = step
        self.last_result = self._runtime.observe(values)
        return control

    def on_save(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
        del args, kwargs
        step = getattr(state, "global_step", None)
        checkpoint = f"checkpoint-{step}" if isinstance(step, int) else "checkpoint"
        self._runtime.checkpoint(
            checkpoint,
            step=step if isinstance(step, int) else None,
        )
        return control

    def on_train_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
        del args, state, kwargs
        self.close(success=True)
        return control

    def close(self, *, success: bool = True) -> None:
        self._runtime.close(success=success)

    def __enter__(self) -> "TrainGuardCallback":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close(success=exc_type is None)


__all__ = ["TrainGuardCallback"]
