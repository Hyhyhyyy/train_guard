"""End-to-end training launch orchestration."""

from __future__ import annotations

import hashlib
import threading
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from ..core.io_util import utc_now_iso, write_json
from ..core.privacy import redact_value
from ..domain import json_safe
from ..env.doctor import run_doctor
from ..state import AuditLog, StateStore
from ..supervisor import (
    FileCheckpointValidator,
    FileHeartbeatProbe,
    ProcessSpec,
    RecoveryGuard,
    RecoveryPolicy,
    SupervisionResult,
    supervise,
)
from .check import run_run_check
from .lifecycle import append_lifecycle_event, lifecycle_path, make_lifecycle_event
from .manifest import run_manifest
from .watch import _cmd_run_watch_loop


class _MonitorController:
    def __init__(self, config: Mapping[str, Any]) -> None:
        self._config = dict(config)
        self._stop_event: Optional[threading.Event] = None
        self._thread: Optional[threading.Thread] = None
        self.exit_codes: list[int] = []

    def start(self, pid: int) -> None:
        self.stop()
        stop_event = threading.Event()
        config = {**self._config, "pid": pid, "_stop_event": stop_event}

        def run() -> None:
            self.exit_codes.append(_cmd_run_watch_loop(config))

        self._stop_event = stop_event
        self._thread = threading.Thread(
            target=run,
            name=f"train-guard-watch-{pid}",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        if self._stop_event is not None:
            self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=max(15.0, float(self._config.get("interval", 5)) + 10.0))
            if self._thread.is_alive():
                raise RuntimeError("training monitor did not stop within the safety timeout")
        self._stop_event = None
        self._thread = None


def _training_command(config: Mapping[str, Any]) -> list[str]:
    command = [str(value) for value in (config.get("training_command") or [])]
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise ValueError("a training command is required after '--'")
    return command


def _validate_recovery(config: Mapping[str, Any]) -> None:
    max_restarts = int(config.get("max_restarts") or 0)
    window = float(config.get("restart_window_seconds") or 3600.0)
    if max_restarts < 0 or window <= 0:
        raise ValueError("restart limits must be non-negative and window positive")
    health_defaults = {
        "health_max_age": 30.0,
        "health_timeout": 120.0,
        "health_interval": 2.0,
    }
    for key, default in health_defaults.items():
        if float(config.get(key) or default) <= 0:
            raise ValueError("health probe intervals must be positive")
    if bool(config.get("restart")) and (
        max_restarts < 1
        or config.get("checkpoint_dir") is None
        or not config.get("required_checkpoint_file")
    ):
        raise ValueError(
            "--restart requires --max-restarts >= 1, --checkpoint-dir, "
            "and at least one --required-checkpoint-file"
        )


def _overall_status(
    doctor: Mapping[str, Any],
    execution: Mapping[str, Any],
    postcheck: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    strict_preflight: bool,
) -> str:
    if execution.get("status") == "failed":
        return "FAIL"
    if strict_preflight and doctor.get("overall_status") == "FAIL":
        return "FAIL"
    if postcheck.get("overall_status") == "FAIL" or manifest.get("overall_status") == "FAIL":
        return "FAIL"
    if (
        doctor.get("overall_status") != "PASS"
        or postcheck.get("overall_status") != "PASS"
        or manifest.get("overall_status") != "PASS"
    ):
        return "WARN"
    return "PASS"


def run_launch(config: Mapping[str, Any]) -> dict[str, Any]:
    """Run preflight, supervised training, monitoring, acceptance, and manifest generation."""
    command = _training_command(config)
    _validate_recovery(config)
    output_dir = Path(str(config.get("output_dir") or "reports/run"))
    output_dir.mkdir(parents=True, exist_ok=True)
    framework = str(config.get("framework") or "generic")
    started_at = utc_now_iso()
    command_identity = "\0".join(command)
    run_id = str(
        config.get("run_id")
        or f"launch-{hashlib.sha256(command_identity.encode('utf-8')).hexdigest()[:12]}"
    )
    state_db = Path(config.get("state_db") or (output_dir / "train_guard_state.sqlite"))
    audit_path = Path(config.get("audit_log") or (output_dir / "train_guard_supervisor.jsonl"))
    summary_path = Path(config.get("summary_out") or (output_dir / "train_guard_run_summary.json"))
    for path in (state_db, audit_path, summary_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    strict_preflight = bool(config.get("strict_preflight"))

    doctor = run_doctor(
        model_path=Path(config["model_path"]) if config.get("model_path") else None,
        expected_gpus=(
            int(config["expected_gpus"]) if config.get("expected_gpus") is not None else None
        ),
    )
    doctor_path = output_dir / "train_guard_doctor.json"
    write_json(doctor_path, doctor, overwrite=True)

    if strict_preflight and doctor.get("overall_status") == "FAIL":
        summary = {
            "tool": "train_guard",
            "command": "run launch",
            "run_id": run_id,
            "started_at": started_at,
            "finished_at": utc_now_iso(),
            "overall_status": "FAIL",
            "phase": "preflight_rejected",
            "preflight": doctor,
            "execution": {"status": "not_started"},
            "postcheck": {},
            "manifest": {},
            "reports": {"doctor": doctor_path.name, "summary": summary_path.name},
        }
        safe_summary = dict(json_safe(redact_value(summary)))
        write_json(summary_path, safe_summary, overwrite=True)
        return safe_summary

    life_path = lifecycle_path(output_dir)
    append_lifecycle_event(
        life_path,
        make_lifecycle_event(
            "train_start",
            framework=framework,
            message="training launched by run launch",
            extra={"run_id": run_id},
        ),
    )
    audit = AuditLog(audit_path)
    monitor = _MonitorController(
        {
            "once": False,
            "interval": max(1, int(config.get("monitor_interval") or 5)),
            "log_file": config.get("log_file"),
            "framework": framework,
            "output_dir": output_dir,
            "expected_gpu_count": config.get("expected_gpus"),
            "run_id": run_id,
            "state_db": state_db,
            "reliability": True,
            "notification_every": int(config.get("notification_every") or 10),
            "step_stall_seconds": float(config.get("step_stall_seconds") or 300.0),
            "gpu_overheat_celsius": float(config.get("gpu_overheat_celsius") or 90.0),
            "checkpoint_stale_seconds": float(config.get("checkpoint_stale_seconds") or 1800.0),
        }
    )
    result: Optional[SupervisionResult] = None
    execution_error: Optional[str] = None

    def record_audit(record: Mapping[str, object]) -> None:
        audit.append(record)
        if record.get("type") == "process_started":
            monitor.start(int(str(record["pid"])))
        elif record.get("type") == "process_exited":
            monitor.stop()

    with StateStore(state_db) as store:
        stored_times = store.get_run_state(run_id, "supervisor.restart_times", [])
        restart_times = (
            [float(value) for value in stored_times] if isinstance(stored_times, list) else []
        )
        guard = RecoveryGuard(
            RecoveryPolicy(
                max_restarts=int(config.get("max_restarts") or 0),
                window_seconds=float(config.get("restart_window_seconds") or 3600.0),
                probe_timeout_seconds=float(config.get("health_timeout") or 120.0),
                probe_interval_seconds=float(config.get("health_interval") or 2.0),
            ),
            restart_times=restart_times,
            on_change=lambda values: store.set_run_state(
                run_id, "supervisor.restart_times", list(values)
            ),
        )
        try:
            result = supervise(
                ProcessSpec(command[0], tuple(command[1:])),
                restart_enabled=bool(config.get("restart")),
                recovery_guard=guard,
                checkpoint_path=(
                    Path(config["checkpoint_dir"]) if config.get("checkpoint_dir") else None
                ),
                checkpoint_validator=(
                    FileCheckpointValidator(
                        tuple(str(value) for value in config.get("required_checkpoint_file") or ())
                    )
                    if config.get("required_checkpoint_file")
                    else None
                ),
                health_probe=(
                    FileHeartbeatProbe(
                        Path(config["health_file"]),
                        float(config.get("health_max_age") or 30.0),
                    )
                    if config.get("health_file")
                    else None
                ),
                audit=record_audit,
                run_id=run_id,
                control_store=store,
                control_enabled=bool(config.get("enable_control")),
            )
        except Exception as exc:  # noqa: BLE001
            execution_error = type(exc).__name__
        finally:
            monitor.stop()
        succeeded = result is not None and result.exit_code == 0
        store.set_run_state(run_id, "lifecycle.phase", "finished" if succeeded else "aborted")

    terminal_kind = (
        "train_finish" if result is not None and result.exit_code == 0 else "train_abort"
    )
    append_lifecycle_event(
        life_path,
        make_lifecycle_event(
            terminal_kind,
            framework=framework,
            message=(
                "training process completed"
                if terminal_kind == "train_finish"
                else "training process failed"
            ),
            extra={"run_id": run_id},
        ),
    )
    execution = (
        {
            "status": "completed" if result.exit_code == 0 else "failed",
            "exit_code": result.exit_code,
            "restart_count": result.restart_count,
            "stopped_reason": result.stopped_reason,
            "checkpoint_errors": list(result.checkpoint_errors),
            "monitor_exit_codes": monitor.exit_codes,
        }
        if result is not None
        else {
            "status": "failed",
            "error": execution_error or "UnknownError",
            "monitor_exit_codes": monitor.exit_codes,
        }
    )
    postcheck = run_run_check(
        output_dir,
        expected_steps=(
            int(config["expected_steps"]) if config.get("expected_steps") is not None else None
        ),
        framework=framework,
        training_type=str(config.get("training_type") or "auto"),
    )
    check_path = output_dir / "train_guard_run_check.json"
    write_json(check_path, postcheck, overwrite=True)
    manifest = run_manifest(
        {
            "output_dir": output_dir,
            "framework": framework,
            "manifest_out": output_dir / "train_guard_manifest.json",
            "expected_steps": config.get("expected_steps"),
            "seed": config.get("seed"),
        }
    )
    overall = _overall_status(
        doctor,
        execution,
        postcheck,
        manifest,
        strict_preflight=strict_preflight,
    )
    summary = {
        "tool": "train_guard",
        "command": "run launch",
        "run_id": run_id,
        "started_at": started_at,
        "finished_at": utc_now_iso(),
        "overall_status": overall,
        "phase": "finished" if execution.get("status") == "completed" else "aborted",
        "training_command": {
            "executable": Path(command[0]).name,
            "argument_count": max(0, len(command) - 1),
        },
        "preflight": doctor,
        "execution": execution,
        "postcheck": postcheck,
        "manifest": manifest,
        "reports": {
            "doctor": doctor_path.name,
            "run_check": check_path.name,
            "manifest": "train_guard_manifest.json",
            "summary": summary_path.name,
            "audit": audit_path.name,
            "state_db": state_db.name,
        },
    }
    safe_summary = dict(json_safe(redact_value(summary)))
    write_json(summary_path, safe_summary, overwrite=True)
    return safe_summary


__all__ = ["run_launch"]
