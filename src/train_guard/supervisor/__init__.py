"""Controlled subprocess, checkpoint validation, and bounded recovery primitives."""

from __future__ import annotations

import hashlib
import os
import signal
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Callable, Mapping, Optional, Protocol, Sequence, Tuple

from train_guard.core.optional import try_import_psutil


@dataclass(frozen=True)
class ProcessSpec:
    executable: str
    arguments: Tuple[str, ...] = ()
    cwd: Optional[Path] = None
    environment: Mapping[str, str] = field(default_factory=dict)
    capture_output: bool = False

    @property
    def command(self) -> Tuple[str, ...]:
        return (self.executable, *self.arguments)


class ManagedProcess:
    """A subprocess launched without a shell and terminated with a grace period."""

    def __init__(self, spec: ProcessSpec) -> None:
        self.spec = spec
        self._process: Optional[subprocess.Popen[str]] = None
        self._stdout_file: Optional[IO[str]] = None
        self._stderr_file: Optional[IO[str]] = None
        self._stdout = ""
        self._stderr = ""

    @property
    def pid(self) -> Optional[int]:
        return None if self._process is None else self._process.pid

    @property
    def stdout(self) -> str:
        return self._stdout

    @property
    def stderr(self) -> str:
        return self._stderr

    def start(self) -> int:
        if self._process is not None and self._process.poll() is None:
            raise RuntimeError("process is already running")
        self._close_capture_files()
        self._stdout = ""
        self._stderr = ""
        environment = os.environ.copy()
        environment.update(self.spec.environment)
        if self.spec.capture_output:
            # Disk-backed files are continuously drained by the child OS write
            # path, unlike PIPE handles which can fill before wait() consumes.
            self._stdout_file = tempfile.TemporaryFile(
                mode="w+t", encoding="utf-8", errors="replace"
            )
            self._stderr_file = tempfile.TemporaryFile(
                mode="w+t", encoding="utf-8", errors="replace"
            )
        self._process = subprocess.Popen(
            self.spec.command,
            cwd=str(self.spec.cwd) if self.spec.cwd else None,
            env=environment,
            stdin=None,
            stdout=self._stdout_file,
            stderr=self._stderr_file,
            text=True,
            shell=False,
        )
        return self._process.pid

    def poll(self) -> Optional[int]:
        return None if self._process is None else self._process.poll()

    def wait(self) -> int:
        if self._process is None:
            raise RuntimeError("process was not started")
        try:
            return int(self._process.wait())
        except KeyboardInterrupt:
            self.terminate()
            raise
        finally:
            if self._process.poll() is not None:
                self._collect_output()

    def terminate(self, grace_seconds: float = 10.0) -> int:
        if self._process is None:
            raise RuntimeError("process was not started")
        if self._process.poll() is None:
            self._terminate_tree(force=False)
            try:
                self._process.wait(timeout=grace_seconds)
            except subprocess.TimeoutExpired:
                self._terminate_tree(force=True)
                self._process.wait()
        self._collect_output()
        return int(self._process.returncode)

    def pause(self) -> None:
        if self._process is None or self._process.poll() is not None:
            raise RuntimeError("process is not running")
        psutil = try_import_psutil()
        if psutil is not None:
            psutil.Process(self._process.pid).suspend()
            return
        if os.name != "nt":
            os.kill(self._process.pid, getattr(signal, "SIGSTOP"))
            return
        raise RuntimeError("pause requires the optional psutil dependency on Windows")

    def resume(self) -> None:
        if self._process is None or self._process.poll() is not None:
            raise RuntimeError("process is not running")
        psutil = try_import_psutil()
        if psutil is not None:
            psutil.Process(self._process.pid).resume()
            return
        if os.name != "nt":
            os.kill(self._process.pid, getattr(signal, "SIGCONT"))
            return
        raise RuntimeError("resume requires the optional psutil dependency on Windows")

    def _terminate_tree(self, *, force: bool) -> None:
        assert self._process is not None
        psutil = try_import_psutil()
        if psutil is None:
            (self._process.kill if force else self._process.terminate)()
            return
        try:
            parent = psutil.Process(self._process.pid)
            processes = [*parent.children(recursive=True), parent]
        except psutil.Error:
            if self._process.poll() is None:
                (self._process.kill if force else self._process.terminate)()
            return
        for process in reversed(processes):
            try:
                (process.kill if force else process.terminate)()
            except psutil.Error:
                continue

    def _collect_output(self) -> None:
        for handle, attribute in (
            (self._stdout_file, "_stdout"),
            (self._stderr_file, "_stderr"),
        ):
            if handle is None:
                continue
            handle.flush()
            handle.seek(0)
            setattr(self, attribute, handle.read())

    def _close_capture_files(self) -> None:
        for handle in (self._stdout_file, self._stderr_file):
            if handle is not None:
                handle.close()
        self._stdout_file = None
        self._stderr_file = None


@dataclass(frozen=True)
class CheckpointValidation:
    valid: bool
    errors: Tuple[str, ...] = ()
    digest: Optional[str] = None


class CheckpointValidator(Protocol):
    def validate(self, path: Path) -> CheckpointValidation: ...


class FileCheckpointValidator:
    """Validate required files, non-empty content, and an optional SHA-256 digest."""

    def __init__(
        self, required_files: Sequence[str], expected_digest: Optional[str] = None
    ) -> None:
        self.required_files = tuple(required_files)
        self.expected_digest = expected_digest

    def validate(self, path: Path) -> CheckpointValidation:
        errors: list[str] = []
        digest = hashlib.sha256()
        if not path.is_dir():
            return CheckpointValidation(False, ("checkpoint directory does not exist",))
        for relative in sorted(self.required_files):
            candidate = path / relative
            if not candidate.is_file():
                errors.append(f"missing required file: {relative}")
                continue
            if candidate.stat().st_size == 0:
                errors.append(f"empty required file: {relative}")
            digest.update(relative.encode("utf-8"))
            with candidate.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
        actual = digest.hexdigest() if not errors else None
        if (
            actual is not None
            and self.expected_digest is not None
            and actual != self.expected_digest
        ):
            errors.append("checkpoint digest mismatch")
        return CheckpointValidation(not errors, tuple(errors), actual)


class HealthProbe(Protocol):
    def healthy(self) -> bool: ...


class FileHeartbeatProbe:
    """Healthy when a local heartbeat/log file was updated recently."""

    def __init__(self, path: Path, max_age_seconds: float = 30.0) -> None:
        if max_age_seconds <= 0:
            raise ValueError("heartbeat max age must be positive")
        self.path = path
        self.max_age_seconds = max_age_seconds

    def healthy(self) -> bool:
        try:
            return time.time() - self.path.stat().st_mtime <= self.max_age_seconds
        except OSError:
            return False


@dataclass(frozen=True)
class RecoveryPolicy:
    max_restarts: int = 3
    window_seconds: float = 3600.0
    probe_timeout_seconds: float = 120.0
    probe_interval_seconds: float = 2.0


class RecoveryGuard:
    """Tracks restart timestamps and prevents an infinite restart loop."""

    def __init__(
        self,
        policy: Optional[RecoveryPolicy] = None,
        *,
        restart_times: Sequence[float] = (),
        on_change: Optional[Callable[[Sequence[float]], None]] = None,
    ) -> None:
        self.policy = policy or RecoveryPolicy()
        self._restarts = [float(timestamp) for timestamp in restart_times]
        self._on_change = on_change

    def permit_restart(self, now: float) -> bool:
        cutoff = now - self.policy.window_seconds
        self._restarts = [timestamp for timestamp in self._restarts if timestamp >= cutoff]
        if len(self._restarts) >= self.policy.max_restarts:
            return False
        self._restarts.append(now)
        if self._on_change is not None:
            self._on_change(tuple(self._restarts))
        return True

    def wait_until_healthy(self, probe: HealthProbe) -> bool:
        deadline = time.monotonic() + self.policy.probe_timeout_seconds
        while time.monotonic() < deadline:
            if probe.healthy():
                return True
            time.sleep(self.policy.probe_interval_seconds)
        return False


@dataclass(frozen=True)
class SupervisionResult:
    exit_code: int
    restart_count: int
    stopped_reason: str
    checkpoint_errors: Tuple[str, ...] = ()


AuditCallback = Callable[[Mapping[str, object]], None]


class ControlStore(Protocol):
    def register_managed_process(
        self,
        run_id: str,
        pid: int,
        status: str,
        capabilities: tuple[str, ...],
    ) -> None: ...

    def claim_control(self, run_id: str, now: float) -> Optional[Mapping[str, object]]: ...

    def complete_control(
        self,
        command_id: str,
        status: str,
        outcome: Mapping[str, object],
    ) -> None: ...

    def record_recovery(
        self,
        run_id: str,
        action: str,
        status: str,
        details: Optional[Mapping[str, object]] = None,
    ) -> None: ...


def _record_automatic_recovery(
    store: Optional[ControlStore],
    run_id: Optional[str],
    status: str,
    details: Mapping[str, object],
) -> None:
    if store is not None and run_id is not None:
        store.record_recovery(run_id, "automatic_restart", status, details)


def supervise(
    spec: ProcessSpec,
    *,
    restart_enabled: bool = False,
    recovery_guard: Optional[RecoveryGuard] = None,
    checkpoint_path: Optional[Path] = None,
    checkpoint_validator: Optional[CheckpointValidator] = None,
    health_probe: Optional[HealthProbe] = None,
    audit: Optional[AuditCallback] = None,
    run_id: Optional[str] = None,
    control_store: Optional[ControlStore] = None,
    control_enabled: bool = False,
) -> SupervisionResult:
    """Run a command and optionally restart only after checkpoint validation.

    The same explicit argv is reused; Train Guard never builds or executes a
    shell string. Training frameworks should be configured to resume from their
    latest checkpoint before restart is enabled.
    """
    guard = recovery_guard or RecoveryGuard()
    restarts = 0
    pending_automatic_restart: Optional[int] = None
    while True:
        process = ManagedProcess(spec)
        pid = process.start()
        capabilities = ["graceful_stop", "terminate"]
        if try_import_psutil() is not None or os.name != "nt":
            capabilities.extend(("pause", "resume"))
        if restart_enabled and checkpoint_path is not None and checkpoint_validator is not None:
            capabilities.append("validated_restart")
        if control_store is not None and run_id is not None:
            control_store.register_managed_process(
                run_id,
                pid,
                "running",
                tuple(capabilities) if control_enabled else (),
            )
        if audit:
            audit({"type": "process_started", "pid": pid, "restart": restarts})
        if health_probe is not None and restarts > 0:
            if not guard.wait_until_healthy(health_probe):
                exit_code = process.terminate()
                if pending_automatic_restart is not None:
                    _record_automatic_recovery(
                        control_store,
                        run_id,
                        "failed",
                        {"restart": pending_automatic_restart, "reason": "health_probe_failed"},
                    )
                    pending_automatic_restart = None
                if audit:
                    audit({"type": "recovery_probe_failed", "exit_code": exit_code})
                return SupervisionResult(exit_code, restarts, "health_probe_failed")
            if pending_automatic_restart is not None:
                _record_automatic_recovery(
                    control_store,
                    run_id,
                    "succeeded",
                    {"restart": pending_automatic_restart, "pid": pid},
                )
                pending_automatic_restart = None
        requested_restart = False
        requested_stop: Optional[str] = None
        if control_enabled and control_store is not None and run_id is not None:
            exit_code, requested_restart, requested_stop = _wait_with_control(
                process,
                run_id,
                control_store,
                checkpoint_path=checkpoint_path,
                checkpoint_validator=checkpoint_validator,
                audit=audit,
            )
        else:
            exit_code = process.wait()
        if control_store is not None and run_id is not None:
            control_store.register_managed_process(run_id, pid, "exited", ())
        if audit:
            audit({"type": "process_exited", "exit_code": exit_code, "restart": restarts})
        if pending_automatic_restart is not None and health_probe is None:
            _record_automatic_recovery(
                control_store,
                run_id,
                "succeeded" if exit_code == 0 else "failed",
                {
                    "restart": pending_automatic_restart,
                    "pid": pid,
                    "exit_code": exit_code,
                },
            )
            pending_automatic_restart = None
        if requested_stop is not None:
            return SupervisionResult(exit_code, restarts, f"control_{requested_stop}")
        if exit_code == 0 and not requested_restart:
            return SupervisionResult(0, restarts, "completed")
        if not restart_enabled and not requested_restart:
            return SupervisionResult(exit_code, restarts, "restart_disabled")
        if checkpoint_path is None or checkpoint_validator is None:
            if not requested_restart:
                _record_automatic_recovery(
                    control_store,
                    run_id,
                    "rejected",
                    {"reason": "checkpoint_validation_not_configured"},
                )
            return SupervisionResult(
                exit_code,
                restarts,
                "checkpoint_validation_not_configured",
                ("checkpoint path and validator are required",),
            )
        validation = checkpoint_validator.validate(checkpoint_path)
        if not validation.valid:
            if not requested_restart:
                _record_automatic_recovery(
                    control_store,
                    run_id,
                    "rejected",
                    {"reason": "checkpoint_invalid", "errors": list(validation.errors)},
                )
            if audit:
                audit(
                    {
                        "type": "checkpoint_rejected",
                        "errors": list(validation.errors),
                    }
                )
            return SupervisionResult(exit_code, restarts, "checkpoint_invalid", validation.errors)
        if not guard.permit_restart(time.time()):
            if not requested_restart:
                _record_automatic_recovery(
                    control_store,
                    run_id,
                    "rejected",
                    {"reason": "restart_budget_exhausted"},
                )
            return SupervisionResult(exit_code, restarts, "restart_budget_exhausted")
        restarts += 1
        if not requested_restart:
            _record_automatic_recovery(
                control_store,
                run_id,
                "attempted",
                {"restart": restarts, "checkpoint_digest": validation.digest},
            )
            pending_automatic_restart = restarts
        if audit:
            audit(
                {
                    "type": "restart_permitted",
                    "restart": restarts,
                    "checkpoint_digest": validation.digest,
                }
            )


def _wait_with_control(
    process: ManagedProcess,
    run_id: str,
    store: ControlStore,
    *,
    checkpoint_path: Optional[Path],
    checkpoint_validator: Optional[CheckpointValidator],
    audit: Optional[AuditCallback],
) -> tuple[int, bool, Optional[str]]:
    requested_restart = False
    requested_stop: Optional[str] = None
    while process.poll() is None:
        command = store.claim_control(run_id, time.time())
        if command is None:
            time.sleep(0.25)
            continue
        command_id = str(command["command_id"])
        action = str(command["action"])
        try:
            outcome: Mapping[str, object]
            if action == "pause":
                process.pause()
                outcome = {"process_status": "paused"}
            elif action == "resume":
                process.resume()
                outcome = {"process_status": "running"}
            elif action == "graceful_stop":
                requested_stop = action
                outcome = {"exit_code": process.terminate(10.0)}
            elif action == "terminate":
                requested_stop = action
                outcome = {"exit_code": process.terminate(0.0)}
            elif action == "validated_restart":
                if checkpoint_path is None or checkpoint_validator is None:
                    raise RuntimeError("checkpoint validation is not configured")
                validation = checkpoint_validator.validate(checkpoint_path)
                if not validation.valid:
                    raise RuntimeError("checkpoint validation failed")
                requested_restart = True
                outcome = {
                    "exit_code": process.terminate(10.0),
                    "checkpoint_digest": validation.digest,
                }
            else:
                raise RuntimeError("action is unavailable for this managed process")
            store.complete_control(command_id, "succeeded", outcome)
            store.record_recovery(run_id, action, "succeeded", outcome)
            if audit:
                audit({"type": "control_succeeded", "action": action, **outcome})
        except Exception as exc:
            outcome = {"error": type(exc).__name__}
            store.complete_control(command_id, "failed", outcome)
            store.record_recovery(run_id, action, "failed", outcome)
            if audit:
                audit({"type": "control_failed", "action": action, **outcome})
        if process.poll() is not None:
            break
    exit_code = process.wait()
    return exit_code, requested_restart, requested_stop


__all__ = [
    "CheckpointValidation",
    "CheckpointValidator",
    "FileCheckpointValidator",
    "FileHeartbeatProbe",
    "HealthProbe",
    "ManagedProcess",
    "ProcessSpec",
    "RecoveryGuard",
    "RecoveryPolicy",
    "SupervisionResult",
    "supervise",
]
