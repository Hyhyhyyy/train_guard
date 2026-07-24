"""Training watch and reliability sampling."""

from __future__ import annotations

import json
import logging
import signal
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, MutableMapping, Optional, Tuple

from ..adapters.huggingface import (
    get_framework_adapter,
    is_bad_loss,
    parse_training_metrics,
)
from ..core.exitcodes import EXIT_FAIL, EXIT_OK, EXIT_RUNTIME, EXIT_WARN
from ..core.io_util import (
    append_jsonl,
    get_cpu_load,
    get_disk_usage,
    get_memory_info,
    pid_alive,
    run_command,
    utc_now_iso,
)
from ..core.privacy import redact_value
from ..domain import Event, json_safe
from ..reliability import ReliabilityEngine
from ..rules import RuleConfig, RuleEngine
from ..sinks import (
    JsonlSink,
    OtelJsonSink,
    PrometheusFileSink,
    WebhookSink,
)
from ..state import AuditLog, StateStore
from .lifecycle import (
    append_lifecycle_event,
    checkpoint_delta,
    lifecycle_path,
    make_lifecycle_event,
)

LOGGER = logging.getLogger("train_guard.run")
_SHUTDOWN = False


def _handle_signal(signum: int, _frame: Any) -> None:
    global _SHUTDOWN
    _SHUTDOWN = True
    LOGGER.warning("Signal %s received; shutting down gracefully", signum)


def query_nvidia_smi() -> Dict[str, Any]:
    """Query GPUs via nvidia-smi (no pynvml)."""
    query = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu,power.draw,driver_version",
        "--format=csv,noheader,nounits",
    ]
    code, out, err = run_command(query, timeout=20.0)
    if code != 0:
        return {
            "ok": False,
            "available": False,
            "error": err.strip() or out.strip() or f"nvidia-smi exit {code}",
            "gpus": [],
            "driver_version": None,
            "count": 0,
        }
    gpus: List[Dict[str, Any]] = []
    driver_version = None
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 9:
            continue
        try:
            gpu = {
                "index": int(parts[0]),
                "name": parts[1],
                "memory_total_mb": float(parts[2]),
                "memory_used_mb": float(parts[3]),
                "memory_free_mb": float(parts[4]),
                "utilization_gpu": float(parts[5]),
                "temperature_c": float(parts[6]) if parts[6] else None,
                "power_draw_w": float(parts[7]) if parts[7] not in {"", "[N/A]", "N/A"} else None,
                "driver_version": parts[8],
            }
            driver_version = parts[8]
            gpus.append(gpu)
        except ValueError:
            continue
    return {
        "ok": True,
        "available": True,
        "gpus": gpus,
        "count": len(gpus),
        "driver_version": driver_version,
        "error": None,
    }


def collect_watch_sample(cfg: Mapping[str, Any], state: MutableMapping[str, Any]) -> Dict[str, Any]:
    """One read-only monitoring sample."""
    alerts: List[Dict[str, str]] = []
    smi = query_nvidia_smi()
    cpu = get_cpu_load()
    mem = get_memory_info()
    disk_root = get_disk_usage(Path("/"))
    disk_cwd = get_disk_usage(Path.cwd())

    expected = cfg.get("expected_gpu_count")
    if smi.get("ok") and expected is not None and smi.get("count") != expected:
        alerts.append(
            {
                "level": "WARN",
                "code": "gpu_count_mismatch",
                "message": f"GPU count {smi.get('count')} != expected {expected}",
            }
        )

    idle_counts: Dict[int, int] = state.setdefault("idle_counts", {})
    idle_threshold = float(cfg.get("idle_gpu_util_threshold", 5.0))
    idle_limit = int(cfg.get("idle_gpu_consecutive", 3))
    if smi.get("ok"):
        for gpu in smi.get("gpus", []):
            idx = int(gpu["index"])
            util = float(gpu.get("utilization_gpu") or 0)
            idle_counts[idx] = idle_counts.get(idx, 0) + 1 if util < idle_threshold else 0
            if idle_counts[idx] >= idle_limit:
                alerts.append(
                    {
                        "level": "WARN",
                        "code": "gpu_idle",
                        "message": f"GPU {idx} idle {idle_counts[idx]} times (will not kill training)",
                    }
                )

    threshold = float(cfg.get("disk_free_gb_threshold", 10.0))
    for disk in (disk_root, disk_cwd):
        if disk.get("ok") and disk.get("free_gb", 0) < threshold:
            alerts.append(
                {
                    "level": "WARN",
                    "code": "disk_low",
                    "message": f"Low disk free_gb={disk.get('free_gb')} path_token",
                }
            )

    pid = cfg.get("pid")
    pid_status: Dict[str, Any] = {"configured": pid is not None}
    if pid is not None:
        alive = pid_alive(int(pid))
        pid_status.update({"pid": int(pid), "alive": alive})
        if not alive:
            alerts.append({"level": "ERROR", "code": "pid_dead", "message": f"PID {pid} not found"})

    metrics: Dict[str, float] = {}
    runtime_signatures: Dict[str, str] = {}
    log_info: Dict[str, Any] = {"configured": bool(cfg.get("log_file"))}
    log_file = cfg.get("log_file")
    if log_file:
        lp = Path(log_file)
        log_info["path_name"] = lp.name
        if lp.exists():
            mtime = lp.stat().st_mtime
            age_min = (time.time() - mtime) / 60.0
            log_info.update({"exists": True, "age_minutes": round(age_min, 2)})
            stale_min = float(cfg.get("stale_log_minutes", 15))
            if age_min > stale_min:
                alerts.append(
                    {
                        "level": "WARN",
                        "code": "stale_log",
                        "message": f"Log stale {age_min:.1f} > {stale_min} min",
                    }
                )
            offset = int(state.get("log_offset", 0))
            size = lp.stat().st_size
            if offset > size:
                offset = 0
            with lp.open("r", encoding="utf-8", errors="replace") as fh:
                fh.seek(offset)
                data = fh.read()
                state["log_offset"] = fh.tell()
            last_fp = state.get("last_metrics_fp")
            for line in data.splitlines():
                lowered = line.lower()
                if "cuda out of memory" in lowered or "cuda error: out of memory" in lowered:
                    runtime_signatures["cuda_oom"] = line
                if "nccl" in lowered and any(
                    marker in lowered
                    for marker in (
                        "error",
                        "failed",
                        "unhandled system error",
                        "remote process exited",
                    )
                ):
                    runtime_signatures["nccl_error"] = line
                if "nvrm: xid" in lowered or "xid error" in lowered:
                    runtime_signatures["gpu_xid"] = line
                parsed = parse_training_metrics(line)
                if not parsed:
                    continue
                fp = json.dumps(parsed, sort_keys=True)
                if fp == last_fp:
                    continue
                metrics = parsed
                last_fp = fp
                if "loss" in parsed and is_bad_loss(parsed["loss"]):
                    alerts.append(
                        {
                            "level": "ERROR",
                            "code": "loss_nan_inf",
                            "message": "Non-finite loss detected",
                        }
                    )
            state["last_metrics_fp"] = last_fp
        else:
            log_info["exists"] = False

    framework = get_framework_adapter(str(cfg.get("framework") or "generic"))
    output_dir = Path(cfg.get("output_dir") or "reports/watch")
    state_path = framework.locate_trainer_state(output_dir)
    trainer_info: Dict[str, Any] = {"ok": False}
    if state_path and state_path.exists():
        try:
            data = json.loads(state_path.read_text(encoding="utf-8"))
            history = data.get("log_history") or []
            latest = history[-1] if isinstance(history, list) and history else {}
            trainer_info = {
                "ok": True,
                "global_step": data.get("global_step"),
                "latest_keys": sorted(latest.keys()) if isinstance(latest, dict) else [],
            }
            if isinstance(latest, dict):
                for key in ("loss", "eval_loss", "learning_rate", "grad_norm", "epoch"):
                    if key in latest and key not in metrics:
                        try:
                            metrics[key] = float(latest[key])
                        except (TypeError, ValueError):
                            pass
        except (OSError, json.JSONDecodeError) as exc:
            trainer_info = {"ok": False, "error": type(exc).__name__}

    checkpoints = [p.name for p in framework.list_checkpoints(output_dir)]
    sample = {
        "timestamp": utc_now_iso(),
        "gpus": smi,
        "cpu": cpu,
        "memory": mem,
        "disk": {"root": disk_root, "cwd": disk_cwd},
        "pid": pid_status,
        "log": log_info,
        "metrics": metrics,
        "runtime_signatures": runtime_signatures,
        "trainer_state": trainer_info,
        "checkpoints": checkpoints,
        "alerts": alerts,
        "note": "Read-only watch; will not terminate training",
    }
    return redact_value(sample)


@contextmanager
def _temporary_watch_signal_handlers() -> Iterator[None]:
    """Install watch handlers and restore the caller's handlers in all cases."""
    previous = {
        signal.SIGINT: signal.getsignal(signal.SIGINT),
        signal.SIGTERM: signal.getsignal(signal.SIGTERM),
    }
    try:
        signal.signal(signal.SIGINT, _handle_signal)
        signal.signal(signal.SIGTERM, _handle_signal)
        yield
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def cmd_run_watch(cfg: Dict[str, Any]) -> int:
    """Periodic or once watch loop with scoped signal handlers."""
    global _SHUTDOWN
    _SHUTDOWN = False
    with _temporary_watch_signal_handlers():
        return _cmd_run_watch_loop(cfg)


def _cmd_run_watch_loop(cfg: Dict[str, Any]) -> int:
    """Implementation of the watcher loop."""
    out_dir = Path(cfg.get("output_dir") or "reports/watch")
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "watch.jsonl"
    life_path = lifecycle_path(out_dir)
    framework = str(cfg.get("framework") or "generic")
    explicit_run_id = cfg.get("run_id")
    run_id = str(explicit_run_id or f"run-{uuid.uuid4().hex}")
    reliability_store: Optional[StateStore] = None
    reliability: Optional[ReliabilityEngine] = None
    if bool(cfg.get("reliability", True)):
        state_db = Path(cfg.get("state_db") or (out_dir / "train_guard_state.sqlite"))
        reliability_store = StateStore(state_db)
        sinks: List[Any] = [JsonlSink(out_dir / "reliability_events.jsonl")]
        webhook_url = cfg.get("webhook_url")
        if webhook_url:
            sinks.append(WebhookSink(str(webhook_url)))
        prometheus_file = cfg.get("prometheus_file")
        if prometheus_file:
            sinks.append(PrometheusFileSink(Path(str(prometheus_file))))
        otel_file = cfg.get("otel_file")
        if otel_file:
            sinks.append(OtelJsonSink(Path(str(otel_file))))
        reliability = ReliabilityEngine(
            reliability_store,
            rules=RuleEngine(
                RuleConfig(
                    stall_seconds=float(cfg.get("step_stall_seconds", 300.0)),
                    gpu_overheat_celsius=float(cfg.get("gpu_overheat_celsius", 90.0)),
                    checkpoint_stale_seconds=float(cfg.get("checkpoint_stale_seconds", 1800.0)),
                )
            ),
            sinks=sinks,
            audit_log=AuditLog(out_dir / "reliability_audit.jsonl"),
            notification_every=int(cfg.get("notification_every", 10)),
        )
        if explicit_run_id:
            persisted_step = reliability_store.get_run_state(run_id, "watch.last_step")
            current_step = _trainer_global_step(out_dir, framework)
            if (
                isinstance(persisted_step, int)
                and current_step is not None
                and current_step < persisted_step
            ):
                LOGGER.info(
                    "Training step moved backwards (%s -> %s); resetting persisted lifecycle state",
                    persisted_step,
                    current_step,
                )
                reliability_store.reset_run(run_id)
    state: Dict[str, Any] = {
        "idle_counts": {},
        "log_offset": 0,
        "last_metrics_fp": None,
        "lifecycle_started": False,
        "seen_checkpoints": [],
    }
    if reliability_store is not None:
        state["log_offset"] = reliability_store.get_offset(run_id, "training_log")
        persisted_idle = reliability_store.get_run_state(run_id, "watch.idle_counts", {})
        if isinstance(persisted_idle, dict):
            state["idle_counts"] = {int(key): int(value) for key, value in persisted_idle.items()}
        persisted_checkpoints = reliability_store.get_run_state(
            run_id, "watch.seen_checkpoints", []
        )
        if isinstance(persisted_checkpoints, list):
            state["seen_checkpoints"] = [str(value) for value in persisted_checkpoints]
    once = bool(cfg.get("once"))
    interval = max(1, int(cfg.get("interval") or 30))
    exit_code = EXIT_OK
    last_step: Optional[int] = None
    last_checkpoints: List[str] = []
    saw_error_alert = False
    try:
        while True:
            sample = collect_watch_sample(cfg, state)
            sample["run_id"] = run_id
            sample["schema_version"] = 1
            metrics = sample.get("metrics") or {}
            trainer = sample.get("trainer_state") or {}
            checkpoints = list(sample.get("checkpoints") or [])
            last_checkpoints = checkpoints
            step_raw = trainer.get("global_step")
            if step_raw is None:
                step_raw = metrics.get("step")
            try:
                last_step = int(step_raw) if step_raw is not None else last_step
            except (TypeError, ValueError):
                pass
            reliability_events: Tuple[Event, ...] = ()
            if reliability is not None:
                result = reliability.evaluate(
                    run_id,
                    _reliability_values(sample, last_step, out_dir, framework),
                    now=time.time(),
                    source=framework,
                )
                reliability_events = result.events
                sample["reliability"] = {
                    "events": [event.to_dict() for event in result.events],
                    "diagnostics": [
                        {
                            "event_id": item.event_id,
                            "summary": item.summary,
                            "evidence": list(item.evidence),
                            "probable_causes": list(item.probable_causes),
                            "recommendations": list(item.recommendations),
                        }
                        for item in result.diagnostics
                    ],
                    "resolved": list(result.resolved),
                }
                reliability_store = reliability.store
                reliability_store.set_offset(
                    run_id, "training_log", int(state.get("log_offset", 0))
                )
                reliability_store.set_run_state(
                    run_id, "watch.idle_counts", state.get("idle_counts", {})
                )
                reliability_store.set_run_state(run_id, "watch.seen_checkpoints", checkpoints)
                if last_step is not None:
                    reliability_store.set_run_state(run_id, "watch.last_step", last_step)
            sample = dict(json_safe(redact_value(sample)))
            if reliability_store is not None:
                reliability_store.record_sample(run_id, time.time(), sample)
                reliability_store.set_run_state(run_id, "lifecycle.phase", "watching")
            append_jsonl(jsonl_path, redact_value(sample))
            alert_codes = [
                str(a.get("code"))
                for a in (sample.get("alerts") or [])
                if isinstance(a, dict) and a.get("code")
            ]
            alert_codes.extend(event.kind.value for event in reliability_events)
            if any(
                (a.get("level") == "ERROR")
                for a in (sample.get("alerts") or [])
                if isinstance(a, dict)
            ):
                saw_error_alert = True
            if any(event.severity.value in {"error", "critical"} for event in reliability_events):
                saw_error_alert = True

            if not state.get("lifecycle_started"):
                append_lifecycle_event(
                    life_path,
                    make_lifecycle_event(
                        "watch_start",
                        framework=framework,
                        global_step=last_step,
                        checkpoints=checkpoints,
                        message="watcher started",
                    ),
                )
                state["lifecycle_started"] = True
                state["seen_checkpoints"] = list(checkpoints)

            new_ckpts = checkpoint_delta(state.get("seen_checkpoints") or [], checkpoints)
            if new_ckpts:
                if reliability_store is not None:
                    for checkpoint in new_ckpts:
                        reliability_store.record_checkpoint(
                            run_id,
                            checkpoint,
                            "observed",
                            {"step": last_step},
                        )
                append_lifecycle_event(
                    life_path,
                    make_lifecycle_event(
                        "watch_checkpoint",
                        framework=framework,
                        global_step=last_step,
                        checkpoints=checkpoints,
                        alert_codes=alert_codes,
                        message=f"new checkpoints: {', '.join(new_ckpts)}",
                        extra={"new_checkpoints": new_ckpts},
                    ),
                )
                state["seen_checkpoints"] = list(checkpoints)

            append_lifecycle_event(
                life_path,
                make_lifecycle_event(
                    "watch_heartbeat",
                    framework=framework,
                    global_step=last_step,
                    checkpoints=checkpoints,
                    alert_codes=alert_codes,
                    extra={
                        "gpu_count": (sample.get("gpus") or {}).get("count"),
                        "pid_alive": (sample.get("pid") or {}).get("alive"),
                    },
                ),
            )

            print(
                f"[{sample['timestamp']}] GPUs={(sample.get('gpus') or {}).get('count')} "
                f"loss={metrics.get('loss')} eval_loss={metrics.get('eval_loss')} "
                f"alerts={len(sample.get('alerts') or [])}"
            )
            for alert in sample.get("alerts") or []:
                print(f"  !! [{alert.get('level')}] {alert.get('message')}")
                if alert.get("level") == "ERROR":
                    exit_code = max(exit_code, EXIT_FAIL)
                elif alert.get("level") == "WARN":
                    exit_code = max(exit_code, EXIT_WARN)
            for event in reliability_events:
                if event.severity.value in {"error", "critical"}:
                    exit_code = max(exit_code, EXIT_FAIL)
                elif event.severity.value == "warning":
                    exit_code = max(exit_code, EXIT_WARN)
            if once or _SHUTDOWN:
                break
            for _ in range(interval):
                if _SHUTDOWN:
                    break
                time.sleep(1)
            if _SHUTDOWN:
                break
    except Exception as exc:  # noqa: BLE001
        LOGGER.error("Watch failed: %s", exc)
        try:
            append_lifecycle_event(
                life_path,
                make_lifecycle_event(
                    "watch_error",
                    framework=framework,
                    global_step=last_step,
                    checkpoints=last_checkpoints,
                    message=f"watcher failed: {type(exc).__name__}",
                ),
            )
        except OSError:
            pass
        if reliability_store is not None:
            reliability_store.set_run_state(run_id, "lifecycle.phase", "watch_error")
            reliability_store.close()
        return EXIT_RUNTIME

    try:
        append_lifecycle_event(
            life_path,
            make_lifecycle_event(
                "watch_stop",
                framework=framework,
                global_step=last_step,
                checkpoints=last_checkpoints,
                message="watcher stopped",
                extra={
                    "reason": "signal" if _SHUTDOWN else ("once" if once else "loop_ended"),
                    "had_error_alert": saw_error_alert,
                },
            ),
        )
    except OSError:
        pass
    print(f"Watch data written: {jsonl_path}")
    print(f"Lifecycle written: {life_path}")
    if reliability_store is not None:
        reliability_store.set_run_state(run_id, "lifecycle.phase", "watch_stopped")
        reliability_store.close()
    return exit_code


def _trainer_global_step(output_dir: Path, framework: str) -> Optional[int]:
    """Read the current trainer step without mutating watcher state."""
    state_path = get_framework_adapter(framework).locate_trainer_state(output_dir)
    if state_path is None:
        return None
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
        value = state.get("global_step") if isinstance(state, dict) else None
        return int(value) if value is not None else None
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return None


def _reliability_values(
    sample: Mapping[str, Any],
    step: Optional[int],
    output_dir: Path,
    framework: str,
) -> Dict[str, Any]:
    """Flatten a watch sample into deterministic rule inputs."""
    values: Dict[str, Any] = dict(sample.get("metrics") or {})
    values.update(sample.get("runtime_signatures") or {})
    if step is not None:
        values["step"] = step
    gpus = (sample.get("gpus") or {}).get("gpus") or []
    if gpus:
        utilization = [
            float(gpu["utilization_gpu"]) for gpu in gpus if gpu.get("utilization_gpu") is not None
        ]
        temperatures = [
            float(gpu["temperature_c"]) for gpu in gpus if gpu.get("temperature_c") is not None
        ]
        if utilization:
            values["gpu_util_percent"] = sum(utilization) / len(utilization)
        if temperatures:
            values["gpu_temperature_c"] = max(temperatures)
    free_values = [
        int(disk["free_bytes"])
        for disk in (sample.get("disk") or {}).values()
        if isinstance(disk, dict) and disk.get("ok") and disk.get("free_bytes") is not None
    ]
    if free_values:
        values["disk_free_bytes"] = min(free_values)
    pid_info = sample.get("pid") or {}
    if pid_info.get("configured"):
        values["process_alive"] = pid_info.get("alive")
    adapter = get_framework_adapter(framework)
    checkpoints = adapter.list_checkpoints(output_dir)
    if checkpoints:
        try:
            newest_mtime = max(path.stat().st_mtime for path in checkpoints)
            values["checkpoint_age_seconds"] = max(0.0, time.time() - newest_mtime)
        except OSError:
            values["checkpoint_valid"] = False
    return values
