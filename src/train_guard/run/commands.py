"""run watch / check / compare / manifest."""

from __future__ import annotations

import hashlib
import json
import logging
import signal
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Optional

from .. import __version__
from ..adapters.huggingface import get_framework_adapter, is_bad_loss, is_finite_number, parse_training_metrics
from ..core.events import CheckItem, overall_status
from ..core.exitcodes import EXIT_FAIL, EXIT_OK, EXIT_RUNTIME, EXIT_WARN
from ..core.io_util import (
    append_jsonl,
    get_cpu_load,
    get_disk_usage,
    get_memory_info,
    pid_alive,
    run_command,
    utc_now_iso,
    write_json,
)
from ..core.privacy import redact_value
from ..report.html import render_html_report

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
    return {"ok": True, "available": True, "gpus": gpus, "count": len(gpus), "driver_version": driver_version, "error": None}


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
        alerts.append({"level": "WARN", "code": "gpu_count_mismatch", "message": f"GPU count {smi.get('count')} != expected {expected}"})

    idle_counts: Dict[int, int] = state.setdefault("idle_counts", {})
    idle_threshold = float(cfg.get("idle_gpu_util_threshold", 5.0))
    idle_limit = int(cfg.get("idle_gpu_consecutive", 3))
    if smi.get("ok"):
        for gpu in smi.get("gpus", []):
            idx = int(gpu["index"])
            util = float(gpu.get("utilization_gpu") or 0)
            idle_counts[idx] = idle_counts.get(idx, 0) + 1 if util < idle_threshold else 0
            if idle_counts[idx] >= idle_limit:
                alerts.append({"level": "WARN", "code": "gpu_idle", "message": f"GPU {idx} idle {idle_counts[idx]} times (will not kill training)"})

    threshold = float(cfg.get("disk_free_gb_threshold", 10.0))
    for disk in (disk_root, disk_cwd):
        if disk.get("ok") and disk.get("free_gb", 0) < threshold:
            alerts.append({"level": "WARN", "code": "disk_low", "message": f"Low disk free_gb={disk.get('free_gb')} path_token"})

    pid = cfg.get("pid")
    pid_status: Dict[str, Any] = {"configured": pid is not None}
    if pid is not None:
        alive = pid_alive(int(pid))
        pid_status.update({"pid": int(pid), "alive": alive})
        if not alive:
            alerts.append({"level": "ERROR", "code": "pid_dead", "message": f"PID {pid} not found"})

    metrics: Dict[str, float] = {}
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
                alerts.append({"level": "WARN", "code": "stale_log", "message": f"Log stale {age_min:.1f} > {stale_min} min"})
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
                parsed = parse_training_metrics(line)
                if not parsed:
                    continue
                fp = json.dumps(parsed, sort_keys=True)
                if fp == last_fp:
                    continue
                metrics = parsed
                last_fp = fp
                if "loss" in parsed and is_bad_loss(parsed["loss"]):
                    alerts.append({"level": "ERROR", "code": "loss_nan_inf", "message": "Non-finite loss detected"})
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
            trainer_info = {"ok": True, "global_step": data.get("global_step"), "latest_keys": sorted(latest.keys()) if isinstance(latest, dict) else []}
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
        "trainer_state": trainer_info,
        "checkpoints": checkpoints,
        "alerts": alerts,
        "note": "Read-only watch; will not terminate training",
    }
    return redact_value(sample)


def cmd_run_watch(cfg: Dict[str, Any]) -> int:
    """Periodic or once watch loop."""
    global _SHUTDOWN
    _SHUTDOWN = False
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    out_dir = Path(cfg.get("output_dir") or "reports/watch")
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "watch.jsonl"
    state: Dict[str, Any] = {"idle_counts": {}, "log_offset": 0, "last_metrics_fp": None}
    once = bool(cfg.get("once"))
    interval = max(1, int(cfg.get("interval") or 30))
    exit_code = EXIT_OK
    try:
        while True:
            sample = collect_watch_sample(cfg, state)
            append_jsonl(jsonl_path, sample)
            metrics = sample.get("metrics") or {}
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
        return EXIT_RUNTIME
    print(f"Watch data written: {jsonl_path}")
    return exit_code


def run_run_check(
    output_dir: Path,
    expected_steps: Optional[int] = None,
    framework: str = "huggingface",
) -> Dict[str, Any]:
    """Post-training completion check (LoRA/Trainer artifacts)."""
    items: List[CheckItem] = []
    reasons: List[str] = []
    adapter = get_framework_adapter(framework)

    if not output_dir.exists():
        items.append(CheckItem("output_dir", "FAIL", "Output directory does not exist", {}))
        return _pack_check(items, ["Output directory does not exist"], None, expected_steps, output_dir.name)
    if not output_dir.is_dir():
        items.append(CheckItem("output_dir", "FAIL", "output_dir is not a directory", {}))
        return _pack_check(items, ["output_dir is not a directory"], None, expected_steps, output_dir.name)

    items.append(CheckItem("output_dir", "PASS", "Output directory readable", {"name": output_dir.name}))

    state_path = adapter.locate_trainer_state(output_dir)
    trainer_state: Dict[str, Any] = {}
    global_step = None
    if state_path is None:
        items.append(CheckItem("trainer_state", "FAIL", "trainer_state.json not found", {}))
        reasons.append("Missing trainer_state.json")
    else:
        try:
            trainer_state = json.loads(state_path.read_text(encoding="utf-8"))
            if not isinstance(trainer_state, dict):
                raise ValueError("root not object")
            items.append(CheckItem("trainer_state", "PASS", "trainer_state.json parsed", {}))
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            items.append(CheckItem("trainer_state", "FAIL", f"Invalid trainer_state.json: {exc}", {}))
            reasons.append("Invalid trainer_state.json")

    if trainer_state:
        try:
            global_step = int(trainer_state["global_step"]) if trainer_state.get("global_step") is not None else None
        except (TypeError, ValueError):
            global_step = None
        if expected_steps is not None:
            if global_step is None:
                items.append(CheckItem("global_step", "FAIL", "Missing global_step", {}))
                reasons.append("Missing global_step")
            elif global_step >= int(expected_steps):
                items.append(CheckItem("global_step", "PASS", f"global_step={global_step} >= {expected_steps}", {"global_step": global_step}))
            else:
                items.append(CheckItem("global_step", "FAIL", f"global_step={global_step} < {expected_steps}", {"global_step": global_step}))
                reasons.append("Insufficient steps")
        history = trainer_state.get("log_history") or []
        bad = []
        loss_seen = eval_seen = 0
        if isinstance(history, list):
            for idx, entry in enumerate(history):
                if not isinstance(entry, dict):
                    continue
                if "loss" in entry:
                    loss_seen += 1
                    if not is_finite_number(entry["loss"]):
                        bad.append(f"loss@{idx}")
                if "eval_loss" in entry:
                    eval_seen += 1
                    if not is_finite_number(entry["eval_loss"]):
                        bad.append(f"eval_loss@{idx}")
        if bad:
            items.append(CheckItem("loss_finite", "FAIL", f"Non-finite losses: {', '.join(bad[:10])}", {}))
            reasons.append("Non-finite loss in log_history")
        else:
            items.append(CheckItem("loss_finite", "PASS" if (loss_seen or eval_seen) else "WARN", f"Finite losses (loss={loss_seen}, eval_loss={eval_seen})", {}))

        train_metrics = {}
        for key in ("train_runtime", "train_loss", "train_samples_per_second"):
            if key in trainer_state:
                train_metrics[key] = trainer_state[key]
        if isinstance(history, list):
            for entry in reversed(history):
                if isinstance(entry, dict):
                    for key in ("train_runtime", "train_loss", "train_samples_per_second"):
                        if key in entry and key not in train_metrics:
                            train_metrics[key] = entry[key]
                    if train_metrics:
                        break
        missing = [k for k in ("train_runtime", "train_loss", "train_samples_per_second") if k not in train_metrics]
        if missing:
            items.append(CheckItem("train_metrics", "WARN", f"Missing end metrics: {', '.join(missing)}", {}))
            reasons.append("Missing train end metrics")
        else:
            items.append(CheckItem("train_metrics", "PASS", "train_runtime/train_loss/train_samples_per_second present", train_metrics))

    artifacts = adapter.find_adapter_artifacts(output_dir)
    if artifacts.checkpoints:
        items.append(CheckItem("checkpoints", "PASS", f"{len(artifacts.checkpoints)} non-empty checkpoint(s)", {"names": [p.name for p in artifacts.checkpoints]}))
    else:
        items.append(CheckItem("checkpoints", "WARN", "No non-empty checkpoint-* dirs", {}))
        reasons.append("No checkpoints")

    cfg_ok = False
    for cfg_path in artifacts.adapter_configs:
        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            if isinstance(cfg, dict):
                cfg_ok = True
                items.append(CheckItem("adapter_config", "PASS", f"adapter_config.json ok (peft_type={cfg.get('peft_type')})", {}))
                break
        except (OSError, json.JSONDecodeError):
            items.append(CheckItem("adapter_config", "FAIL", "adapter_config.json invalid", {}))
            reasons.append("Invalid adapter_config.json")
            break
    if not cfg_ok and not any(i.name == "adapter_config" for i in items):
        items.append(CheckItem("adapter_config", "FAIL", "adapter_config.json not found", {}))
        reasons.append("Missing adapter_config.json")

    nonzero = [w for w in artifacts.weight_files if w.get("ok")]
    zero = [w for w in artifacts.weight_files if w.get("size_bytes") == 0]
    if nonzero:
        items.append(CheckItem("lora_weights", "PASS", f"{len(nonzero)} non-zero weight file(s)", {"files": nonzero}))
    elif zero:
        items.append(CheckItem("lora_weights", "FAIL", "Weight file(s) exist but size=0", {}))
        reasons.append("Empty LoRA weights")
    else:
        items.append(CheckItem("lora_weights", "FAIL", "No LoRA/adapter weight files found", {}))
        reasons.append("Missing LoRA weights")

    items.append(CheckItem("resume_artifacts", "INFO", "Optional resume files are not required", {}))
    return _pack_check(items, reasons, global_step, expected_steps, output_dir.name)


def _pack_check(
    items: List[CheckItem],
    reasons: List[str],
    global_step: Optional[int],
    expected_steps: Optional[int],
    output_name: str,
) -> Dict[str, Any]:
    overall = overall_status(items)
    uniq = []
    for r in reasons:
        if r not in uniq:
            uniq.append(r)
    if overall == "PASS" and not uniq:
        uniq = ["All critical checks passed"]
    return redact_value(
        {
            "tool": "train_guard",
            "command": "run check",
            "version": __version__,
            "timestamp": utc_now_iso(),
            "overall_status": overall,
            "output_dir_name": output_name,
            "expected_steps": expected_steps,
            "global_step": global_step,
            "reasons": uniq,
            "checks": [i.__dict__ for i in items],
            "disclaimer": "Artifact completeness only; not a quality or domain validity claim.",
        }
    )


def run_manifest(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Create run manifest + experiment fingerprint (no raw training content)."""
    output_dir = Path(cfg.get("output_dir") or ".")
    framework = str(cfg.get("framework") or "generic")
    parts = [
        framework,
        output_dir.name,
        str(cfg.get("expected_steps") or ""),
        str(cfg.get("seed") or ""),
    ]
    # Include sorted names of checkpoint dirs and adapter presence, not file contents
    adapter = get_framework_adapter(framework)
    ckpt_names = ",".join(p.name for p in adapter.list_checkpoints(output_dir))
    parts.append(ckpt_names)
    state = adapter.locate_trainer_state(output_dir)
    if state and state.is_file():
        try:
            data = json.loads(state.read_text(encoding="utf-8"))
            parts.append(str(data.get("global_step")))
            parts.append(str(data.get("best_metric")))
        except (OSError, json.JSONDecodeError):
            parts.append("state_unreadable")
    fingerprint = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
    manifest = {
        "tool": "train_guard",
        "command": "manifest",
        "version": __version__,
        "timestamp": utc_now_iso(),
        "framework": framework,
        "output_dir_name": output_dir.name,
        "checkpoint_names": ckpt_names.split(",") if ckpt_names else [],
        "experiment_fingerprint": fingerprint,
        "note": "Fingerprint excludes raw samples, logs text, and absolute paths.",
    }
    out = Path(cfg.get("manifest_out") or (output_dir / "train_guard_manifest.json"))
    write_json(out, redact_value(manifest), overwrite=True)
    manifest["manifest_written"] = out.name
    return redact_value(manifest)


def run_run_compare(left_dir: Path, right_dir: Path) -> Dict[str, Any]:
    """Compare two run output dirs by fingerprint/steps (no raw content)."""
    def snap(d: Path) -> Dict[str, Any]:
        fw = get_framework_adapter("huggingface")
        state = fw.locate_trainer_state(d)
        step = None
        if state and state.is_file():
            try:
                data = json.loads(state.read_text(encoding="utf-8"))
                step = data.get("global_step")
            except (OSError, json.JSONDecodeError):
                step = None
        arts = fw.find_adapter_artifacts(d)
        return {
            "name": d.name,
            "global_step": step,
            "checkpoints": len(arts.checkpoints),
            "weights": len([w for w in arts.weight_files if w.get("ok")]),
            "has_adapter_config": bool(arts.adapter_configs),
        }

    a = snap(left_dir)
    b = snap(right_dir)
    report = {
        "tool": "train_guard",
        "command": "run compare",
        "version": __version__,
        "timestamp": utc_now_iso(),
        "left": a,
        "right": b,
        "step_delta": None if a["global_step"] is None or b["global_step"] is None else (b["global_step"] - a["global_step"]),
        "overall_status": "PASS",
        "disclaimer": "Compares metadata only.",
    }
    return redact_value(report)
