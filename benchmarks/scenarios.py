"""Reproducible synthetic training scenarios; no accelerator is required."""

from __future__ import annotations

import random
from typing import Any, Dict, List, Tuple

from . import BENCHMARK_VERSION, SCHEMA_VERSION

INTERVAL_SECONDS = 30.0
SAMPLES = 20
FAULT_INDEX = 8

SCENARIOS: Tuple[Tuple[str, str | None], ...] = (
    ("nan_loss", "nan_inf"),
    ("inf_loss", "nan_inf"),
    ("loss_spike", "loss_spike"),
    ("grad_spike", "grad_spike"),
    ("step_stall", "step_stalled"),
    ("throughput_drop", "throughput_drop"),
    ("gpu_idle", "gpu_idle"),
    ("gpu_overheat", "gpu_overheat"),
    ("disk_low", "disk_low"),
    ("process_dead", "process_dead"),
    ("cuda_oom", "cuda_oom"),
    ("nccl_error", "nccl_error"),
    ("gpu_xid", "gpu_xid"),
    ("checkpoint_stale", "checkpoint_stale"),
    ("checkpoint_corrupt", "checkpoint_corrupt"),
    ("normal_training", None),
)


def _base_metrics(index: int, rng: random.Random) -> Dict[str, Any]:
    return {
        "loss": round(max(0.05, 2.0 - index * 0.035 + rng.uniform(-0.01, 0.01)), 6),
        "grad_norm": round(1.0 + rng.uniform(-0.03, 0.03), 6),
        "throughput": round(100.0 + rng.uniform(-2.0, 2.0), 6),
        "gpu_util_percent": round(82.0 + rng.uniform(-3.0, 3.0), 6),
        "gpu_temperature_c": round(67.0 + rng.uniform(-2.0, 2.0), 6),
        "disk_free_bytes": 100 * 1024**3,
        "process_alive": True,
        "checkpoint_age_seconds": float((index % 5) * INTERVAL_SECONDS),
        "checkpoint_valid": True,
    }


def build_dataset(seed: int = 20260724) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Return telemetry and expected-alert records in stable scenario/time order."""
    telemetry: List[Dict[str, Any]] = []
    expected: List[Dict[str, Any]] = []
    for scenario_number, (scenario_id, alert_kind) in enumerate(SCENARIOS):
        rng = random.Random(seed + scenario_number)
        run_id = f"benchmark-{scenario_id}"
        onset = FAULT_INDEX * INTERVAL_SECONDS
        if scenario_id == "step_stall":
            onset = (FAULT_INDEX - 1) * INTERVAL_SECONDS + 300.0
        for index in range(SAMPLES):
            metrics = _base_metrics(index, rng)
            step = index
            if index >= FAULT_INDEX:
                if scenario_id == "nan_loss":
                    metrics["loss"] = "NaN"
                elif scenario_id == "inf_loss":
                    metrics["loss"] = "Infinity"
                elif scenario_id == "loss_spike" and index == FAULT_INDEX:
                    metrics["loss"] = 8.0
                elif scenario_id == "grad_spike" and index == FAULT_INDEX:
                    metrics["grad_norm"] = 10.0
                elif scenario_id == "step_stall":
                    step = FAULT_INDEX - 1
                elif scenario_id == "throughput_drop":
                    metrics["throughput"] = 20.0
                elif scenario_id == "gpu_idle":
                    metrics["gpu_util_percent"] = 1.0
                elif scenario_id == "gpu_overheat":
                    metrics["gpu_temperature_c"] = 96.0
                elif scenario_id == "disk_low":
                    metrics["disk_free_bytes"] = 1024**3
                elif scenario_id == "process_dead":
                    metrics["process_alive"] = False
                elif scenario_id == "cuda_oom":
                    metrics["cuda_oom"] = "CUDA out of memory"
                elif scenario_id == "nccl_error":
                    metrics["nccl_error"] = "NCCL unhandled system error"
                elif scenario_id == "gpu_xid":
                    metrics["gpu_xid"] = "NVRM: Xid 79"
                elif scenario_id == "checkpoint_stale":
                    metrics["checkpoint_age_seconds"] = 1900.0
                elif scenario_id == "checkpoint_corrupt":
                    metrics["checkpoint_valid"] = False
            telemetry.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "benchmark_version": BENCHMARK_VERSION,
                    "record_type": "telemetry",
                    "scenario_id": scenario_id,
                    "run_id": run_id,
                    "timestamp_s": index * INTERVAL_SECONDS,
                    "step": step,
                    "metrics": metrics,
                }
            )
        if alert_kind is not None:
            expected.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "benchmark_version": BENCHMARK_VERSION,
                    "record_type": "expected_alert",
                    "scenario_id": scenario_id,
                    "run_id": run_id,
                    "alert_kind": alert_kind,
                    "fault_onset_s": onset,
                    "evaluation_end_s": (SAMPLES - 1) * INTERVAL_SECONDS,
                }
            )
    return telemetry, expected


__all__ = ["INTERVAL_SECONDS", "SCENARIOS", "build_dataset"]
