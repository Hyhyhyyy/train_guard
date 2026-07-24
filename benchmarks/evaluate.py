"""Evaluate detector alerts against the public fault-injection benchmark."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from . import BENCHMARK_VERSION, SCHEMA_VERSION
from .scenarios import INTERVAL_SECONDS


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("schema_version") != SCHEMA_VERSION:
                raise ValueError(f"{path}:{line_number}: unsupported schema_version")
            records.append(record)
    return records


def _decode_metric(value: Any) -> Any:
    return {"NaN": math.nan, "Infinity": math.inf, "-Infinity": -math.inf}.get(value, value)


def run_train_guard_rules(telemetry: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Optional reference adapter. Import project rules only when explicitly requested."""
    try:
        from train_guard.rules import RuleConfig, RuleContext, RuleEngine
    except (ImportError, ModuleNotFoundError):
        source_root = Path(__file__).resolve().parents[1] / "src"
        if not (source_root / "train_guard" / "rules" / "__init__.py").is_file():
            raise
        sys.path.insert(0, str(source_root))
        sys.modules.pop("train_guard", None)
        from train_guard.rules import RuleConfig, RuleContext, RuleEngine

    engine = RuleEngine(RuleConfig())
    detections: List[Dict[str, Any]] = []
    for sample in telemetry:
        values = {key: _decode_metric(value) for key, value in sample["metrics"].items()}
        values["step"] = sample["step"]
        context = RuleContext(
            run_id=str(sample["run_id"]),
            values=values,
            now=float(sample["timestamp_s"]),
            source="benchmark",
        )
        for event in engine.evaluate(context):
            detections.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "record_type": "alert",
                    "run_id": sample["run_id"],
                    "alert_kind": event.kind.value,
                    "timestamp_s": sample["timestamp_s"],
                }
            )
    return detections


def evaluate(
    telemetry: Sequence[Mapping[str, Any]],
    expected: Sequence[Mapping[str, Any]],
    detections: Sequence[Mapping[str, Any]],
    recoveries: Sequence[Mapping[str, Any]] = (),
) -> Dict[str, Any]:
    """Compute event-level detection and recovery metrics.

    Repeated alerts with the same run/kind are treated as one alert lifecycle.
    """
    first_alert: Dict[tuple[str, str], float] = {}
    for alert in detections:
        key = (str(alert["run_id"]), str(alert["alert_kind"]))
        timestamp = float(alert["timestamp_s"])
        first_alert[key] = min(timestamp, first_alert.get(key, timestamp))

    expected_by_key = {(str(item["run_id"]), str(item["alert_kind"])): item for item in expected}
    latencies: List[float] = []
    matched: set[tuple[str, str]] = set()
    false_positives = 0
    for key, timestamp in first_alert.items():
        target = expected_by_key.get(key)
        if target is None or timestamp < float(target["fault_onset_s"]):
            false_positives += 1
            continue
        if timestamp <= float(target["evaluation_end_s"]):
            matched.add(key)
            latencies.append(timestamp - float(target["fault_onset_s"]))
        else:
            false_positives += 1

    true_positives = len(matched)
    false_negatives = len(expected) - true_positives
    predicted_positives = true_positives + false_positives
    precision = true_positives / predicted_positives if predicted_positives else 0.0
    recall = true_positives / len(expected) if expected else 0.0

    spans: Dict[str, List[float]] = {}
    for sample in telemetry:
        spans.setdefault(str(sample["run_id"]), []).append(float(sample["timestamp_s"]))
    observed_seconds = sum(max(times) - min(times) + INTERVAL_SECONDS for times in spans.values())
    observed_hours = observed_seconds / 3600.0

    attempts = len(recoveries)
    successful = sum(1 for item in recoveries if item.get("status") == "succeeded")
    avoided_seconds = 0.0
    for recovery in recoveries:
        if recovery.get("status") != "succeeded":
            continue
        run_id = str(recovery["run_id"])
        candidates = [item for item in expected if str(item["run_id"]) == run_id]
        if candidates:
            target = candidates[0]
            success_at = max(float(recovery["timestamp_s"]), float(target["fault_onset_s"]))
            avoided_seconds += max(0.0, float(target["evaluation_end_s"]) - success_at)

    return {
        "schema_version": SCHEMA_VERSION,
        "benchmark_version": BENCHMARK_VERSION,
        "record_type": "benchmark_result",
        "counts": {
            "expected_faults": len(expected),
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
        },
        "detection": {
            "precision": precision,
            "recall": recall,
            "mttd_seconds": sum(latencies) / len(latencies) if latencies else None,
            "detection_latencies_seconds": sorted(latencies),
        },
        "false_positives_per_1000_training_hours": (
            false_positives / observed_hours * 1000.0 if observed_hours else 0.0
        ),
        "observed_training_hours": observed_hours,
        "recovery": {
            "attempts": attempts,
            "successes": successful,
            "success_rate": successful / attempts if attempts else None,
            "estimated_avoided_loss_seconds": avoided_seconds,
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--telemetry", type=Path, required=True)
    parser.add_argument("--expected", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--alerts", type=Path)
    source.add_argument("--use-train-guard-rules", action="store_true")
    parser.add_argument("--recoveries", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--min-precision", type=float, default=0.0)
    parser.add_argument("--min-recall", type=float, default=0.0)
    parser.add_argument("--max-mttd-seconds", type=float)
    parser.add_argument("--min-recovery-success-rate", type=float)
    args = parser.parse_args(argv)

    telemetry = read_jsonl(args.telemetry)
    expected = read_jsonl(args.expected)
    alerts = (
        run_train_guard_rules(telemetry) if args.use_train_guard_rules else read_jsonl(args.alerts)
    )
    recoveries = read_jsonl(args.recoveries) if args.recoveries else []
    result = evaluate(telemetry, expected, alerts, recoveries)
    rendered = json.dumps(result, indent=2, sort_keys=True, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    detection = result["detection"]
    failures = []
    if detection["precision"] < args.min_precision:
        failures.append("precision")
    if detection["recall"] < args.min_recall:
        failures.append("recall")
    if (
        args.max_mttd_seconds is not None
        and detection["mttd_seconds"] is not None
        and detection["mttd_seconds"] > args.max_mttd_seconds
    ):
        failures.append("mttd")
    recovery_rate = result["recovery"]["success_rate"]
    if args.min_recovery_success_rate is not None and (
        recovery_rate is None or recovery_rate < args.min_recovery_success_rate
    ):
        failures.append("recovery_success_rate")
    if failures:
        print(f"acceptance failed: {', '.join(failures)}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
