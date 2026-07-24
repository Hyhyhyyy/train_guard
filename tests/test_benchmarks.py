from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from benchmarks.evaluate import evaluate, run_train_guard_rules
from benchmarks.generate import generate
from benchmarks.scenarios import SCENARIOS, build_dataset


class BenchmarkTests(unittest.TestCase):
    def test_dataset_is_reproducible_and_covers_scenarios(self) -> None:
        first = build_dataset(42)
        second = build_dataset(42)
        self.assertEqual(first, second)
        telemetry, expected = first
        actual_scenarios = {record["scenario_id"] for record in telemetry}
        self.assertEqual(actual_scenarios, {item[0] for item in SCENARIOS})
        self.assertEqual(len(expected), len(SCENARIOS) - 1)
        self.assertNotIn("normal_training", {record["scenario_id"] for record in expected})

    def test_generator_writes_strict_versioned_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            telemetry_path, expected_path = generate(Path(directory), seed=7)
            for path in (telemetry_path, expected_path):
                lines = path.read_text(encoding="utf-8").splitlines()
                records = [json.loads(line) for line in lines]
                self.assertTrue(records)
                self.assertTrue(all(record["schema_version"] == "1.0" for record in records))
                self.assertTrue(all(record["benchmark_version"] == "1.0.0" for record in records))
            nan_sample = next(
                record
                for record in (json.loads(line) for line in telemetry_path.read_text().splitlines())
                if record["scenario_id"] == "nan_loss" and record["timestamp_s"] == 240.0
            )
            self.assertEqual(nan_sample["metrics"]["loss"], "NaN")

    def test_evaluator_metrics_and_recovery(self) -> None:
        telemetry, expected = build_dataset()
        alerts = [
            {
                "schema_version": "1.0",
                "record_type": "alert",
                "run_id": item["run_id"],
                "alert_kind": item["alert_kind"],
                "timestamp_s": item["fault_onset_s"] + 30.0,
            }
            for item in expected
        ]
        alerts.append(
            {
                "schema_version": "1.0",
                "record_type": "alert",
                "run_id": "benchmark-normal_training",
                "alert_kind": "loss_spike",
                "timestamp_s": 240.0,
            }
        )
        recovery = [
            {
                "schema_version": "1.0",
                "record_type": "recovery",
                "run_id": expected[0]["run_id"],
                "timestamp_s": 300.0,
                "status": "succeeded",
            }
        ]
        result = evaluate(telemetry, expected, alerts, recovery)
        self.assertEqual(result["counts"]["true_positives"], len(expected))
        self.assertEqual(result["counts"]["false_positives"], 1)
        self.assertEqual(result["detection"]["recall"], 1.0)
        self.assertEqual(result["detection"]["mttd_seconds"], 30.0)
        self.assertEqual(result["recovery"]["success_rate"], 1.0)
        self.assertEqual(result["recovery"]["estimated_avoided_loss_seconds"], 270.0)

    def test_optional_rules_adapter_detects_every_fault_without_healthy_alerts(self) -> None:
        telemetry, expected = build_dataset()
        alerts = run_train_guard_rules(telemetry)
        result = evaluate(telemetry, expected, alerts)
        self.assertEqual(result["counts"]["false_negatives"], 0)
        self.assertEqual(result["counts"]["false_positives"], 0)


if __name__ == "__main__":
    unittest.main()
