from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.evaluate import evaluate, read_jsonl, run_train_guard_rules
from benchmarks.generate import generate


def test_tracked_benchmark_reference_matches_rules(tmp_path: Path) -> None:
    telemetry_path, expected_path = generate(tmp_path, seed=20260724)
    telemetry = read_jsonl(telemetry_path)
    expected = read_jsonl(expected_path)
    actual = evaluate(telemetry, expected, run_train_guard_rules(telemetry))
    reference = json.loads(
        (ROOT / "benchmarks" / "results" / "reference-20260724.json").read_text(encoding="utf-8")
    )

    assert actual["benchmark_version"] == reference["benchmark_version"]
    assert actual["counts"] == reference["counts"]
    assert actual["detection"]["mttd_seconds"] == reference["detection"]["mttd_seconds"]
    assert actual["detection"]["precision"] == reference["detection"]["precision"]
    assert actual["detection"]["recall"] == reference["detection"]["recall"]
    assert (
        actual["false_positives_per_1000_training_hours"]
        == reference["false_positives_per_1000_training_hours"]
    )
