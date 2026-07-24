# Train Guard public fault-injection benchmark

This benchmark uses only the Python standard library and CPU. A fixed seed
generates synthetic telemetry for NaN loss, infinite loss, loss spikes,
gradient spikes, step stalls, throughput drops, GPU idle/overheat, low disk,
dead processes, CUDA OOM, NCCL/Xid errors, stale/corrupt checkpoints, and a
healthy-training negative.

## Run

From the repository root:

```console
python -m benchmarks.generate --output-dir benchmark-output --seed 20260724 --json
python -m benchmarks.evaluate --telemetry benchmark-output/telemetry.jsonl --expected benchmark-output/expected_alerts.jsonl --alerts your-alerts.jsonl --output benchmark-output/result.json
```

If the current source tree contains an importable `train_guard.rules`, replace
`--alerts` with `--use-train-guard-rules` to run the optional reference
adapter. That import is delayed until the flag is used. Generation, data
formats, and evaluation do not depend on unfinished Train Guard modules.
External detectors only need to emit the `alert` interface in `schema.json`.

## Data and schema

Every line in `telemetry.jsonl` and `expected_alerts.jsonl` is an independent
JSON object carrying `schema_version: "1.0"` and
`benchmark_version: "1.0.0"`. The complete JSON Schema is `schema.json`.
Because JSON has no finite-number exceptions, NaN, positive infinity, and
negative infinity are encoded as `"NaN"`, `"Infinity"`, and `"-Infinity"`.
Detector adapters must decode these before numeric evaluation.

Minimum detector output:

```json
{"schema_version":"1.0","record_type":"alert","run_id":"benchmark-nan_loss","alert_kind":"nan_inf","timestamp_s":240}
```

Optional recovery input:

```json
{"schema_version":"1.0","record_type":"recovery","run_id":"benchmark-nan_loss","timestamp_s":270,"status":"succeeded"}
```

## Metric definitions

- Detection latency is first matching alert time minus detectable fault onset.
  MTTD is the mean latency among detected faults.
- Precision and recall use event-level matching. Repeated alerts with the same
  run and kind are one alert lifecycle. Wrong kinds, alerts on healthy
  training, pre-onset alerts, and alerts outside the window are false positives.
- False positives per 1,000 training hours use all synthetic observed time,
  including the final sampling interval, as the denominator.
- Recovery success rate is successful records divided by attempts. Estimated
  avoided loss time is summed for successful recoveries as
  `window end - max(recovery time, fault onset)`. This transparent synthetic
  proxy is not a measured cost or hardware saving.

## Claim limitation

The benchmark simulates GPU utilization and temperature fields but uses no
real GPU, training workload, or power measurement. Results from this benchmark
must not claim validated GPU-hour savings, cost savings, energy reduction, or
production recovery impact. Such claims require separate real-GPU workloads,
hardware details, measurement methods, and uncertainty reporting.
