from __future__ import annotations

import io
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

from train_guard.adapters.base import FieldMap
from train_guard.adapters.generic import GenericDatasetAdapter
from train_guard.core import io_util
from train_guard.domain import Event, EventKind, Severity
from train_guard.env import doctor
from train_guard.eval.metrics import confusion_binary, normalize_text, run_eval
from train_guard.run import commands
from train_guard.run import watch as watch_commands
from train_guard.sinks import (
    ConsoleSink,
    JsonlSink,
    OtelJsonSink,
    WebhookSink,
    otel_json,
    prometheus_text,
)
from train_guard.supervisor import (
    CheckpointValidation,
    FileCheckpointValidator,
    ManagedProcess,
    ProcessSpec,
    RecoveryGuard,
    RecoveryPolicy,
    supervise,
)


def event(**values: object) -> Event:
    defaults: dict[str, object] = {
        "run_id": "run-1",
        "kind": EventKind.LOSS_SPIKE,
        "severity": Severity.WARNING,
        "message": "loss increased",
        "source": "test",
        "timestamp": "2025-01-01T00:00:00Z",
        "event_id": "event-1",
    }
    defaults.update(values)
    return Event(**defaults)  # type: ignore[arg-type]


def test_generic_adapter_all_json_shapes_and_limits(tmp_path: Path) -> None:
    adapter = GenericDatasetAdapter()
    empty = tmp_path / "empty.json"
    empty.write_text("", encoding="utf-8")
    assert list(adapter.iter_objects(empty)) == []
    assert list(adapter.iter_objects(empty, sample_limit=0)) == []

    lines = tmp_path / "records.txt"
    lines.write_text('42\n{"id": 1}\n[]\n{"id": 2}\n', encoding="utf-8")
    assert list(adapter.iter_objects(lines, sample_limit=1)) == [{"id": 1}]

    array = tmp_path / "array.json"
    array.write_text('[{"id": 1}, "skip", {"id": 2}]', encoding="utf-8")
    assert [row["id"] for row in adapter.iter_objects(array)] == [1, 2]

    for key in ("data", "samples", "instances", "annotations"):
        wrapped = tmp_path / f"{key}.json"
        wrapped.write_text(json.dumps({key: [{"id": key}, None]}), encoding="utf-8")
        assert list(adapter.iter_objects(wrapped)) == [{"id": key}]


def test_generic_adapter_extracts_media_content_and_records(tmp_path: Path) -> None:
    adapter = GenericDatasetAdapter(
        FieldMap(
            group_id="case",
            split="partition",
            media="assets",
            messages="chat",
            input_field="prompt",
            output_field="response",
        )
    )
    raw = {
        "id": 9,
        "partition": "train",
        "assets": ["a.png", {"path": "b.png"}, {"url": "https://example.invalid/x"}],
        "chat": [{"role": "user", "content": "hi"}, "skip", {"role": "gpt", "content": None}],
    }
    assert [item.path for item in adapter.extract_media_refs(raw)] == [
        "a.png",
        "b.png",
        "https://example.invalid/x",
    ]
    assert adapter.extract_group_id(raw) == "9"
    assert adapter.extract_split(raw) == "train"
    assert adapter.extract_answer_text(raw) == ""
    assert adapter.extract_media_refs({"assets": ""}) == []
    assert adapter.extract_media_refs({}) == []
    assert adapter.extract_group_id({}) is None
    assert adapter.extract_split({}) is None

    fallback = {"prompt": 10, "answer": 20}
    content = adapter.extract_messages_or_io(fallback)
    assert (content.input_text, content.output_text) == ("10", "20")
    assert adapter.extract_answer_text(fallback) == "20"

    source = tmp_path / "source.json"
    source.write_text(json.dumps([raw]), encoding="utf-8")
    record = next(adapter.iter_records(source))
    assert record.index == 0
    assert record.group_id == "9"


@pytest.mark.parametrize(
    ("side_effect", "expected_code", "expected_text"),
    [
        (FileNotFoundError(), 127, "not found"),
        (subprocess.TimeoutExpired("tool", 0.1), 124, "timed out"),
        (OSError("denied"), 1, "denied"),
    ],
)
def test_run_command_errors(
    side_effect: BaseException, expected_code: int, expected_text: str
) -> None:
    with mock.patch.object(io_util.subprocess, "run", side_effect=side_effect):
        code, stdout, stderr = io_util.run_command(["tool"], timeout=0.1)
    assert (code, stdout) == (expected_code, "")
    assert expected_text in stderr


def test_run_command_success() -> None:
    completed = SimpleNamespace(returncode=3, stdout=None, stderr="bad")
    with mock.patch.object(io_util.subprocess, "run", return_value=completed) as run:
        assert io_util.run_command(["tool", "arg"]) == (3, "", "bad")
    assert run.call_args.kwargs["shell"] is False


def test_io_helpers_edges(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "value.txt"
    io_util.atomic_write_text(path, "first")
    with pytest.raises(FileExistsError):
        io_util.atomic_write_text(path, "second")
    io_util.atomic_write_text(path, "second", overwrite=True)
    assert path.read_text(encoding="utf-8") == "second"

    log = tmp_path / "events.jsonl"
    io_util.append_jsonl(log, {"value": "é"})
    assert json.loads(log.read_text(encoding="utf-8")) == {"value": "é"}
    assert len(io_util.sha256_file(path, chunk_size=2)) == 64

    with mock.patch.object(io_util.shutil, "disk_usage", side_effect=OSError("bad disk")):
        assert io_util.get_disk_usage(tmp_path)["ok"] is False
    with mock.patch.object(io_util.os, "getloadavg", return_value=(1.0, 2.0, 3.0), create=True):
        assert io_util.get_cpu_load()["load15"] == 3.0
    with mock.patch.object(
        io_util.os, "getloadavg", side_effect=OSError("unsupported"), create=True
    ):
        assert io_util.get_cpu_load()["ok"] is False


def test_memory_and_pid_fallbacks() -> None:
    vm = SimpleNamespace(total=100, available=40, used=60, percent=60.0)
    psutil = SimpleNamespace(virtual_memory=lambda: vm)
    with mock.patch("train_guard.core.optional.try_import_psutil", return_value=psutil):
        assert io_util.get_memory_info()["source"] == "psutil"
    with (
        mock.patch("train_guard.core.optional.try_import_psutil", return_value=None),
        mock.patch.object(io_util.platform, "system", return_value="Windows"),
    ):
        assert io_util.get_memory_info()["ok"] is False

    assert io_util.pid_alive(0) is False
    for error, expected in (
        (None, True),
        (ProcessLookupError(), False),
        (PermissionError(), True),
        (OSError(), False),
    ):
        with mock.patch.object(io_util.os, "kill", side_effect=error):
            assert io_util.pid_alive(123) is expected


def test_doctor_torch_gpu_and_model_paths(tmp_path: Path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text('{"model_type": "tiny"}', encoding="utf-8")
    (model / "weights.safetensors").write_bytes(b"x")
    (model / "model.safetensors.index.json").write_text(
        '{"weight_map": {"layer": "missing.safetensors"}}',
        encoding="utf-8",
    )
    fake_torch = SimpleNamespace(
        __version__="2.test",
        version=SimpleNamespace(cuda="12.test"),
        cuda=SimpleNamespace(is_available=lambda: True),
    )
    smi = {
        "ok": True,
        "count": 1,
        "driver_version": "test",
        "gpus": [{"index": 0, "name": "mock", "memory_total_mb": 1.0}],
    }
    with (
        mock.patch.object(doctor, "try_import_torch", return_value=fake_torch),
        mock.patch.object(doctor, "query_nvidia_smi", return_value=smi),
        mock.patch.object(doctor, "package_version", return_value=None),
        mock.patch.object(
            doctor,
            "get_disk_usage",
            return_value={"ok": True, "free_gb": 10, "free_bytes": 1},
        ),
    ):
        report = doctor.run_doctor(model, expected_gpus=2)
    checks = {item["name"]: item for item in report["checks"]}
    assert checks["torch_cuda_available"]["status"] == "PASS"
    assert checks["gpus"]["status"] == "WARN"
    assert checks["model_config"]["status"] == "PASS"
    assert checks["safetensors_index"]["status"] == "FAIL"


def test_doctor_missing_torch_invalid_models_and_bundle(tmp_path: Path) -> None:
    missing = tmp_path / "missing"
    with (
        mock.patch.object(doctor, "try_import_torch", return_value=None),
        mock.patch.object(
            doctor,
            "query_nvidia_smi",
            return_value={"ok": False, "error": "missing"},
        ),
        mock.patch.object(doctor, "package_version", return_value="1"),
        mock.patch.object(doctor, "get_disk_usage", return_value={"ok": False}),
    ):
        report = doctor.run_doctor(missing)
    assert any(
        item["name"] == "model_path" and item["status"] == "FAIL" for item in report["checks"]
    )
    assert doctor.status_to_exit("FAIL") == 2
    assert doctor.status_to_exit("WARN") == 1
    assert doctor.status_to_exit("PASS") == 0

    script = tmp_path / "bundle.py"
    script.write_text("print('ok')", encoding="utf-8")
    info = doctor.run_bundle_info(script)
    assert info["sha256"]
    with mock.patch.object(doctor, "sha256_file", side_effect=OSError("unreadable")):
        assert doctor.run_bundle_info(script)["sha256"] is None
    assert doctor.run_bundle_info(missing)["sha256"] is None


def test_eval_metrics_binary_references_keywords_and_missing(tmp_path: Path) -> None:
    predictions = tmp_path / "predictions.jsonl"
    references = tmp_path / "references.jsonl"
    predictions.write_text(
        "\n".join(
            [
                json.dumps({"id": "a", "prediction": " YES! ", "predicted_label": 1, "label": 1}),
                json.dumps({"id": "b", "prediction": "", "predicted_label": 1, "label": 0}),
                json.dumps({"id": "c"}),
            ]
        ),
        encoding="utf-8",
    )
    references.write_text(
        "\n".join(
            [
                json.dumps({"id": "a", "reference": "yes"}),
                json.dumps({"id": "b", "answer": "must contain alpha"}),
            ]
        ),
        encoding="utf-8",
    )
    report = run_eval(
        {
            "predictions": str(predictions),
            "references": str(references),
            "keywords": ["alpha"],
            "report_dir": str(tmp_path / "report"),
        }
    )
    metrics = report["metrics"]
    assert report["overall_status"] == "WARN"
    assert metrics["missing_predictions"] == 1
    assert metrics["empty_predictions"] == 1
    assert metrics["classification"]["type"] == "binary"
    assert normalize_text(" YES!? ") == "yes"
    assert confusion_binary([1, 1, 0, 0], [1, 0, 0, 1], 1)["accuracy"] == 0.5


def test_eval_multiclass_empty_and_validation(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="predictions"):
        run_eval({"report_dir": str(tmp_path)})
    empty = tmp_path / "empty.json"
    empty.write_text("[]", encoding="utf-8")
    assert (
        run_eval({"predictions": str(empty), "report_dir": str(tmp_path / "empty-report")})[
            "overall_status"
        ]
        == "FAIL"
    )

    multi = tmp_path / "multi.json"
    multi.write_text(
        json.dumps(
            [
                {"prediction": "a", "reference": "a", "predicted_label": "a", "label": "a"},
                {"prediction": "b", "reference": "c", "predicted_label": "b", "label": "c"},
            ]
        ),
        encoding="utf-8",
    )
    result = run_eval({"predictions": str(multi), "report_dir": str(tmp_path / "multi-report")})
    assert result["metrics"]["classification"]["type"] == "multiclass"


def test_query_nvidia_smi_parses_and_skips_bad_rows() -> None:
    output = "\n".join(
        [
            "short,row",
            "bad, GPU, 1, 2, 3, 4, 5, 6, driver",
            "0, Mock GPU, 100, 20, 80, 50, , N/A, 555",
        ]
    )
    with mock.patch.object(watch_commands, "run_command", return_value=(0, output, "")):
        result = watch_commands.query_nvidia_smi()
    assert result["count"] == 1
    assert result["gpus"][0]["temperature_c"] is None
    assert result["gpus"][0]["power_draw_w"] is None
    with mock.patch.object(watch_commands, "run_command", return_value=(1, "out", "")):
        assert watch_commands.query_nvidia_smi()["error"] == "out"


def test_collect_watch_sample_alerts_and_trainer_state(tmp_path: Path) -> None:
    log = tmp_path / "train.log"
    log.write_text("loss: NaN\nloss: NaN\nstep: 3 loss: 1.0\n", encoding="utf-8")
    state_file = tmp_path / "trainer_state.json"
    state_file.write_text(
        json.dumps({"global_step": 3, "log_history": [{"eval_loss": 0.5, "bad": "x"}]}),
        encoding="utf-8",
    )
    adapter = mock.MagicMock()
    adapter.locate_trainer_state.return_value = state_file
    adapter.list_checkpoints.return_value = [tmp_path / "checkpoint-3"]
    smi = {
        "ok": True,
        "count": 1,
        "gpus": [{"index": 0, "utilization_gpu": 0}],
    }
    with (
        mock.patch.object(watch_commands, "query_nvidia_smi", return_value=smi),
        mock.patch.object(watch_commands, "get_cpu_load", return_value={"ok": True}),
        mock.patch.object(watch_commands, "get_memory_info", return_value={"ok": True}),
        mock.patch.object(
            watch_commands,
            "get_disk_usage",
            return_value={"ok": True, "free_gb": 1, "free_bytes": 1},
        ),
        mock.patch.object(watch_commands, "pid_alive", return_value=False),
        mock.patch.object(watch_commands, "get_framework_adapter", return_value=adapter),
        mock.patch.object(watch_commands.time, "time", return_value=log.stat().st_mtime + 3600),
    ):
        state: dict[str, object] = {"log_offset": 9999}
        sample = watch_commands.collect_watch_sample(
            {
                "expected_gpu_count": 2,
                "idle_gpu_consecutive": 1,
                "disk_free_gb_threshold": 2,
                "pid": 999,
                "log_file": str(log),
                "stale_log_minutes": 1,
                "output_dir": str(tmp_path),
            },
            state,
        )
    codes = {alert["code"] for alert in sample["alerts"]}
    assert {"gpu_count_mismatch", "gpu_idle", "disk_low", "pid_dead", "stale_log"} <= codes
    assert sample["trainer_state"]["global_step"] == 3


def test_collect_watch_sample_missing_and_invalid_files(tmp_path: Path) -> None:
    state_file = tmp_path / "trainer_state.json"
    state_file.write_text("{broken", encoding="utf-8")
    adapter = mock.MagicMock()
    adapter.locate_trainer_state.return_value = state_file
    adapter.list_checkpoints.return_value = []
    with (
        mock.patch.object(watch_commands, "query_nvidia_smi", return_value={"ok": False}),
        mock.patch.object(watch_commands, "get_cpu_load", return_value={"ok": False}),
        mock.patch.object(watch_commands, "get_memory_info", return_value={"ok": False}),
        mock.patch.object(watch_commands, "get_disk_usage", return_value={"ok": False}),
        mock.patch.object(watch_commands, "get_framework_adapter", return_value=adapter),
    ):
        sample = watch_commands.collect_watch_sample(
            {"log_file": str(tmp_path / "missing.log"), "output_dir": str(tmp_path)},
            {},
        )
    assert sample["log"]["exists"] is False
    assert sample["trainer_state"]["error"] == "JSONDecodeError"


def test_reliability_values_and_signal_handler(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    adapter = mock.MagicMock()
    adapter.list_checkpoints.return_value = [checkpoint]
    sample = {
        "metrics": {"loss": 1},
        "gpus": {
            "gpus": [
                {"utilization_gpu": 20, "temperature_c": 70},
                {"utilization_gpu": 40, "temperature_c": 80},
            ]
        },
        "disk": {
            "root": {"ok": True, "free_bytes": 20},
            "cwd": {"ok": True, "free_bytes": 10},
        },
        "pid": {"configured": True, "alive": True},
    }
    with (
        mock.patch.object(watch_commands, "get_framework_adapter", return_value=adapter),
        mock.patch.object(watch_commands.time, "time", return_value=checkpoint.stat().st_mtime + 5),
    ):
        values = watch_commands._reliability_values(sample, 4, tmp_path, "generic")
    assert values["gpu_util_percent"] == 30
    assert values["gpu_temperature_c"] == 80
    assert values["disk_free_bytes"] == 10
    assert values["checkpoint_age_seconds"] == 5
    watch_commands._handle_signal(2, None)
    assert watch_commands._SHUTDOWN is True


def test_run_check_rejection_paths(tmp_path: Path) -> None:
    missing = tmp_path / "missing"
    assert commands.run_run_check(missing)["overall_status"] == "FAIL"
    file_path = tmp_path / "file"
    file_path.write_text("x", encoding="utf-8")
    assert commands.run_run_check(file_path)["overall_status"] == "FAIL"

    run = tmp_path / "run"
    run.mkdir()
    (run / "trainer_state.json").write_text(
        json.dumps({"global_step": "bad", "log_history": ["skip", {"eval_loss": "inf"}]}),
        encoding="utf-8",
    )
    (run / "adapter_config.json").write_text("{broken", encoding="utf-8")
    (run / "adapter_model.safetensors").write_bytes(b"")
    report = commands.run_run_check(run, expected_steps=10, training_type="adapter")
    assert report["overall_status"] == "FAIL"
    assert "Missing global_step" in report["reasons"]
    assert "Invalid adapter_config.json" in report["reasons"]
    assert "Empty adapter weights" in report["reasons"]
    unsupported = commands.run_run_check(run, training_type="mystery")
    assert "Unsupported training type" in unsupported["reasons"]


def test_run_check_full_weight_and_lifecycle_variants(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    (run / "trainer_state.json").write_text(
        json.dumps(
            {
                "global_step": 1,
                "train_runtime": 1,
                "train_loss": 1,
                "train_samples_per_second": 1,
                "log_history": [{"loss": 1}],
            }
        ),
        encoding="utf-8",
    )
    (run / "model.safetensors").write_bytes(b"")
    report = commands.run_run_check(run, expected_steps=2, training_type="full")
    assert "Insufficient steps" in report["reasons"]
    assert "Empty full-model weights" in report["reasons"]


def test_manifest_and_compare_file_inputs(tmp_path: Path) -> None:
    not_dir = tmp_path / "not-dir"
    not_dir.write_text("x", encoding="utf-8")
    manifest_path = tmp_path / "invalid-manifest.json"
    manifest = commands.run_manifest(
        {"output_dir": str(not_dir), "manifest_out": str(manifest_path)}
    )
    assert manifest["manifest_written"] == manifest_path.name
    comparison = commands.run_run_compare(not_dir, tmp_path / "missing")
    assert comparison["left"]["reasons"] == ["path is not a directory"]


def test_sinks_formats_and_mock_webhook(tmp_path: Path) -> None:
    item = event()
    stream = io.StringIO()
    console = ConsoleSink(stream)
    jsonl = JsonlSink(tmp_path / "events.jsonl")
    otel = OtelJsonSink(tmp_path / "otel.jsonl")
    for sink in (console, jsonl, otel):
        sink.emit(item)
    assert "[WARNING]" in stream.getvalue()
    assert (
        json.loads((tmp_path / "events.jsonl").read_text(encoding="utf-8"))["event_id"] == "event-1"
    )
    assert (
        json.loads((tmp_path / "otel.jsonl").read_text(encoding="utf-8"))["severityText"]
        == "WARNING"
    )
    assert otel_json(item)["timeUnixNano"] == "1735689600000000000"
    text = prometheus_text([item, item], metric_prefix="train guard!")
    assert "train_guard__events_total" in text
    assert text.endswith(" 2\n")

    with pytest.raises(ValueError):
        WebhookSink("file:///tmp/test")
    response = mock.MagicMock()
    response.status = 204
    response.__enter__.return_value = response
    webhook = WebhookSink("https://example.invalid/hook", headers={"X-Test": "1"})
    with mock.patch("urllib.request.urlopen", return_value=response) as urlopen:
        webhook.emit(item)
    assert urlopen.call_args.kwargs["timeout"] == 5.0
    response.status = 500
    with (
        mock.patch("urllib.request.urlopen", return_value=response),
        pytest.raises(RuntimeError, match="500"),
    ):
        webhook.emit(item)


def test_checkpoint_validator_and_recovery_guard(tmp_path: Path) -> None:
    validator = FileCheckpointValidator(["a.bin", "missing.bin"])
    assert validator.validate(tmp_path / "missing").valid is False
    (tmp_path / "a.bin").write_bytes(b"")
    result = validator.validate(tmp_path)
    assert "empty required file: a.bin" in result.errors
    assert "missing required file: missing.bin" in result.errors

    (tmp_path / "a.bin").write_bytes(b"data")
    digest = FileCheckpointValidator(["a.bin"]).validate(tmp_path).digest
    assert digest
    mismatch = FileCheckpointValidator(["a.bin"], expected_digest="bad").validate(tmp_path)
    assert mismatch.errors == ("checkpoint digest mismatch",)
    assert FileCheckpointValidator(["a.bin"], expected_digest=digest).validate(tmp_path).valid

    guard = RecoveryGuard(RecoveryPolicy(max_restarts=1, window_seconds=10))
    assert guard.permit_restart(10)
    assert not guard.permit_restart(11)
    assert guard.permit_restart(30)
    probe = mock.MagicMock()
    probe.healthy.side_effect = [False, True]
    fast = RecoveryGuard(RecoveryPolicy(probe_timeout_seconds=10, probe_interval_seconds=0))
    with mock.patch("train_guard.supervisor.time.monotonic", side_effect=[0, 1, 2]):
        assert fast.wait_until_healthy(probe)


def test_managed_process_state_guards() -> None:
    managed = ManagedProcess(ProcessSpec("python"))
    assert managed.pid is None
    assert managed.poll() is None
    with pytest.raises(RuntimeError, match="not started"):
        managed.wait()
    with pytest.raises(RuntimeError, match="not started"):
        managed.terminate()


def test_supervise_branch_results(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    outcomes = iter([1, 0])

    class FakeProcess:
        def __init__(self, spec: ProcessSpec) -> None:
            self.exit_code = next(outcomes)

        def start(self) -> int:
            return 123

        def wait(self) -> int:
            return self.exit_code

        def terminate(self) -> int:
            return 9

    monkeypatch.setattr("train_guard.supervisor.ManagedProcess", FakeProcess)
    audit: list[object] = []
    validator = mock.MagicMock()
    validator.validate.return_value = CheckpointValidation(True, digest="digest")
    result = supervise(
        ProcessSpec("mock"),
        restart_enabled=True,
        recovery_guard=RecoveryGuard(RecoveryPolicy(max_restarts=1)),
        checkpoint_path=tmp_path,
        checkpoint_validator=validator,
        audit=audit.append,
    )
    assert result.stopped_reason == "completed"
    assert result.restart_count == 1
    assert len(audit) == 5


@pytest.mark.parametrize(
    ("restart", "validation", "expected"),
    [
        (False, None, "restart_disabled"),
        (True, None, "checkpoint_validation_not_configured"),
        (
            True,
            CheckpointValidation(False, ("bad checkpoint",)),
            "checkpoint_invalid",
        ),
    ],
)
def test_supervise_failure_branches(
    monkeypatch: pytest.MonkeyPatch,
    restart: bool,
    validation: CheckpointValidation | None,
    expected: str,
) -> None:
    fake = mock.MagicMock()
    fake.start.return_value = 1
    fake.wait.return_value = 2
    monkeypatch.setattr("train_guard.supervisor.ManagedProcess", lambda spec: fake)
    validator = None
    path = None
    if validation is not None:
        validator = mock.MagicMock()
        validator.validate.return_value = validation
        path = Path("checkpoint")
    result = supervise(
        ProcessSpec("mock"),
        restart_enabled=restart,
        checkpoint_path=path,
        checkpoint_validator=validator,
    )
    assert result.stopped_reason == expected
