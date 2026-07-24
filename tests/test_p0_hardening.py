from __future__ import annotations

import json
import multiprocessing
import sys
import time
from pathlib import Path
from typing import Any
from unittest import mock

import pytest

from train_guard.domain import Event, EventKind, Severity
from train_guard.eval.metrics import run_eval
from train_guard.state import AuditLog, StateStore
from train_guard.supervisor import ManagedProcess, ProcessSpec


def _sqlite_increment_worker(path: str, start: Any, errors: Any, iterations: int) -> None:
    try:
        start.wait(20)
        with StateStore(Path(path)) as store:
            for _ in range(iterations):
                store.increment("stress", "count")
    except BaseException as exc:  # pragma: no cover - asserted in parent process
        errors.put(f"{type(exc).__name__}: {exc}")


class _Sink:
    def __init__(self) -> None:
        self.events: list[Event] = []

    def emit(self, event: Event) -> None:
        self.events.append(event)


def _watch_sample(step: int, loss: float = 1.0) -> dict[str, Any]:
    return {
        "timestamp": "2026-01-01T00:00:00Z",
        "gpus": {"ok": False, "count": 0, "gpus": []},
        "cpu": {"ok": False},
        "memory": {"ok": False},
        "disk": {},
        "pid": {"configured": False},
        "log": {"configured": False},
        "metrics": {"loss": loss, "step": step},
        "runtime_signatures": {},
        "trainer_state": {"ok": True, "global_step": step},
        "checkpoints": [],
        "alerts": [],
        "note": "test",
    }


def test_sqlite_first_open_and_increment_are_multiprocess_safe(tmp_path: Path) -> None:
    context = multiprocessing.get_context("spawn")
    start = context.Event()
    errors = context.Queue()
    path = tmp_path / "shared.sqlite"
    workers = [
        context.Process(target=_sqlite_increment_worker, args=(str(path), start, errors, 80))
        for _ in range(6)
    ]
    for worker in workers:
        worker.start()
    start.set()
    for worker in workers:
        worker.join(30)
        assert worker.exitcode == 0
    assert errors.empty()
    with StateStore(path) as store:
        assert store.increment("stress", "count", 0) == 480


def test_alert_identity_and_reopen_transition_notify_immediately(tmp_path: Path) -> None:
    from train_guard.reliability import ReliabilityEngine, event_fingerprint
    from train_guard.rules import RuleConfig, RuleEngine

    loss = Event(
        "run",
        EventKind.NAN_INF,
        Severity.CRITICAL,
        "metric is not finite",
        attributes={"metric": "loss"},
    )
    grad = Event(
        "run",
        EventKind.NAN_INF,
        Severity.CRITICAL,
        "metric is not finite",
        attributes={"metric": "grad_norm"},
    )
    assert event_fingerprint(loss) != event_fingerprint(grad)

    sink = _Sink()
    with StateStore(tmp_path / "state.sqlite") as store:
        engine = ReliabilityEngine(
            store,
            rules=RuleEngine(RuleConfig(loss_spike_ratio=2.0)),
            sinks=(sink,),
            notification_every=99,
        )
        engine.evaluate("run", {"step": 1, "loss": 1.0}, now=1)
        opened = engine.evaluate("run", {"step": 2, "loss": 3.0}, now=2)
        assert opened.transitions[0].state == "opened"
        engine.evaluate("run", {"step": 3, "loss": 4.0}, now=3)
        reopened = engine.evaluate("run", {"step": 4, "loss": 20.0}, now=4)
        assert [item.state for item in reopened.transitions] == ["reopened"]
        assert len(reopened.notified) == 1
        assert len(sink.events) == 2


def test_default_run_ids_isolate_same_directory_and_explicit_id_resumes(
    tmp_path: Path,
) -> None:
    from train_guard.run import watch as commands

    output = tmp_path / "run"
    output.mkdir()
    samples = iter((_watch_sample(1), _watch_sample(2), _watch_sample(3)))
    with mock.patch.object(commands, "collect_watch_sample", side_effect=lambda *_: next(samples)):
        assert commands.cmd_run_watch({"once": True, "output_dir": str(output)}) == 0
        assert commands.cmd_run_watch({"once": True, "output_dir": str(output)}) == 0
        assert (
            commands.cmd_run_watch({"once": True, "output_dir": str(output), "run_id": "explicit"})
            == 0
        )
    records = [
        json.loads(line)
        for line in (output / "watch.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert records[0]["run_id"] != records[1]["run_id"]
    assert records[2]["run_id"] == "explicit"


def test_explicit_run_step_rollback_resets_persisted_state(tmp_path: Path) -> None:
    from train_guard.run import watch as commands

    output = tmp_path / "run"
    output.mkdir()
    trainer_state = output / "trainer_state.json"
    trainer_state.write_text('{"global_step": 10}', encoding="utf-8")
    samples = iter((_watch_sample(10), _watch_sample(1)))
    config = {"once": True, "output_dir": str(output), "run_id": "stable"}
    with mock.patch.object(commands, "collect_watch_sample", side_effect=lambda *_: next(samples)):
        assert commands.cmd_run_watch(config) == 0
        with StateStore(output / "train_guard_state.sqlite") as store:
            store.set_offset("stable", "training_log", 999)
            store.set_run_state("stable", "sentinel", True)
        trainer_state.write_text('{"global_step": 1}', encoding="utf-8")
        assert commands.cmd_run_watch(config) == 0
    with StateStore(output / "train_guard_state.sqlite") as store:
        assert store.get_run_state("stable", "sentinel") is None
        assert store.get_run_state("stable", "watch.last_step") == 1


def test_json_outputs_never_emit_nonstandard_numbers(tmp_path: Path) -> None:
    event = Event(
        "run",
        EventKind.NAN_INF,
        Severity.CRITICAL,
        "bad",
        attributes={"nan": float("nan"), "pos": float("inf"), "neg": float("-inf")},
    )
    encoded = event.to_json()
    parsed = json.loads(
        encoded,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
    )
    assert parsed["attributes"] == {"nan": "NaN", "neg": "-Infinity", "pos": "Infinity"}

    audit = AuditLog(tmp_path / "audit.jsonl")
    audit.append({"value": float("nan")})
    json.loads(
        (tmp_path / "audit.jsonl").read_text(encoding="utf-8"),
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
    )


def test_completion_and_eval_require_independent_reference_evidence(tmp_path: Path) -> None:
    from train_guard.run import commands

    run = tmp_path / "run"
    run.mkdir()
    (run / "trainer_state.json").write_text(
        '{"global_step": 1, "log_history": [{"loss": 1.0}]}', encoding="utf-8"
    )
    (run / "model.safetensors").write_bytes(b"weights")
    check = commands.run_run_check(run, training_type="full")
    assert check["overall_status"] == "WARN"
    assert "Insufficient completion evidence" in check["reasons"]

    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text('{"prediction": "answer"}\n', encoding="utf-8")
    report = run_eval({"predictions": str(predictions), "report_dir": str(tmp_path / "eval")})
    assert report["overall_status"] == "WARN"
    assert report["evaluation_mode"] == "prediction_only"


def test_captured_output_flood_and_termination_do_not_deadlock() -> None:
    payload = (
        "import sys; "
        "sys.stdout.write('o'*1500000); sys.stdout.flush(); "
        "sys.stderr.write('e'*1500000); sys.stderr.flush()"
    )
    process = ManagedProcess(ProcessSpec(sys.executable, ("-c", payload), capture_output=True))
    process.start()
    assert process.wait() == 0
    assert len(process.stdout) == 1_500_000
    assert len(process.stderr) == 1_500_000

    endless = ManagedProcess(
        ProcessSpec(
            sys.executable,
            ("-c", "import time; print('started', flush=True); time.sleep(30)"),
            capture_output=True,
        )
    )
    endless.start()
    time.sleep(0.1)
    assert endless.terminate(grace_seconds=1) != 0
    assert "started" in endless.stdout

    interrupted = ManagedProcess(ProcessSpec(sys.executable, ("-c", "import time; time.sleep(30)")))
    interrupted.start()
    assert interrupted._process is not None
    with (
        mock.patch.object(interrupted._process, "wait", side_effect=KeyboardInterrupt),
        mock.patch.object(interrupted, "terminate", return_value=-1) as terminate,
        pytest.raises(KeyboardInterrupt),
    ):
        interrupted.wait()
    terminate.assert_called_once_with()
    interrupted._process.kill()
    interrupted._process.wait()
