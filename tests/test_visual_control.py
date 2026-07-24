from __future__ import annotations

import json
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from train_guard.control import ControlRequest, ControlToken, bearer_token, origin_is_local
from train_guard.dashboard import create_server
from train_guard.dashboard.assets import JS
from train_guard.state import StateStore
from train_guard.status import build_status_snapshot
from train_guard.supervisor import ProcessSpec, SupervisionResult, supervise
from train_guard.tui import render_terminal_snapshot


def test_control_request_token_and_origin_contract() -> None:
    request = ControlRequest.create(
        "run-1",
        "validated_restart",
        now=10,
        ttl_seconds=5,
        command_id="command-1",
    )
    assert not request.expired(14.9)
    assert request.expired(15)
    assert request.to_dict()["action"] == "validated_restart"
    with pytest.raises(ValueError, match="unsupported"):
        ControlRequest.create("run-1", "shell")

    token = ControlToken("local-secret")
    assert token.verify("local-secret")
    assert not token.verify("wrong")
    assert bearer_token("Bearer local-secret") == "local-secret"
    assert bearer_token("Basic local-secret") == ""
    assert origin_is_local("http://127.0.0.1:8765", "localhost", 8765)
    assert not origin_is_local("https://remote.invalid", "localhost", 8765)


def test_shared_status_and_plain_tui_fallback(tmp_path: Path) -> None:
    with StateStore(tmp_path / "state.sqlite") as store:
        store.set_run_state("run-1", "lifecycle.phase", "running")
        store.record_sample("run-1", 1.0, {"step": 2, "loss": 0.5})
        store.record_checkpoint("run-1", "checkpoint-2", "valid")
        store.record_recovery("run-1", "validated_restart", "succeeded")
        store.register_managed_process(
            "run-1",
            123,
            "running",
            ("terminate", "validated_restart"),
        )
        snapshot = build_status_snapshot(store, "run-1").to_dict()
    text = render_terminal_snapshot(snapshot)
    assert "run=run-1" in text
    assert "phase=running" in text
    assert snapshot["latest_sample"]["metrics"]["loss"] == 0.5
    assert snapshot["series"]["loss"] == (0.5,)
    assert snapshot["checkpoints"][0]["name"] == "checkpoint-2"
    assert snapshot["recoveries"][0]["status"] == "succeeded"
    alert_text = render_terminal_snapshot(
        {
            **snapshot,
            "active_alerts": [
                {
                    "event": {
                        "severity": "warning",
                        "kind": "step_stall",
                        "message": "no progress",
                    }
                }
            ],
        }
    )
    assert "[WARNING] step_stall" in alert_text


def test_dashboard_javascript_escapes_values_and_normalizes_gpu_payload() -> None:
    assert "replace(/[&<>\"']/g" in JS
    assert "Array.isArray(gpuPayload?.gpus)" in JS
    actions = JS.split("const actions=", 1)[1].split(";", 1)[0]
    assert "checkpoint" not in actions
    assert "validated_restart" in actions


def test_dashboard_assets_status_and_read_only_control(tmp_path: Path) -> None:
    with StateStore(tmp_path / "state.sqlite") as store:
        store.set_run_state("run-1", "lifecycle.phase", "running")
        server = create_server(store, port=0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        port = int(server.server_address[1])
        try:
            for path, expected in (
                ("/", "Train Guard"),
                ("/assets/app.css", "--surface"),
                ("/assets/app.js", "/api/status"),
            ):
                with urllib.request.urlopen(
                    f"http://127.0.0.1:{port}{path}",
                    timeout=2,
                ) as response:
                    assert expected in response.read().decode("utf-8")
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/api/status?run_id=run-1",
                timeout=2,
            ) as response:
                payload = json.loads(response.read())
            assert payload["run_id"] == "run-1"
            assert payload["phase"] == "running"

            request = urllib.request.Request(
                f"http://127.0.0.1:{port}/api/commands",
                data=b'{"run_id":"run-1","action":"terminate"}',
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with pytest.raises(urllib.error.HTTPError) as error:
                urllib.request.urlopen(request, timeout=2)
            assert error.value.code == 403
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)


def test_dashboard_authorized_control_is_idempotent(tmp_path: Path) -> None:
    token = ControlToken("dashboard-secret")
    with StateStore(tmp_path / "state.sqlite") as store:
        store.register_managed_process("run-1", 123, "running", ("terminate", "pause"))
        server = create_server(
            store,
            port=0,
            enable_control=True,
            authorization=token,
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        port = int(server.server_address[1])
        body = b'{"run_id":"run-1","action":"terminate","command_id":"web-command"}'
        try:
            unauthorized = urllib.request.Request(
                f"http://127.0.0.1:{port}/api/commands",
                data=body,
                method="POST",
            )
            with pytest.raises(urllib.error.HTTPError) as error:
                urllib.request.urlopen(unauthorized, timeout=2)
            assert error.value.code == 401

            bad_origin = urllib.request.Request(
                f"http://127.0.0.1:{port}/api/commands",
                data=body,
                headers={
                    "Authorization": "Bearer dashboard-secret",
                    "Origin": "https://remote.invalid",
                },
                method="POST",
            )
            with pytest.raises(urllib.error.HTTPError) as error:
                urllib.request.urlopen(bad_origin, timeout=2)
            assert error.value.code == 403

            headers = {
                "Authorization": "Bearer dashboard-secret",
                "Origin": "http://127.0.0.1:0",
            }
            request = urllib.request.Request(
                f"http://127.0.0.1:{port}/api/commands",
                data=body,
                headers=headers,
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=2) as response:
                assert response.status == 202
            duplicate = urllib.request.Request(
                f"http://127.0.0.1:{port}/api/commands",
                data=body,
                headers=headers,
                method="POST",
            )
            with urllib.request.urlopen(duplicate, timeout=2) as response:
                assert response.status == 200
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)


def test_control_queue_is_managed_idempotent_and_expires(tmp_path: Path) -> None:
    with StateStore(tmp_path / "state.sqlite") as store:
        request = ControlRequest.create(
            "run-1",
            "terminate",
            now=10,
            ttl_seconds=5,
            command_id="same-command",
        )
        with pytest.raises(ValueError, match="managed"):
            store.enqueue_control(request)
        store.register_managed_process("run-1", 123, "running", ("terminate", "pause"))
        with pytest.raises(ValueError, match="unavailable"):
            store.enqueue_control(ControlRequest.create("run-1", "resume"))
        assert store.enqueue_control(request)
        assert not store.enqueue_control(request)
        claimed = store.claim_control("run-1", 11)
        assert claimed is not None
        assert claimed["action"] == "terminate"
        store.complete_control("same-command", "succeeded", {"exit_code": 0})

        expired = ControlRequest.create(
            "run-1",
            "pause",
            now=20,
            ttl_seconds=1,
            command_id="expired-command",
        )
        assert store.enqueue_control(expired)
        assert store.claim_control("run-1", 22) is None


def test_supervisor_consumes_only_managed_control_requests(tmp_path: Path) -> None:
    with StateStore(tmp_path / "state.sqlite") as store:
        result: dict[str, SupervisionResult] = {}

        def run() -> None:
            result["value"] = supervise(
                ProcessSpec(
                    sys.executable,
                    ("-c", "import time; time.sleep(30)"),
                ),
                run_id="run-1",
                control_store=store,
                control_enabled=True,
            )

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        deadline = time.time() + 5
        while not store.managed_process("run-1") and time.time() < deadline:
            time.sleep(0.02)
        assert store.managed_process("run-1")["status"] == "running"
        assert store.enqueue_control(ControlRequest.create("run-1", "terminate", ttl_seconds=10))
        thread.join(timeout=8)
        assert not thread.is_alive()
        supervision = result["value"]
        assert supervision.stopped_reason == "control_terminate"
        assert next(store.recovery_history("run-1"))["status"] == "succeeded"
